"""Research-only native-order reproduction on real8B weights and original gates.

Known global SplitK3/5/7 uses64-aligned contiguous partitions and FP32 forward
reduction. Intrawarp sliced paths use interleaved2x32. No native GPU fallback.
The public API is temporarily wrapped only in this diagnostic process.
Original trace audits are retained as-is and cannot qualify the new kernels.
"""
import argparse,contextlib,gc,json,os,subprocess,threading
from pathlib import Path
import torch,triton
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from qwen3_triton.inference import kernels
from qwen3_triton.inference import model as im
from qwen3_triton.runtime.memory import raw_empty
from qwen3_triton.kernels.gemm_experiments import _split_gemm
from scripts.diagnose_qwen3_sliced_coverage_v15 import _probe_interleaved
from scripts.probe_qwen3_native_splitk_v15 import _ordered_reduce
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from scripts import verify_qwen3_inference_v15 as verifier
ROOT=Path(__file__).resolve().parents[1]
ORIGINAL_LINEAR=kernels.linear
ORIGINAL_FUSED=im.gate_up_swiglu
ORIGINAL_SWIGLU=im.swiglu

def route(m,n,k):
 if m==1 and n==4096 and k in (4096,12288):return 'sliced_2x32'
 if m==17 and (n,k)==(1024,4096):return 'interleaved_2x32'
 if m==17 and (n,k)==(12288,4096):return 'split_3'
 if m==65:
  if (n,k)==(4096,4096):return 'split_3'
  if (n,k)==(1024,4096):return 'split_5'
  if (n,k)==(4096,12288):return 'split_7'
 return 'default'

def matched_linear(x,w,**kw):
 m=x.numel()//x.shape[-1];n,k=w.shape;algorithm=route(m,n,k)
 if algorithm=='default' or not x.is_contiguous() or not w.is_contiguous() or kw.get('accurate',False):return ORIGINAL_LINEAR(x,w,**kw)
 if algorithm=='sliced_2x32':return kernels.sliced_linear(x,w)
 splits=2 if algorithm=='interleaved_2x32' else int(algorithm.rsplit('_',1)[1])
 partial=raw_empty((splits,m,n),device=x.device,dtype=torch.float32);y=raw_empty((*x.shape[:-1],n),device=x.device,dtype=x.dtype)
 if algorithm=='interleaved_2x32':_probe_interleaved[(triton.cdiv(m,16)*triton.cdiv(n,32),splits)](x,w,partial,m,n,k,splits,32,num_warps=4,num_stages=2)
 else:_split_gemm[(triton.cdiv(m,16)*triton.cdiv(n,32),splits)](x,w,partial,m,n,k,k,1,1,k,16,32,64,1,splits,num_warps=4,num_stages=2)
 _ordered_reduce[(triton.cdiv(m*n,256),)](partial,y,m*n,splits,0,256,enable_fp_fusion=False)
 return y

@contextlib.contextmanager
def matched():
 def fused(x,wg,wu,**kw):
  if x.numel()//x.shape[-1]==17 and wg.shape==(12288,4096):return ORIGINAL_SWIGLU(matched_linear(x,wg),matched_linear(x,wu))
  return ORIGINAL_FUSED(x,wg,wu,**kw)
 kernels.linear=matched_linear;im.gate_up_swiglu=fused
 try:yield
 finally:kernels.linear=ORIGINAL_LINEAR;im.gate_up_swiglu=ORIGINAL_FUSED

@torch.inference_mode()
def local_checks():
 torch.manual_seed(20261116)
 h=AutoModelForCausalLM.from_pretrained('/data1/tzh/models/Qwen/Qwen3-8B',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().cuda()
 phase={};rows=[];handles=[]
 def hook(name):
  def check(module,inp,reference):
   x=inp[0].contiguous();n,k=module.weight.shape;m=x.numel()//k;algorithm=route(m,n,k)
   actual=matched_linear(x,module.weight);default=ORIGINAL_LINEAR(x,module.weight)
   rows.append(dict(**phase,name=name,m=m,n=n,k=k,algorithm=algorithm,baseline_unequal=int((default!=reference).sum()),matched_unequal=int((actual!=reference).sum()),max_abs=float((actual.float()-reference.float()).abs().max())))
  return check
 for name,module in h.named_modules():
  if isinstance(module,torch.nn.Linear) and name.startswith('model.layers.'):handles.append(module.register_forward_hook(hook(name)))
 for length in [17,65]:
  ids=torch.randint(1,100000,(1,length+4),device='cuda');cache=DynamicCache()
  for step in range(5):
   phase.update(prompt_length=length,step=step)
   current=ids[:,:length].contiguous() if step==0 else ids[:,length+step-1:length+step].contiguous()
   h(current,past_key_values=cache,use_cache=True,logits_to_keep=1)
   selected=[v for v in rows if v['prompt_length']==length and v['step']==step and v['algorithm']!='default']
   print(json.dumps(dict(local_phase=dict(phase),selected_calls=len(selected),baseline_unequal=sum(v['baseline_unequal'] for v in selected),matched_unequal=sum(v['matched_unequal'] for v in selected))),flush=True)
 for handle in handles:handle.remove()
 return rows

@torch.inference_mode()
def run(out,foreign):
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 files=[*sorted((ROOT/'qwen3_triton').rglob('*.py')),Path(__file__),ROOT/'scripts/verify_qwen3_inference_v15.py',ROOT/'scripts/diagnose_qwen3_sliced_coverage_v15.py',ROOT/'scripts/probe_qwen3_native_splitk_v15.py']
 hashes={str(p.relative_to(ROOT)):verifier.sha(p) for p in files}
 rows=local_checks();selected=[v for v in rows if v['algorithm']!='default']
 assert len(rows)==2520 and sum(v['baseline_unequal'] for v in selected)>0
 gc.collect();torch.cuda.empty_cache()
 full=out/'original_gates';full.mkdir()
 args=argparse.Namespace(model='/data1/tzh/models/Qwen/Qwen3-8B',identity='/data1/tzh/proj/qwen_operator_lab/artifacts/official_qwen3_inputs_v1.json',physical_device=1,prefill_tiling=False,sliced_decode=False,warmup_collection=True,diagnostic=True,corpus_blocks=8,corpus_sequence=128)
 with matched():verifier.run(args,full,foreign)
 report=json.loads((full/'result.json').read_text())
 for p in files:assert verifier.sha(p)==hashes[str(p.relative_to(ROOT))],p
 result=dict(complete=True,diagnostic_only=True,production_changed=False,physical_GPU=1,GPU=torch.cuda.get_device_name(),local_rows=rows,selected_local_calls=len(selected),selected_local_baseline_unequal=sum(v['baseline_unequal'] for v in selected),selected_local_matched_unequal=sum(v['matched_unequal'] for v in selected),original_numerical_gates_passed=report['numerics'],greedy_exact_match=report['greedy_exact_match'],original_verifier_report=dict(path='original_gates/result.json',sha256=verifier.sha(full/'result.json')),source_sha256=hashes,foreign_context_observations=list(foreign),resource_clean=not foreign,qualified=False,scope='Research-only matched_linear and M17gate/up wrappers override arithmetic during the recorded verifier run. Underlying model flags remainFalse; this outer source receipt includes all actual override kernels. Original raw trace audits are retained and reject the unregistered new kernel names/counts. Requires exported implementation and complete updated trace audit before qualification.')
 raw=json.dumps(result,indent=2)+'\n';assert len(raw.encode())<=4*2**20;(out/'result.json').write_text(raw)
 print(json.dumps({k:v for k,v in result.items() if k not in ['local_rows','source_sha256']}),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and out.is_relative_to(Path('/data1/tzh')) and not out.exists();out.mkdir(parents=True)
 foreign=[];stop=threading.Event()
 with stable_lease([1]):
  uuid=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
  def monitor():
   while not stop.wait(.5):
    try:
     raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=10)
     foreign.extend(line for line in raw.splitlines() if line.split(',')[0].strip()==uuid and int(line.split(',')[1])!=os.getpid())
    except Exception as exc:foreign.append(repr(exc))
  thread=threading.Thread(target=monitor,daemon=True);thread.start()
  try:run(out,foreign)
  except BaseException as exc:(out/'failure.json').write_text(json.dumps(dict(error=repr(exc),foreign_context_observations=foreign),indent=2)+'\n');raise
  finally:stop.set();thread.join()
if __name__=='__main__':main()
