"""Controlled three-term BF16 dots to distinguish native accumulation schedules.

Each output has +2**30, -2**30 and a unit marker at selected K positions.
Exact arithmetic gives one (zero for markers overwritten by the large pair).
FP32 accumulation may lose the marker, exposing order/partition information.
This diagnoses arithmetic schedules, not model correctness or performance.
"""
import argparse,json,os,subprocess,threading
from pathlib import Path
import torch,triton
from qwen3_triton.inference import kernels
from qwen3_triton.kernels.gemm_experiments import _reduce_split
from scripts.diagnose_qwen3_sliced_coverage_v15 import _probe_interleaved
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from scripts.verify_qwen3_inference_v15 import sha
ROOT=Path(__file__).resolve().parents[1]

def bits(values):
 flags=(values==1).cpu().tolist();packed=bytearray((len(flags)+7)//8)
 for i,flag in enumerate(flags):
  if flag:packed[i//8]|=1<<(i%8)
 return packed.hex()

@torch.inference_mode()
def run(out,foreign):
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 files=[*sorted((ROOT/'qwen3_triton').rglob('*.py')),Path(__file__),ROOT/'scripts/diagnose_qwen3_sliced_coverage_v15.py']
 hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
 coverage_path=ROOT/'results/qwen3_triton_inference_v15_sliced_shape_coverage_v1.json';coverage=json.loads(coverage_path.read_text())
 rows=[];native=[];n=k=4096
 configs=[(1,16),(2,16),(2,32),(2,64),(4,16),(4,32)]
 for m in [1,17,65,512]:
  x=torch.ones((1,m,k),device='cuda',dtype=torch.bfloat16);w=torch.zeros((n,k),device='cuda',dtype=torch.bfloat16);indices=torch.arange(n,device='cuda')
  profile=None
  pairs=[(0,15),(0,16),(0,32),(0,64),(16,48),(0,k//2),(k-32,k-1)]
  for pair_index,(positive,negative) in enumerate(pairs):
   w.zero_();w[indices,indices]=1;w[:,positive]=2.**30;w[:,negative]=-(2.**30)
   reference=torch.nn.functional.linear(x,w)
   if pair_index==0:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
     for _ in range(64):kernels.positions(1,1,0,torch.device('cuda'))
     torch.cuda.synchronize()
     for _ in range(4):repeated=torch.nn.functional.linear(x,w)
     torch.cuda.synchronize()
    assert torch.equal(reference,repeated)
    profile={e.key:e.count for e in prof.key_averages() if any(t in e.key.lower() for t in ['gemm','gemv','xmma'])}
    phase='decode_17' if m==1 else 'prefill_'+str(m)
    expected=coverage['native_kernels_by_module'][phase+':model.layers.0.self_attn.q_proj']
    native.append(dict(m=m,kernels=profile,original_expected=expected,kernel_identity_matches_original=set(profile)==set(expected)))
   assert not bool(((reference!=0)&(reference!=1)).any()),'Three-term native output is not binary'
   base=dict(m=m,n=n,k=k,positive_K=positive,negative_K=negative,native_ones=int((reference[0,0]==1).sum()),native_row_variation=int((reference!=reference[:,0:1,:]).sum()),native_first_row_one_bits_lsb_first_hex=bits(reference[0,0]))
   outputs=[('current_default',kernels.linear(x,w))]
   for splits,chunk in configs:
    partial=torch.empty((splits,m,n),device='cuda',dtype=torch.float32);actual=torch.empty_like(reference)
    _probe_interleaved[(triton.cdiv(m,16)*triton.cdiv(n,32),splits)](x,w,partial,m,n,k,splits,chunk,num_warps=4,num_stages=2)
    _reduce_split[(triton.cdiv(m*n,512),)](partial,actual,m*n,splits,512)
    outputs.append((f'interleaved_{splits}x{chunk}',actual))
   for algorithm,actual in outputs:
    assert not bool(((actual!=0)&(actual!=1)).any()),algorithm
    mismatch=(actual!=reference);first=torch.nonzero(mismatch[0,0]).flatten()[:64].cpu().tolist()
    row=dict(**base,algorithm=algorithm,unequal_to_native=int(mismatch.sum()),first_row_unequal_columns=first,ones=int((actual[0,0]==1).sum()),first_row_one_bits_lsb_first_hex=bits(actual[0,0]))
    rows.append(row);print(json.dumps({key:value for key,value in row.items() if 'hex' not in key and key!='first_row_unequal_columns'}),flush=True)
 for p in files:assert sha(p)==hashes[str(p.relative_to(ROOT))],p
 result=dict(complete=True,diagnostic_only=True,production_changed=False,physical_GPU=1,GPU=torch.cuda.get_device_name(),torch=torch.__version__,CUDA=torch.version.cuda,native_kernel_identities=native,rows=rows,source_sha256=hashes,expected_native_report_sha256=sha(coverage_path),foreign_context_observations=list(foreign),resource_clean=not foreign,limitations='Synthetic cancellation probes distinguish schedules only on these sparse inputs. Native algorithm names must match original operand study before attribution. No speed, full-model precision, or general algorithm-identity claim.')
 raw=json.dumps(result,indent=2)+'\n';assert len(raw.encode())<=4*2**20;(out/'result.json').write_text(raw)
 print(json.dumps(dict(complete=True,rows=len(rows),native_kernel_matches=[v['kernel_identity_matches_original'] for v in native],resource_clean=not foreign)),flush=True)

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
