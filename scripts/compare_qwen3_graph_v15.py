"""Monitored paired HF/Triton eager and graph fixed-budget request comparison.

Qualification is bound per route to a complete matching report. Greedy, no earlyEOS.
Both graphs include prefill and exactly8output tokens for the same65-token input.
"""
import argparse,hashlib,json,os,statistics,subprocess,threading,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from qwen3_triton.inference import TritonQwen3Inference,CapturedGreedyRequest
from qwen3_triton.benchmark_utils.stable_idle import stable_lease

@torch.inference_mode()
def hf_greedy(model,ids,n):
 cache=DynamicCache();current=ids;pieces=[ids]
 for _ in range(n):
  logits=model(current,past_key_values=cache,use_cache=True,logits_to_keep=1).logits
  current=logits[:,-1,:].argmax(-1,keepdim=True);pieces.append(current)
 return torch.cat(pieces,dim=1)

class HFCaptured:
 @torch.inference_mode()
 def __init__(self,model,ids,n):
  self.input=ids.clone();self.stream=torch.cuda.Stream();self.stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(self.stream):
   for _ in range(2):hf_greedy(model,self.input,n)
  torch.cuda.current_stream().wait_stream(self.stream)
  self.graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(self.graph,stream=self.stream):self.output=hf_greedy(model,self.input,n)
 @torch.inference_mode()
 def __call__(self,ids):
  self.input.copy_(ids);self.graph.replay();return self.output

def reference_evidence(path,root,*,prefill_tiling):
 report_path=Path(path).resolve();record=json.loads(report_path.read_text())
 assert record.get('complete') and record.get('qualified') and record.get('functional_qualified') and record.get('corpus_numerics'),'A complete passing qualification report is required'
 assert record['reference_accumulation'] and record['prefill_tiling']==prefill_tiling and not record['sliced_decode'],'Qualified route options differ'
 assert record['model']=='Qwen/Qwen3-8B' and record['parameter_count']==8190735360
 assert not record['foreign_GPU_processes'] and record['corpus_protocol']['full_validation']
 current={str(p.relative_to(root)) for p in (root/'qwen3_triton').rglob('*.py')}
 assert current.issubset(record['source_sha256']),'Qualification omits current model sources'
 for name,digest in record['source_sha256'].items():
  source=(root/name).resolve();assert source.is_relative_to(root)
  assert hashlib.sha256(source.read_bytes()).hexdigest()==digest,source
 for phase in ['setup','prefill17','prefill','decode','generate']:
  proof=record['traces'][phase];raw=report_path.parent/(phase+'.json.gz')
  assert proof['passed'] and hashlib.sha256(raw.read_bytes()).hexdigest()==proof['sha256'],phase
 identity=Path('/data1/tzh/proj/qwen_operator_lab/artifacts/official_qwen3_inputs_v1.json')
 assert hashlib.sha256(identity.read_bytes()).hexdigest()==record['input_identity_sha256']
 return dict(qualified=True,report=str(report_path),sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),reference_accumulation=True,prefill_tiling=prefill_tiling)


@torch.inference_mode()
def run(a,out,foreign):
 torch.set_num_threads(2);torch.manual_seed(42)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 root=Path(__file__).resolve().parents[1];files=[*sorted((root/'qwen3_triton').rglob('*.py')),Path(__file__)]
 hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 proof=reference_evidence(a.qualification,root,prefill_tiling=a.reference_prefill_tiling) if a.include_reference else None
 model='/data1/tzh/models/Qwen/Qwen3-8B'
 c=TritonQwen3Inference.from_pretrained(model)
 h=AutoModelForCausalLM.from_pretrained(model,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().cuda()
 ids=torch.randint(1,100000,(1,65),device='cuda');n=8
 def route(*,sliced=False,reference=False):
  c.sliced_decode=sliced;c.reference_accumulation=reference;c.prefill_tiling=reference and a.reference_prefill_tiling
 def candidate_eager(x,*,sliced=False,reference=False):
  route(sliced=sliced,reference=reference)
  return c.generate(x,max_new_tokens=n,eos_token_id=[],pad_token_id=0)
 cg=CapturedGreedyRequest(c,ids,max_new_tokens=n);hg=HFCaptured(h,ids,n)
 funcs={'HF_eager':lambda x:hf_greedy(h,x,n),'HF_graph':hg,'Triton_eager':lambda x:candidate_eager(x),'Triton_graph':cg}
 if a.include_sliced:
  route(sliced=True);sg=CapturedGreedyRequest(c,ids,max_new_tokens=n);route()
  funcs.update(Triton_sliced_eager=lambda x:candidate_eager(x,sliced=True),Triton_sliced_graph=sg)
 if a.include_reference:
  route(reference=True);rg=CapturedGreedyRequest(c,ids,max_new_tokens=n);route()
  funcs.update(Triton_reference_eager=lambda x:candidate_eager(x,reference=True),Triton_reference_graph=rg)
 matches=[]
 for j in range(3):
  x=torch.randint(1,100000,(1,65),device='cuda')
  ce=funcs['Triton_eager'](x);he=funcs['HF_eager'](x)
  row=dict(case=j,Triton_graph_equal=bool(torch.equal(ce,cg(x))),HF_graph_equal=bool(torch.equal(he,hg(x))),HF_vs_Triton_equal=bool(torch.equal(ce,he)))
  if a.include_sliced:
   se=funcs['Triton_sliced_eager'](x)
   row.update(Triton_sliced_graph_equal=bool(torch.equal(se,sg(x))),HF_vs_Triton_sliced_equal=bool(torch.equal(se,he)),Triton_default_vs_sliced_equal=bool(torch.equal(ce,se)))
   assert row['Triton_sliced_graph_equal'],row
  if a.include_reference:
   re=funcs['Triton_reference_eager'](x)
   row.update(Triton_reference_graph_equal=bool(torch.equal(re,rg(x))),HF_vs_Triton_reference_equal=bool(torch.equal(re,he)),reference_output_tokens=re[0,-n:].cpu().tolist(),HF_output_tokens=he[0,-n:].cpu().tolist())
   assert row['Triton_reference_graph_equal'] and row['HF_vs_Triton_reference_equal'],row
  matches.append(row);assert row['Triton_graph_equal'] and row['HF_graph_equal'],row
 samples=[]
 for order,names in [(1,list(funcs)),(2,list(reversed(funcs)))]:
  for name in names:
   values=[]
   for repeat in range(7):
    torch.cuda.synchronize();start=time.perf_counter();funcs[name](ids);torch.cuda.synchronize();elapsed=time.perf_counter()-start
    if repeat>=2:values.append(elapsed)
   samples.append(dict(order=order,implementation=name,median_seconds=statistics.median(values),seconds=values))
 for p in files:assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(root))],p
 if proof:assert hashlib.sha256(Path(proof['report']).read_bytes()).hexdigest()==proof['sha256']
 result=dict(complete=True,qualification_by_implementation=dict(Triton_default=False,Triton_sliced=False,Triton_reference=bool(proof)),reference_qualification=proof,include_sliced=a.include_sliced,include_reference=a.include_reference,reference_prefill_tiling=a.reference_prefill_tiling,physical_GPU=1,GPU=torch.cuda.get_device_name(),torch=torch.__version__,CUDA=torch.version.cuda,
  prompt_tokens=65,output_tokens=8,batch=1,eos_policy='fixed token budget, no early stop in any arm',comparisons=matches,latencies=samples,foreign_context_observations=list(foreign),resource_clean=not foreign,
  source_sha256=hashes,scope='Matched sameGPU two-order full requests. Each graph compared with own eager path on3changed inputs. Reference quality is bound to matching complete validation/source/options; default and sliced routes remain unqualified.',limitations='SingleA6000/fixedB1S65/8tokens; no model or graph preparation cost, no serving throughput or universal quality equivalence. Both model instances resident simultaneously.')
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--include-sliced',action='store_true');p.add_argument('--include-reference',action='store_true');p.add_argument('--reference-prefill-tiling',action='store_true');p.add_argument('--qualification',type=Path);a=p.parse_args();
 if a.include_reference and a.qualification is None:p.error('--include-reference requires --qualification')
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1','Explicit physical GPU1 mapping required';out=a.output.resolve()
 assert out.is_relative_to(Path('/data1/tzh')) and not out.exists();out.mkdir(parents=True)
 foreign=[];stop=threading.Event()
 with stable_lease([1]):
  uuid=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
  def monitor():
   while not stop.wait(.5):
    try:
     raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=10)
     for line in raw.splitlines():
      fields=line.split(',')
      if fields[0].strip()==uuid and int(fields[1])!=os.getpid():foreign.append(line)
    except Exception as exc:foreign.append(repr(exc))
  thread=threading.Thread(target=monitor,daemon=True);thread.start()
  try:run(a,out,foreign)
  except BaseException as exc:(out/'failure.json').write_text(json.dumps(dict(error=repr(exc),foreign_context_observations=foreign),indent=2)+'\n');raise
  finally:stop.set();thread.join()
if __name__=='__main__':main()
