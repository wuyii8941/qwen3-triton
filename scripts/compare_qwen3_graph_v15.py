"""Monitored paired HF/Triton eager and graph fixed-budget request comparison.

Quality against HF is separate and remains unqualified. Greedy, no earlyEOS.
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

@torch.inference_mode()
def run(a,out,foreign):
 torch.set_num_threads(2);torch.manual_seed(42)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 root=Path(__file__).resolve().parents[1];files=[*sorted((root/'qwen3_triton').rglob('*.py')),Path(__file__)]
 hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 model='/data1/tzh/models/Qwen/Qwen3-8B'
 c=TritonQwen3Inference.from_pretrained(model)
 h=AutoModelForCausalLM.from_pretrained(model,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().cuda()
 ids=torch.randint(1,100000,(1,65),device='cuda');n=8
 cg=CapturedGreedyRequest(c,ids,max_new_tokens=n);hg=HFCaptured(h,ids,n)
 funcs={'HF_eager':lambda x:hf_greedy(h,x,n),'HF_graph':hg,'Triton_eager':lambda x:c.generate(x,max_new_tokens=n,eos_token_id=[],pad_token_id=0),'Triton_graph':cg}
 matches=[]
 for j in range(3):
  x=torch.randint(1,100000,(1,65),device='cuda')
  ce=funcs['Triton_eager'](x);he=funcs['HF_eager'](x)
  row=dict(case=j,Triton_graph_equal=bool(torch.equal(ce,cg(x))),HF_graph_equal=bool(torch.equal(he,hg(x))),HF_vs_Triton_equal=bool(torch.equal(ce,he)))
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
 result=dict(complete=True,HF_numerical_qualified=False,physical_GPU=1,GPU=torch.cuda.get_device_name(),torch=torch.__version__,CUDA=torch.version.cuda,
  prompt_tokens=65,output_tokens=8,batch=1,eos_policy='fixed token budget, no early stop in any arm',comparisons=matches,latencies=samples,foreign_context_observations=list(foreign),resource_clean=not foreign,
  source_sha256=hashes,scope='Matched sameGPU two-order full requests. Each graph compared with own eager path on3changed inputs. No inference quality claim; original HF logit gates remain required.',limitations='SingleA6000/fixedB1S65/8tokens; no model or graph preparation cost, no serving throughput or quality equivalence. Both model instances resident simultaneously.')
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
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
