"""Compare graph/eager identical Triton greedy computation; HF quality is separate."""
import argparse,json,time,statistics,torch
from pathlib import Path
from transformers import Qwen3Config,Qwen3ForCausalLM
from qwen3_triton.inference import TritonQwen3Inference,CapturedGreedyRequest
from qwen3_triton.benchmark_utils.stable_idle import stable_lease

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--small',action='store_true');a=p.parse_args()
 torch.set_num_threads(2);torch.manual_seed(42)
 with stable_lease([1]):
  if a.small:
   c=Qwen3Config(vocab_size=257,hidden_size=256,intermediate_size=512,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=128,max_position_embeddings=1024)
   ref=Qwen3ForCausalLM(c).to(torch.bfloat16).eval();m=TritonQwen3Inference(c).load_hf_state_dict(ref.state_dict())
   with m.runtime:m.cuda()
   del ref
  else:m=TritonQwen3Inference.from_pretrained('/data1/tzh/models/Qwen/Qwen3-8B')
  ids=torch.randint(1,min(m.config.vocab_size,100000),(1,65),device='cuda');n=8
  captured=CapturedGreedyRequest(m,ids,max_new_tokens=n)
  matches=[]
  for j in range(3):
   x=torch.randint(1,min(m.config.vocab_size,100000),(1,65),device='cuda')
   eager=m.generate(x,max_new_tokens=n,eos_token_id=[],pad_token_id=0)
   replay=captured(x).clone();matches.append(torch.equal(eager,replay));assert matches[-1]
  result=[]
  for order,names in [(1,['eager','graph']),(2,['graph','eager'])]:
   for name in names:
    values=[]
    for repeat in range(7):
     torch.cuda.synchronize();start=time.perf_counter()
     out=captured(ids) if name=='graph' else m.generate(ids,max_new_tokens=n,eos_token_id=[],pad_token_id=0)
     torch.cuda.synchronize();elapsed=time.perf_counter()-start
     if repeat>=2:values.append(elapsed)
    result.append(dict(order=order,mode=name,median_seconds=statistics.median(values),samples=values))
  report=dict(complete=True,small_fixture_only=a.small,HF_numerical_qualification=False,graph_matches_eager=matches,prompt_tokens=65,generated_tokens=n,latencies=result,limitations='Graph vs same Triton implementation only; no HF quality claim, fixed shape/budget, no earlyEOS, one request object cannot be used concurrently.')
  Path(a.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
