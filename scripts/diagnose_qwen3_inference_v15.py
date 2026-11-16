"""Official8B fixed-input GEMV/split-attention numerical isolation; no qualification."""
import argparse,json,torch
from pathlib import Path
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from qwen3_triton.inference import TritonQwen3Inference
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from verify_qwen3_inference_v15 import metrics

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
 torch.set_num_threads(2)
 with stable_lease([1]):
  torch.manual_seed(20261116);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
  model='/data1/tzh/models/Qwen/Qwen3-8B';c=TritonQwen3Inference.from_pretrained(model)
  hf=AutoModelForCausalLM.from_pretrained(model,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().cuda()
  torch.randint(1,100000,(1,21),device='cuda');ids=torch.randint(1,100000,(1,69),device='cuda');prefix=ids[:,:65].contiguous()
  hc=DynamicCache();refs=[hf(prefix,past_key_values=hc,use_cache=True,logits_to_keep=1).logits]
  for j in range(4):refs.append(hf(ids[:,65+j:66+j].contiguous(),past_key_values=hc,use_cache=True,logits_to_keep=1).logits)
  rows=[]
  for gemv,split in [(True,True),(False,True),(True,False),(False,False)]:
   c.gemv=gemv;c.split_decode=split;cache=c.new_cache(1,80)
   out=[c(prefix,past_key_values=cache).logits]
   for j in range(4):out.append(c(ids[:,65+j:66+j].contiguous(),past_key_values=cache).logits)
   row=dict(gemv=gemv,split_decode=split,comparisons=[metrics(x,y) for x,y in zip(out,refs)])
   rows.append(row);print(json.dumps(row),flush=True)
  Path(a.output).write_text(json.dumps(dict(complete=True,diagnostic_only=True,rows=rows),indent=2)+'\n')
if __name__=='__main__':main()
