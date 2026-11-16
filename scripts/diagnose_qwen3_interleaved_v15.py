"""Test interleaved warp-K accumulation against exact HF8B operands.

Reference-only probing. No default/production algorithm is changed.
"""
import argparse,json,torch,triton
import triton.language as tl
from pathlib import Path
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from qwen3_triton.inference import kernels
from qwen3_triton.kernels.gemm_experiments import _reduce_split
from qwen3_triton.benchmark_utils.stable_idle import stable_lease

@triton.jit
def _probe_interleaved(X,W,P,N:tl.constexpr,K:tl.constexpr,SPLITS:tl.constexpr,CHUNK:tl.constexpr):
 split=tl.program_id(1);row=tl.arange(0,16);col=tl.program_id(0)*32+tl.arange(0,32);kk=tl.arange(0,CHUNK)
 accum=tl.zeros((16,32),tl.float32)
 for block in range(tl.cdiv(K,SPLITS*CHUNK)):
  index=(block*SPLITS+split)*CHUNK+kk
  x=tl.load(X+index[None,:]+tl.zeros((16,1),tl.int32), (row[:,None]==0)&(index[None,:]<K),0)
  w=tl.load(W+col[None,:]*K+index[:,None],(col[None,:]<N)&(index[:,None]<K),0)
  accum=tl.dot(x,w,accum)
 tl.store(P+split*N+col[None,:]+row[:,None]*0,accum,(row[:,None]==0)&(col[None,:]<N))

def stat(y,ref):
 d=y.float()-ref.float();return dict(unequal=int((y!=ref).sum()),relative_l2=float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(ref.float()).clamp_min(1e-12)))

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(2)
 with stable_lease([1]):
  torch.manual_seed(20261116);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
  hf=AutoModelForCausalLM.from_pretrained('/data1/tzh/models/Qwen/Qwen3-8B',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().cuda()
  rows=[];phase=['prefill'];handles=[];native={}
  def hook(name):
   def check(module,inputs,ref):
    if phase[0]=='prefill':return
    x=inputs[0].contiguous();n,k=module.weight.shape
    if phase[0]=='decode_0':
     with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
      duplicate=torch.nn.functional.linear(x,module.weight);torch.cuda.synchronize()
     assert torch.equal(duplicate,ref),'Native repeated reference differs'
     native[name]={e.key:e.count for e in prof.key_averages() if any(t in e.key.lower() for t in ['gemm','gemv','xmma'])}
    rows.append(dict(phase=phase[0],name=name,algorithm='current_GEMV',**stat(kernels.linear(x,module.weight),ref)))
    for splits in [2,4]:
     for chunk in [16,32,64]:
      partial=torch.empty((splits,n),device='cuda',dtype=torch.float32);out=torch.empty_like(ref)
      _probe_interleaved[(triton.cdiv(n,32),splits)](x,module.weight,partial,n,k,splits,chunk,num_warps=4,num_stages=2)
      _reduce_split[(triton.cdiv(n,512),)](partial,out,n,splits,512)
      rows.append(dict(phase=phase[0],name=name,algorithm='interleaved_'+str(splits)+'x'+str(chunk),**stat(out,ref)))
   return check
  selected=['model.layers.0.self_attn.q_proj','model.layers.0.self_attn.k_proj','model.layers.0.mlp.gate_proj','model.layers.0.mlp.down_proj','model.layers.31.self_attn.q_proj']
  for name,module in hf.named_modules():
   if name in selected:handles.append(module.register_forward_hook(hook(name)))
  torch.randint(1,100000,(1,21),device='cuda');ids=torch.randint(1,100000,(1,69),device='cuda');cache=DynamicCache()
  hf(ids[:,:65].contiguous(),past_key_values=cache,use_cache=True,logits_to_keep=1)
  for j in range(4):phase[0]='decode_'+str(j);hf(ids[:,65+j:66+j].contiguous(),past_key_values=cache,use_cache=True,logits_to_keep=1)
  for h in handles:h.remove()
  d=dict(complete=True,diagnostic_only=True,native_kernels_by_module=native,rows=rows)
  Path(a.output).write_text(json.dumps(d,indent=2)+'\n')
  print(json.dumps(dict(complete=True,summary={algorithm:sum(x['unequal'] for x in rows if x['algorithm']==algorithm) for algorithm in sorted(set(x['algorithm'] for x in rows))})),flush=True)
if __name__=='__main__':main()
