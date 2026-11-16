"""Isolate cached attention/RoPE on identical officialHF Q/K/V; no qualification."""
import argparse,json,torch,math,struct,triton
from pathlib import Path
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from qwen3_triton.inference import kernels
from qwen3_triton.benchmark_utils.stable_idle import stable_lease

def stats(a,b):
 d=a.float()-b.float();return dict(max_abs=float(d.abs().max()),mean_abs=float(d.abs().mean()),rel=float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(b.float()).clamp_min(1e-12)),unequal=int((a!=b).sum()))
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
 torch.set_num_threads(2)
 with stable_lease([1]):
  torch.manual_seed(20261116);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
  hf=AutoModelForCausalLM.from_pretrained('/data1/tzh/models/Qwen/Qwen3-8B',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().cuda()
  backend_kernels={};rows=[];phase=['prefill'];positions={};state={};ks={};vs={};handles=[]
  def rotary(m,inp,out):positions['cos'],positions['sin']=out
  handles.append(hf.model.rotary_emb.register_forward_hook(rotary))
  for index,layer in enumerate(hf.model.layers):
   def save(tag,i):
    def hook(m,inp,out):state[(i,tag)]=out
    return hook
   for tag,m in [('q',layer.self_attn.q_norm),('k',layer.self_attn.k_norm),('v',layer.self_attn.v_proj)]:handles.append(m.register_forward_hook(save(tag,index)))
   def compare(i):
    def hook(m,inp):
     q=state[i,'q'].transpose(1,2);k=state[i,'k'].transpose(1,2);b,h,s,d=q.shape
     q,k=apply_rotary_pos_emb(q,k,positions['cos'],positions['sin']);v=state[i,'v'].view(b,s,8,d).transpose(1,2)
     start=0 if phase[0]=='prefill' else ks[i].shape[2]
     ks[i]=k if not start else torch.cat([ks[i],k],2);vs[i]=v if not start else torch.cat([vs[i],v],2)
     for mode in ['split','tile64','tile32_M64','tile32_M128']:
      query=q.transpose(1,2).contiguous();key=ks[i].contiguous();value=vs[i].contiguous()
      if mode in ['split','tile64']:
       out=kernels.attention(query,key,value,start,split_decode=mode=='split')
      else:
       out=torch.empty_like(query);m=128 if mode=='tile32_M128' else 64
       scale=struct.unpack('f',struct.pack('f',d**-.5))[0];sl2=struct.unpack('f',struct.pack('f',scale*math.log2(math.e)))[0]
       kernels._inference_cached_attention[(triton.cdiv(s,m),b*h)](query,key,value,out,s,start+s,start+s,h,8,d,start,sl2,m,32,num_warps=4,num_stages=1,enable_fp_fusion=False)
      rows.append(dict(phase=phase[0],layer=i,mode=mode,**stats(out.reshape(b,s,-1),inp[0])))
    return hook
   handles.append(layer.self_attn.o_proj.register_forward_pre_hook(compare(index)))
  torch.randint(1,100000,(1,21),device='cuda');ids=torch.randint(1,100000,(1,69),device='cuda');cache=DynamicCache()
  hf(ids[:,:65].contiguous(),past_key_values=cache,use_cache=True,logits_to_keep=1)
  for j in range(4):
   phase[0]='decode_'+str(j)
   if j==0:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
     hf(ids[:,65+j:66+j].contiguous(),past_key_values=cache,use_cache=True,logits_to_keep=1);torch.cuda.synchronize()
    backend_kernels={x.key:x.count for x in prof.key_averages() if any(tag in x.key.lower() for tag in ['flash','fmha','attention','softmax'])}
   else:hf(ids[:,65+j:66+j].contiguous(),past_key_values=cache,use_cache=True,logits_to_keep=1)
  for h in handles:h.remove()
  Path(a.output).write_text(json.dumps(dict(complete=True,diagnostic_only=True,backend_kernels=backend_kernels,rows=rows),indent=2)+'\n')
  print(json.dumps(dict(complete=True,backend_kernels=backend_kernels,worst=sorted(rows,key=lambda x:x['rel'],reverse=True)[:12])),flush=True)
if __name__=='__main__':main()
