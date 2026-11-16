"""Small independent numerical/cache/generation checks; never claims8B qualification."""
import argparse,json,torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from qwen3_triton.inference import TritonQwen3Inference
from qwen3_triton.inference import kernels
from qwen3_triton.benchmark_utils.stable_idle import stable_lease


def error(a,b):
    x=a.float();y=b.float()
    return dict(max_abs=float((x-y).abs().max()),relative_l2=float(torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(y).clamp_min(1e-8)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(42)
    with stable_lease([0]),torch.inference_mode():
        checks=[]
        for b,s,start in [(1,17,0),(2,3,15),(1,1,0),(2,1,31),(1,1,256),(1,1,513)]:
            q=torch.randn(b,s,4,128,device='cuda',dtype=torch.bfloat16);ck=torch.randn(b,2,start+s+5,128,device='cuda',dtype=torch.bfloat16);cv=torch.randn_like(ck)
            mask=torch.arange(start+s,device='cuda')[None,:]<=start+torch.arange(s,device='cuda')[:,None]
            ref=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2).float(),ck[:,:,:start+s].float().repeat_interleave(2,1),cv[:,:,:start+s].float().repeat_interleave(2,1),attn_mask=mask).transpose(1,2)
            from qwen3_triton.runtime.dispatch import TritonRuntimeMode
            with TritonRuntimeMode():out=kernels.attention(q,ck,cv,start)
            e=error(out,ref);assert e['max_abs']<.04 and e['relative_l2']<.02,(b,s,start,e)
            checks.append(dict(kind='attention',batch=b,sequence=s,start=start,**e))
        c=Qwen3Config(vocab_size=257,hidden_size=256,intermediate_size=512,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=128,max_position_embeddings=1024,eos_token_id=256,pad_token_id=0)
        ref=Qwen3ForCausalLM(c).to(torch.bfloat16).eval();model=TritonQwen3Inference(c).load_hf_state_dict(ref.state_dict())
        with model.runtime:model.cuda()
        ref.cuda();ids=torch.randint(1,255,(2,19),device='cuda');cache=model.new_cache(2,32)
        full=model(ids,logits_to_keep=0).logits
        chunks=[]
        for lo,hi in [(0,3),(3,18),(18,19)]:chunks.append(model(ids[:,lo:hi].contiguous(),past_key_values=cache,logits_to_keep=0).logits)
        cached=torch.cat(chunks,1);expected=ref(ids,use_cache=False).logits
        for tag,v in [('full',full),('chunked',cached)]:
            e=error(v,expected);assert e['max_abs']<.06 and e['relative_l2']<.04,(tag,e);checks.append(dict(kind=tag,**e))
        assert cache.length==19
        e=error(cached,full);assert e['relative_l2']<.04;checks.append(dict(kind='cache_vs_recompute',**e))
        cache.reset();assert cache.length==0 and cache.valid
        replay=model(ids,past_key_values=cache,logits_to_keep=0).logits;assert torch.equal(replay,full)
        try:model(ids,past_key_values=model.new_cache(2,1))
        except ValueError:pass
        else:raise AssertionError('capacity guard')
        other=TritonQwen3Inference(c)
        try:other._forward(ids,cache,1)
        except ValueError:pass
        else:raise AssertionError('foreign cache guard')
        bad=ids.clone();bad[0,0]=-1
        try:model(bad)
        except ValueError:pass
        else:raise AssertionError('token bounds guard')
        # Reference greedy path and full-recompute candidate agree for this fixture.
        prompt=ids[:1,:5].contiguous();actual=model.generate(prompt,max_new_tokens=6,eos_token_id=[],pad_token_id=0)
        expected=prompt
        for _ in range(6):expected=torch.cat([expected,ref(expected,use_cache=False).logits[:,-1,:].argmax(-1,keepdim=True)],1)
        assert torch.equal(actual,expected),(actual,expected)
        assert torch.equal(model.generate(prompt,max_new_tokens=0),prompt)
        first=model(prompt).logits[:,-1,:].argmax(-1).item()
        ended=model.generate(prompt,max_new_tokens=6,eos_token_id=[first]);assert ended.shape[1]==prompt.shape[1]+1
        # Force different completion states and verify lowest-index tie + padding.
        logits=torch.zeros(2,257,device='cuda',dtype=torch.bfloat16);logits[0,8]=4;logits[1,11]=4
        out=torch.zeros(2,3,device='cuda',dtype=torch.int64);done=torch.tensor([True,False],device='cuda');next_token=torch.empty(2,1,device='cuda',dtype=torch.int64)
        with model.runtime:kernels.select(logits,out,next_token,done,1,(11,),0)
        assert next_token.cpu().tolist()==[[0],[11]] and done.all()
        result=dict(complete=True,small_fixture_only=True,checks=checks,cache_reset=True,capacity_guard=True,foreign_cache_guard=True,token_bounds_guard=True,greedy_matches_HF=True,zero_tokens=True,EOS=True,mixed_finished_rows=True,strict_GPU_dispatch=model.runtime.report(),limitations='Two-layer random fixture, not official8B verification or timing.')
        from pathlib import Path
        Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(complete=True,checks=len(checks))))


if __name__=='__main__':main()
