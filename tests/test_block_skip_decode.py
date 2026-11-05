import math
import pytest
import torch
from ops.triton.block_skip_decode import block_skip_attention
from ops.reference.frequency_sparse import dense_scores,selected_attention

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float32])
@pytest.mark.parametrize('splits',[1,8,32])
def test_zero_threshold_is_dense(dtype,splits):
    torch.manual_seed(81)
    with torch.inference_mode():
        q=torch.randn(2,4,128,device='cuda',dtype=dtype)
        k=torch.randn(2,258,2,128,device='cuda',dtype=dtype)[:,1:]
        v=torch.randn_like(k)
        k[:,129:]=float('nan');v[:,129:]=float('nan')
        used=torch.tensor([129,17],device='cuda',dtype=torch.int32)
        out,stats=block_skip_attention(q,k,v,used,threshold=0,splits=splits)
        for bi,n in enumerate([129,17]):
            indices=torch.arange(n,device='cuda').expand(1,4,-1).contiguous()
            expected=selected_attention(q[bi:bi+1],k[bi:bi+1],v[bi:bi+1],indices)
            torch.testing.assert_close(out[bi:bi+1].float(),expected,atol=.005,rtol=.02)
        assert stats[...,0].sum()==0


def test_actual_skipping_matches_selected_reference():
    with torch.inference_mode():
        torch.manual_seed(82)
        q=torch.ones(1,2,128,device='cuda',dtype=torch.float16)
        k=torch.full((1,128,1,128),-1.,device='cuda',dtype=q.dtype)
        k[:,96:]=1.
        v=torch.randn_like(k)
        used=torch.tensor([128],device='cuda',dtype=torch.int32)
        out,stats=block_skip_attention(q,k,v,used,threshold=1e-4,splits=1,block_n=32)
        indices=torch.arange(96,128,device='cuda').expand(1,2,-1).contiguous()
        expected=selected_attention(q,k,v,indices)
        torch.testing.assert_close(out.float(),expected,atol=.002,rtol=.01)
        assert torch.equal(stats[...,0],torch.full_like(stats[...,0],3))
        assert torch.equal(stats[...,1],torch.full_like(stats[...,1],4))


def test_nonzero_threshold_random_split_oracle():
    with torch.inference_mode():
        torch.manual_seed(83)
        q=torch.randn(1,4,128,device='cuda',dtype=torch.float32)*3
        k=torch.randn(1,257,2,128,device='cuda')
        v=torch.randn_like(k)
        used=torch.tensor([257],device='cuda',dtype=torch.int32)
        threshold,block,splits=.5,32,2
        out,stats=block_skip_attention(q,k,v,used,threshold=threshold,block_n=block,splits=splits)
        scores=dense_scores(q,k)*128**-.5
        for head in range(4):
            selected=[];skipped=0
            for split in range(splits):
                mx=-float('inf')
                for tile in reversed(range(9*split//splits,9*(split+1)//splits)):
                    start,end=tile*block,min((tile+1)*block,257)
                    blockmax=float(scores[0,head,start:end].max())
                    mx=max(mx,blockmax)
                    if blockmax-mx>=math.log(threshold):selected.extend(range(start,end))
                    else:skipped+=1
            indices=torch.tensor(selected,device='cuda').view(1,1,-1)
            kvhead=head//2
            expected=selected_attention(q[:,head:head+1],k[:,:,kvhead:kvhead+1],v[:,:,kvhead:kvhead+1],indices)
            torch.testing.assert_close(out[:,head:head+1],expected,atol=3e-5,rtol=3e-5)
            assert stats[0,head,:,0].sum()==skipped


def test_graph_full_model_and_restore():
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from model.hf_inference import qwen3_triton_inference
    from model.static_decode import StaticDecodeRunner
    from model.block_skip import BlockSkipRunner
    c=Qwen3Config(vocab_size=128,hidden_size=256,intermediate_size=512,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=128,rope_theta=1000000.)
    c._attn_implementation='sdpa'
    torch.manual_seed(84)
    model=Qwen3ForCausalLM(c).cuda().half().eval()
    ids=torch.randint(128,(1,25),device='cuda')
    with torch.inference_mode():
        expected=model(ids,use_cache=False).logits
        with qwen3_triton_inference(model,fuse_qk=True,compact_gqa=True,static_decode=True,
                                   direct_k_cache=True,direct_v_cache=True):
            dense=StaticDecodeRunner(model,1,25)
            candidate=BlockSkipRunner(model,1,25,threshold=0)
            dense.prefill(ids[:,:17]);candidate.prefill(ids[:,:17])
            dense.capture(ids[:,17:18]);candidate.capture(ids[:,17:18])
            for pos in range(17,25):
                torch.testing.assert_close(candidate.decode(ids[:,pos:pos+1]),
                    dense.decode(ids[:,pos:pos+1]),atol=.003,rtol=.01)
        torch.testing.assert_close(model(ids,use_cache=False).logits,expected,atol=0,rtol=0)
