import pytest
import torch
from ops.reference.frequency_sparse import contextual_agreement, proxy_scores as reference_scores, selected_attention
from ops.triton.frequency_sparse import pack_proxy,proxy_scores,indexed_attention,validate_pairs


def test_split_half_reference():
    # Construct two keys: only the BACK HALF of pair 0 determines ranking.
    q=torch.zeros(1,2,128)
    k=torch.zeros(1,2,1,128)
    q[:,:,64]=1
    k[0,:,0,64]=torch.tensor([3.,-2.])
    pairs=torch.tensor([[0],[0]])
    torch.testing.assert_close(reference_scores(q,k,pairs),torch.tensor([[[3.,-2.],[3.,-2.]]]))
    agreement=contextual_agreement(q,k,top_k=1)
    assert agreement.shape==(2,64) and torch.equal(agreement[:,0],torch.ones(2))


cuda=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@cuda
@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float32])
@pytest.mark.parametrize('batch,heads,kv,seq,f,budget',[(1,16,8,129,16,65),(2,4,1,513,32,128),(1,4,2,17,8,17)])
def test_pack_scores_indexed_attention(dtype,batch,heads,kv,seq,f,budget):
    torch.manual_seed(71)
    with torch.inference_mode():
        cap=seq+11
        # Nonzero storage offsets and noncontiguous cache/query views.
        k=torch.randn(batch,cap+2,kv,128,device='cuda',dtype=dtype)[:,1:cap+1]
        v=torch.randn_like(k)
        q=torch.randn(batch,heads,2,128,device='cuda',dtype=dtype)[:,:,1]
        pairs=torch.stack([torch.randperm(64,device='cuda')[:f] for _ in range(heads)]).int()
        validate_pairs(pairs,heads)
        proxy=torch.full((batch,heads,cap,2*f),float('nan'),device='cuda',dtype=dtype)
        used=torch.full((batch,),seq,device='cuda',dtype=torch.int32)
        positions=torch.arange(seq,device='cuda')
        pack_proxy(k,pairs,positions,proxy)
        scores=proxy_scores(q,proxy,pairs,used)
        torch.testing.assert_close(scores[:,:,:seq],reference_scores(q,k[:,:seq],pairs),atol=3e-5,rtol=3e-5)
        assert torch.isneginf(scores[:,:,seq:]).all()
        assert torch.isnan(proxy[:,:,seq:]).all()
        indices=scores.topk(budget,sorted=False).indices
        expected=selected_attention(q,k,v,indices,scale=.12)
        for splits in (1,4,16):
            out=indexed_attention(q,k,v,indices,used,scale=.12,splits=splits)
            torch.testing.assert_close(out.float(),expected,atol=.008,rtol=.02)


@cuda
def test_full_budget_graph_dynamic_lengths_and_poisoned_tail():
    torch.manual_seed(72)
    with torch.inference_mode():
        b,h,hk,cap=2,4,2,256
        q=torch.randn(b,h,128,device='cuda',dtype=torch.float16)
        k=torch.randn(b,cap,hk,128,device='cuda',dtype=q.dtype)
        v=torch.randn_like(k)
        k[:,129:]=float('nan')
        v[:,129:]=float('nan')
        pairs=torch.arange(64,device='cuda',dtype=torch.int32).expand(h,-1).contiguous()
        proxy=torch.full((b,h,cap,128),float('nan'),device='cuda',dtype=q.dtype)
        pack_proxy(k,pairs,torch.arange(129,device='cuda'),proxy)
        used=torch.tensor([129,33],device='cuda',dtype=torch.int32)
        def fn():
            scores=proxy_scores(q,proxy,pairs,used)
            indices=scores.topk(cap,sorted=False).indices
            return indexed_attention(q,k,v,indices,used,splits=4)
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): out=fn()
        for lengths in ([129,33],[17,128],[1,9]):
            used.copy_(torch.tensor(lengths,device='cuda',dtype=torch.int32))
            graph.replay()
            for bi,n in enumerate(lengths):
                indices=torch.arange(n,device='cuda').expand(1,h,-1).contiguous()
                expected=selected_attention(q[bi:bi+1],k[bi:bi+1],v[bi:bi+1],indices)
                torch.testing.assert_close(out[bi:bi+1].float(),expected,atol=.003,rtol=.01)


@cuda
def test_model_full_budget_and_restore():
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from model.hf_inference import qwen3_triton_inference
    from model.static_decode import StaticDecodeRunner
    from model.frequency_sparse import FrequencySparseRunner
    c=Qwen3Config(vocab_size=256,hidden_size=256,intermediate_size=512,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=128,rope_theta=1000000.)
    c._attn_implementation='sdpa'
    torch.manual_seed(73)
    model=Qwen3ForCausalLM(c).cuda().half().eval()
    ids=torch.randint(256,(1,21),device='cuda')
    pairs=torch.arange(16,device='cuda').expand(2,4,-1).contiguous()
    with torch.inference_mode():
        before=model(ids,use_cache=False).logits
        with qwen3_triton_inference(model,fuse_qk=True,compact_gqa=True,static_decode=True,
                                   direct_k_cache=True,direct_v_cache=True):
            dense=StaticDecodeRunner(model,1,24)
            sparse=FrequencySparseRunner(model,1,24,pairs=pairs,budget=24)
            a,b=dense.prefill(ids[:,:17]),sparse.prefill(ids[:,:17])
            torch.testing.assert_close(a,b,atol=0,rtol=0)
            dense.capture(ids[:,17:18]); sparse.capture(ids[:,17:18])
            for pos in range(17,21):
                a=dense.decode(ids[:,pos:pos+1]); b=sparse.decode(ids[:,pos:pos+1])
                torch.testing.assert_close(a,b,atol=.004,rtol=.01)
        torch.testing.assert_close(before,model(ids,use_cache=False).logits,atol=0,rtol=0)
        assert '_update_causal_mask' not in model.model.__dict__


def test_bad_pair_contract():
    with pytest.raises(ValueError,match='Distinct'):
        validate_pairs(torch.zeros(2,16,dtype=torch.int32),2)
    with pytest.raises(ValueError,match='F='):
        validate_pairs(torch.arange(7).expand(2,-1).contiguous(),2)
