"""CE-only chunking prototype: preserve the full native LM-head GEMM.

Diagnostic only. It retains BF16 logits/gradient, but only creates FP32 CE
intermediates for a row chunk. It does NOT chunk or recompute linear GEMMs.
"""
import argparse
import copy
from functools import partial
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from training.ce_only_loss import chunked_causal_ce


def gradient_difference(a,b,atol=2e-5,rtol=2e-3):
    if a.shape!=b.shape:raise ValueError('Gradient shapes differ')
    failed=0;maximum=0.
    # Bound FP32 diagnostic temporaries even for the 622M-element real head.
    for x,y in zip(a.reshape(-1).split(1<<22),b.reshape(-1).split(1<<22)):
        delta=(x.float()-y.float()).abs()
        maximum=max(maximum,delta.max().item())
        failed+=(delta>atol+rtol*x.float().abs()).sum().item()
    return dict(max_abs=maximum,failed_elements=failed)


def loss_matches(a,b):
    try:torch.testing.assert_close(a,b)
    except AssertionError:return False
    return True


def large_head_probe():
    from transformers.loss.loss_utils import ForCausalLMLoss
    if torch.cuda.mem_get_info()[0]<20*2**30:
        raise RuntimeError('At least 20 GiB free required; refuse large-head diagnostic')
    torch.manual_seed(20261108)
    weight=torch.randn(151936,4096,device='cuda',dtype=torch.bfloat16).mul_(.02).requires_grad_()
    rows=[]
    for seq in (512,2048):
        hidden=torch.randn(1,seq,4096,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        labels=torch.randint(151936,(1,seq),device='cuda')
        weight.grad=None;torch.cuda.synchronize()
        base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        logits=F.linear(hidden,weight);loss=ForCausalLMLoss(logits,labels,151936)
        (loss*.37).backward();torch.cuda.synchronize()
        native_peak=torch.cuda.max_memory_allocated()-base;native_loss=loss.item()
        dx,dw=hidden.grad.detach(),weight.grad.detach()
        hidden.grad=None;weight.grad=None;del logits,loss
        torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        logits=F.linear(hidden,weight);loss=chunked_causal_ce(logits,labels,151936,chunk_size=128)
        (loss*.37).backward();torch.cuda.synchronize()
        candidate_peak=torch.cuda.max_memory_allocated()-base
        candidate_loss=loss.item()
        gradients=dict(hidden=gradient_difference(dx,hidden.grad),weight=gradient_difference(dw,weight.grad))
        close=loss_matches(torch.tensor(native_loss),torch.tensor(candidate_loss))
        rows.append(dict(seq=seq,hidden_size=4096,vocab_size=151936,dtype='torch.bfloat16',
            loss_max_abs=abs(native_loss-candidate_loss),gradients=gradients,
            native_peak_allocated_delta_bytes=native_peak,ce_only_peak_allocated_delta_bytes=candidate_peak,
            passed=close and all(g['failed_elements']==0 for g in gradients.values())))
        del logits,loss,dx,dw,hidden,labels
        weight.grad=None
    return dict(complete=True,passed=all(r['passed'] for r in rows),records=rows,
        scope='Synthetic Qwen3-8B head shapes only; allocated deltas exclude preexisting tensors; not model/FSDP timing or quality proof')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--large-head',action='store_true')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve diagnostic evidence')
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from benchmarks.fsdp_reference import native_microbatch_reference
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.use_deterministic_algorithms(True)
    if args.large_head:
        report=large_head_probe()
        with args.output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
        print(json.dumps(report,indent=2));return
    rows=[]
    for tied in (False,True):
        torch.manual_seed(8)
        config=Qwen3Config(vocab_size=97,hidden_size=512,intermediate_size=768,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
            head_dim=128,tie_word_embeddings=tied,attention_dropout=0.)
        config._attn_implementation='sdpa'
        reference=Qwen3ForCausalLM(config).cuda().train();candidate=copy.deepcopy(reference)
        candidate.loss_function=partial(chunked_causal_ce,chunk_size=7)
        for seed in (400,420,440,460):
            torch.manual_seed(seed);ids=torch.randint(97,(2,11),device='cuda')
            outputs=[]
            for model in (reference,candidate):
                model.zero_grad(set_to_none=True)
                out=native_microbatch_reference(model,ids,dtype=torch.bfloat16,tied=tied)
                out.loss.backward();outputs.append(out)
            failures=[]
            for (name,a),(other,b) in zip(reference.named_parameters(),candidate.named_parameters()):
                assert name==other
                atol,rtol=(3e-4,.03) if tied else (2e-5,2e-3)
                delta=(a.grad-b.grad).abs()
                failed=(delta>atol+rtol*a.grad.abs()).sum().item()
                if failed:failures.append(dict(name=name,failed_elements=failed,max_abs=delta.max().item()))
            loss_close=loss_matches(outputs[0].loss,outputs[1].loss)
            logits_equal=torch.equal(outputs[0].logits,outputs[1].logits)
            rows.append(dict(tied=tied,seed=seed,loss_max_abs=(outputs[0].loss-outputs[1].loss).abs().item(),
                loss_close=loss_close,logits_equal=logits_equal,gradient_failures=failures,
                passed=loss_close and logits_equal and not failures))
        del reference,candidate
    report=dict(complete=True,passed=all(r['passed'] for r in rows),records=rows,
        scope='Native-only deterministic CE-only chunking prototype; not FSDP, real-model quality or performance proof')
    with args.output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
