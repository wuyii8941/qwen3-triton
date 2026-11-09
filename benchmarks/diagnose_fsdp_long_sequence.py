"""Native-only BF16 repeatability probe; no FSDP and no tolerance changes."""
import argparse
import json
from pathlib import Path
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from benchmarks.fsdp_reference import native_microbatch_reference


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seq',type=int,default=1024)
    p.add_argument('--deterministic',action='store_true');args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve diagnostics')
    torch.use_deterministic_algorithms(args.deterministic)
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.manual_seed(8)
    config=Qwen3Config(vocab_size=97,hidden_size=512,intermediate_size=768,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=128,
        tie_word_embeddings=False,attention_dropout=0.)
    config._attn_implementation='sdpa'
    model=Qwen3ForCausalLM(config).cuda().train();rows=[]
    for seed in (400,420,440,460):
        torch.manual_seed(seed);ids=torch.randint(0,97,(2,args.seq),device='cuda')
        snapshots=[];logits=[]
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            out=native_microbatch_reference(model,ids,dtype=torch.bfloat16,tied=False)
            logits.append(out.logits.detach().clone());out.loss.backward()
            snapshots.append({n:p.grad.detach().clone() for n,p in model.named_parameters()})
        parameters=[]
        for name,a in snapshots[0].items():
            b=snapshots[1][name];delta=(a-b).abs()
            failures=(delta>2e-5+2e-3*b.abs()).sum().item()
            if delta.max().item():
                parameters.append(dict(name=name,max_abs=delta.max().item(),
                    relative_l2=((a-b).norm()/b.norm().clamp_min(1e-30)).item(),
                    failed_elements=failures,numel=a.numel()))
        rows.append(dict(seed=seed,logits_max_abs=(logits[0]-logits[1]).abs().max().item(),
            differing_parameters=parameters))
    report=dict(seq=args.seq,head_dim=128,pytorch=torch.__version__,deterministic=args.deterministic,records=rows,
        scope='Native-only repeated identical BF16 forward/backward; diagnostic, not an FSDP correctness waiver')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(source=str(args.output),deterministic=args.deterministic,
        records=[dict(seed=r['seed'],logits_max_abs=r['logits_max_abs'],
            differing_parameters=len(r['differing_parameters']),
            failed_elements=sum(p['failed_elements'] for p in r['differing_parameters'])) for r in rows]),indent=2))


if __name__=='__main__':main()
