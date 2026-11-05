"""Short same-data training trajectory and held-out loss check, not convergence.

No checkpoints written. Every variant reloads identical pretrained weights.
BF16 parameters/Adam moments match the performance experiment exactly.
"""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from torch.nn.attention import sdpa_kernel,SDPBackend
from training.hf_operators import training_operators


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=64)
    p.add_argument('--seq',type=int,default=512)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--candidate',choices=['swiglu_hf_bwd','triton_adamw'],default='swiglu_hf_bwd')
    p.add_argument('--adamw-math',choices=['legacy','rounded'],default='legacy')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(4)
    path='/data1/tzh/models/Qwen/Qwen3-1.7B'
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    corpus=json.loads(Path('artifacts/frequency_wikitext_data.json').read_text())
    ids={s:torch.tensor(tok(corpus['splits'][s]['text'],add_special_tokens=False,verbose=False)['input_ids'])
         for s in ('validation','test')}
    train=ids['validation'][:a.steps*a.seq].reshape(a.steps,1,a.seq).cuda()
    test=ids['test'][:8*a.seq].reshape(8,1,a.seq).cuda()
    report=dict(scope='Short trajectory only; not pretraining convergence or task-quality proof',
        steps=a.steps,seq=a.seq,optimizer='fused AdamW lr1e-5, BF16 params/moments',
        gates={'max_train_loss_abs':.02,'heldout_loss_abs':.02,'global_parameter_relative_l2':.001},variants={})
    baseline_state=None
    for variant in ('hf',a.candidate):
        torch.manual_seed(20261102)
        model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,
            torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().train()
        if variant=='triton_adamw':
            from training.triton_adamw import TritonAdamW
            optimizer=TritonAdamW(model.parameters(),lr=1e-5,math_mode=a.adamw_math)
        else:optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,fused=True)
        losses=[]
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION),training_operators(model,
                swiglu=variant=='swiglu_hf_bwd',hf_semantics='triton_backward' if variant=='swiglu_hf_bwd' else False):
            for step,batch in enumerate(train):
                optimizer.zero_grad(set_to_none=True)
                loss=model(batch,labels=batch,use_cache=False).loss
                loss.backward();optimizer.step()
                losses.append(float(loss.detach()))
                if not torch.isfinite(loss):raise AssertionError('Nonfinite loss')
                if step%8==0:print(variant,step,losses[-1],flush=True)
            model.eval()
            with torch.no_grad():heldout=[float(model(b,labels=b,use_cache=False).loss) for b in test]
        row=dict(train_losses=losses,heldout_losses=heldout,heldout_mean=sum(heldout)/len(heldout))
        if baseline_state is None:
            baseline_state={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        else:
            diff2=ref2=0.;max_abs=0.
            for n,param in model.named_parameters():
                w=param.detach().cpu().float();ref=baseline_state[n].float();delta=w-ref
                diff2+=float(delta.square().sum());ref2+=float(ref.square().sum())
                max_abs=max(max_abs,float(delta.abs().max()))
            baseline=report['variants']['hf']
            train_error=max(abs(x-y) for x,y in zip(losses,baseline['train_losses']))
            eval_error=abs(row['heldout_mean']-baseline['heldout_mean'])
            row.update(max_train_loss_abs=train_error,heldout_loss_abs=eval_error,
                global_parameter_relative_l2=(diff2/max(ref2,1e-20))**.5,max_parameter_abs=max_abs)
            row['accepted']=train_error<.02 and eval_error<.02 and row['global_parameter_relative_l2']<.001
        report['variants'][variant]=row
        a.output.write_text(json.dumps(report,indent=2))
        del model,optimizer,loss
        torch.cuda.empty_cache()


if __name__=='__main__':main()
