"""Gate packed MLP on full-model compiled gradients; no optimizer updates."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from benchmarks.screen_research_candidates import load_model
from benchmarks.test_training_fusions import patch_model

def main():
    path=Path('artifacts/packed_training_gate_v1.json')
    if path.exists():raise FileExistsError(path)
    torch.set_num_threads(4);torch.manual_seed(20261106)
    model=load_model()
    ids=np.memmap('artifacts/convergence_corpus_v1/validation.bin',dtype='<u4',mode='r')
    x=torch.from_numpy(ids[:512].astype(np.int64)).unsqueeze(0).cuda()
    def fn(x):return model(x,labels=x,use_cache=False).loss
    compiled=torch.compile(fn,fullgraph=True,dynamic=False)
    ref=None;checks=[]
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for stage in ('baseline','repeat','packed'):
            model.zero_grad(set_to_none=True)
            if stage=='packed':patch_model(model,'packed')
            loss=compiled(x);loss.backward()
            grads={}
            for name,p in model.named_parameters():
                if p.grad is None:continue
                if 'packed_gate_up.weight' in name:
                    gate,up=p.grad.chunk(2,dim=0)
                    grads[name.replace('packed_gate_up','gate_proj')]=gate.detach().cpu().clone()
                    grads[name.replace('packed_gate_up','up_proj')]=up.detach().cpu().clone()
                else:grads[name]=p.grad.detach().cpu().clone()
            if ref is None:ref=(float(loss.detach()),grads);continue
            assert grads.keys()==ref[1].keys()
            num=den=0.;rows=[]
            for name,g in grads.items():
                a,b=g.float(),ref[1][name].float()
                err=float((a-b).square().sum());norm=float(b.square().sum())
                num+=err;den+=norm
                rows.append(dict(name=name,relative_l2=(err/max(norm,1e-30))**.5))
            global_error=(num/max(den,1e-30))**.5
            worst=max(r['relative_l2'] for r in rows)
            checks.append(dict(stage=stage,loss_abs=abs(float(loss.detach())-ref[0]),
                global_gradient_l2=global_error,max_parameter_gradient_l2=worst,
                passed=global_error<=.02 and worst<=.05 and abs(float(loss.detach())-ref[0])<=.01,
                worst_parameters=sorted(rows,key=lambda r:r['relative_l2'],reverse=True)[:10]))
            del grads
    result=dict(checks=checks,enabled=False,passed=all(c['passed'] for c in checks),
        note='No default path changed; failed gate means do not integrate. Global2%, per-parameter5%, lossabs0.01.')
    path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
