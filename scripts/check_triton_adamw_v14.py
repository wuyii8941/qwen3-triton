"""Real-GPU AdamW comparison, independent CPU oracle, restore and coverage.

Reference/check operations may use native CUDA. The separately captured
candidate-only optimizer profile must contain exclusively Triton kernels.
No full-Qwen training/performance qualification is claimed by this module.
"""
import argparse,copy,hashlib,json,math,os
from collections import Counter
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from qwen3_triton.runtime.adamw import TritonAdamW


def errors(actual,reference,atol):
    a=actual.detach().double();r=reference.detach().double();delta=(a-r).abs()
    gate=bool((delta<=atol+2e-6*r.abs()).all())
    return dict(passed=gate,max_abs=float(delta.max()),relative_l2=float((a-r).norm()/r.norm().clamp_min(1e-30)),bitwise_equal=torch.equal(actual,reference))


def compare(ours,native,params,refs):
    rows=[]
    for p,r in zip(params,refs):
        a,b=ours.state.get(p),native.state.get(r)
        row=dict(parameter=errors(p,r,1e-8))
        if a or b:
            assert a and b
            row.update(step_equal=float(a['step'])==float(b['step']),exp_avg=errors(a['exp_avg'],b['exp_avg'],1e-12),
                exp_avg_sq=errors(a['exp_avg_sq'],b['exp_avg_sq'],1e-14))
        row['passed']=all(v['passed'] if isinstance(v,dict) else v for k,v in row.items())
        rows.append(row)
    assert all(r['passed'] for r in rows),rows
    return rows


def cpu_oracle(p,g,m,v,step,lr,b1,b2,decay,eps):
    # CPU double arithmetic with explicit FP32 boundaries, independent of
    # both GPU implementations. C++ native fused coefficients are doubles.
    p=(p.double()-lr*decay*p.double()).float()
    m=(b1*m.double()+(1-b1)*g.double()).float()
    v=(b2*v.double()+(1-b2)*g.double()*g.double()).float()
    bc1=torch.tensor(1-b1**step,dtype=torch.float32)
    bc2=torch.tensor(math.sqrt(1-b2**step),dtype=torch.float32)
    size=torch.tensor(lr/float(bc1),dtype=torch.float32)
    denom=((v.sqrt()/bc2).double()+eps).float()
    return p-size*m/denom,m,v


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    rank=int(os.environ['RANK']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.set_num_threads(2)
    torch.manual_seed(731+rank);torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=True
    dist.init_process_group('gloo',timeout=timedelta(minutes=20))
    try:
        p=[torch.nn.Parameter(torch.randn(n,device='cuda')*.2) for n in [1,7,257,4097,32771]]
        r=[torch.nn.Parameter(x.detach().clone()) for x in p]
        ours=TritonAdamW([dict(params=p[:3],lr=1e-5),dict(params=p[3:],lr=.003,betas=(.8,.95),eps=1e-4,weight_decay=.2)])
        native=torch.optim.AdamW([dict(params=r[:3],lr=1e-5),dict(params=r[3:],lr=.003,betas=(.8,.95),eps=1e-4,weight_decay=.2)],lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01,fused=True)
        # Nonzero moments/counters, and coordinates whose gradients are tiny
        # or zero, distinguish actual stateful AdamW from a first-step proxy.
        for i,(x,y) in enumerate(zip(p,r)):
            state=dict(step=torch.tensor(float(i+17),device='cuda'),exp_avg=torch.randn_like(x)*.001,exp_avg_sq=torch.rand_like(x)*.01)
            ours.state[x]={k:v.clone() for k,v in state.items()};native.state[y]={k:v.clone() for k,v in state.items()}
        cpu_p=p[0].detach().cpu();cpu_m=ours.state[p[0]]['exp_avg'].cpu();cpu_v=ours.state[p[0]]['exp_avg_sq'].cpu()
        checkpoints=[];oracle=[]
        for step in range(1,129):
            for i,(x,y) in enumerate(zip(p,r)):
                if i and (step+i)%5==0:x.grad=None;y.grad=None;continue
                g=torch.randn_like(x)*(10.**(-8+i))
                if step%11==0:g.zero_()
                x.grad=g;y.grad=g.clone()
            cpu_p,cpu_m,cpu_v=cpu_oracle(cpu_p,p[0].grad.cpu(),cpu_m,cpu_v,17+step,1e-5,.9,.999,.01,1e-8)
            ours.step();native.step()
            if step in [1,17,64,128]:
                checkpoints.append(dict(after_updates=step,rows=compare(ours,native,p,r)))
                oracle.append(dict(after_updates=step,parameter=errors(p[0].cpu(),cpu_p,1e-8),exp_avg=errors(ours.state[p[0]]['exp_avg'].cpu(),cpu_m,1e-12),exp_avg_sq=errors(ours.state[p[0]]['exp_avg_sq'].cpu(),cpu_v,1e-14)))
            if step==64:
                saved=copy.deepcopy(ours.state_dict())
                for state in saved['state'].values():
                    for k,v in state.items():
                        if isinstance(v,torch.Tensor):state[k]=v.cpu()
                restored=TritonAdamW([dict(params=p[:3]),dict(params=p[3:])]);restored.load_state_dict(saved);ours=restored
        assert all(v['passed'] for row in oracle for k,v in row.items() if isinstance(v,dict))
        # Official8B local norm, Q-projection, MLP and vocabulary shard sizes.
        shape_rows=[]
        for shape in [(32,),(1024,4096),(3072,4096),(37984,4096)]:
            x=torch.nn.Parameter(torch.randn(shape,device='cuda')*.01);y=torch.nn.Parameter(x.detach().clone())
            g=torch.randn_like(x)*1e-5;x.grad=g;y.grad=g.clone()
            opt=TritonAdamW([x]);ref=torch.optim.AdamW([y],lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01,fused=True)
            for _ in range(3):opt.step();ref.step()
            shape_rows.append(dict(shape=shape,rows=compare(opt,ref,[x],[y])))
            del x,y,g,opt,ref
        rejected=False
        bad=torch.nn.Parameter(torch.zeros(7,device='cuda',dtype=torch.bfloat16));bad.grad=torch.zeros_like(bad)
        try:TritonAdamW([bad]).step()
        except ValueError:rejected=True
        assert rejected
        # Only the candidate update is in this CUDA profile, including state
        # steps, coefficient prep, metadata DMA and the fused update itself.
        for x in p:x.grad=torch.randn_like(x)*1e-4
        torch.cuda.synchronize()
        profile=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) if rank==0 else None
        if profile is not None:profile.__enter__()
        ours.step();torch.cuda.synchronize()
        if profile is not None:
            profile.__exit__(None,None,None);path=a.output/'adamw_trace.json';profile.export_chrome_trace(str(path))
            counts=Counter(e['name'] for e in json.loads(path.read_text())['traceEvents'] if e.get('cat')=='kernel')
            assert counts['_adamw_prepare']==counts['_adamw_update']==2 and set(counts)=={'_adamw_prepare','_adamw_update'},counts
            trace=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),kernels=dict(counts),only_Triton=True)
        else:trace=None
        report=dict(rank=rank,complete=True,passed=True,updates=128,nonzero_initial_state=True,skip_gradient_tested=True,
            CPU_checkpoint_restore_after64=True,native_comparisons=checkpoints,CPU_oracle=oracle,official8B_shard_shapes=shape_rows,
            unsupported_dtype_rejected=rejected,candidate_only_trace=trace,scope=__doc__)
        reports=[None]*4;dist.all_gather_object(reports,report)
        if rank==0:(a.output/'check.json').write_text(json.dumps(dict(complete=True,passed=True,kind='adamw',ranks=reports),indent=2)+'\n')
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
