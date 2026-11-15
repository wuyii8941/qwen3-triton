"""Real FSDP host hooks with a strict Triton GPU data path and state updates.

This is a small integration gate, not full8B qualification. Full unsharded
Triton model gradients on each rank are exchanged on CPU/Gloo to construct
independent FP64 averaged shard references. Native AdamW updates only the
reference shards outside candidate profiles. Both reshard/retain policies run.
"""
import argparse,copy,hashlib,json,os
from collections import Counter
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard,MixedPrecisionPolicy
from qwen3_triton.kernels.linear import linear
from qwen3_triton.runtime.adamw import TritonAdamW,local
from qwen3_triton.runtime.collectives import TritonCollectives
from qwen3_triton.runtime.dispatch import TritonRuntimeMode
from qwen3_triton.runtime.fsdp import configure


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.randn(128,128)*.02)
    def forward(self,x):return linear(x,self.weight,optimized=False)+x


class Model(torch.nn.Module):
    def __init__(self):super().__init__();self.blocks=torch.nn.ModuleList([Block(),Block()])
    def forward(self,x):
        for layer in self.blocks:x=layer(x)
        return x


def compare(a,b,*,atol=1e-7):
    aa=a.detach().cpu().double();bb=b.detach().cpu().double()
    passed=bool(((aa-bb).abs()<=atol+2e-6*bb.abs()).all());assert passed,(float((aa-bb).abs().max()),tuple(aa.shape))
    return dict(passed=passed,max_abs=float((aa-bb).abs().max()),bitwise_equal=torch.equal(aa,bb))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rank=int(os.environ['RANK']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    dist.init_process_group('gloo',timeout=timedelta(minutes=20))
    try:
        comm=TritonCollectives(capacities=dict.fromkeys(['gather','scatter','reduce','alltoall'],2**20))
        policies=[]
        for retain in [False,True]:
            torch.manual_seed(712);model=Model();reference=copy.deepcopy(model).bfloat16().cuda()
            mp=MixedPrecisionPolicy(param_dtype=torch.bfloat16,reduce_dtype=torch.float32)
            for layer in model.blocks:fully_shard(layer,mp_policy=mp,reshard_after_forward=not retain)
            fully_shard(model,mp_policy=mp);hooks=configure(model,comm)
            parameters=list(model.parameters());assert len(parameters)==2
            masters=[torch.nn.Parameter(local(t).detach().clone()) for t in parameters]
            optimizer=TritonAdamW(parameters);native=torch.optim.AdamW(masters,lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01,fused=True)
            steps=[];modes=[]
            for step in range(1,5):
                generator=torch.Generator().manual_seed(1600+step*13+rank)
                host=(torch.randn((17,128),generator=generator)*.1).bfloat16()
                x=host.cuda().requires_grad_();ref_x=host.cuda().requires_grad_();reference.zero_grad(set_to_none=True)
                ref_out=reference(ref_x);ref_loss=ref_out.float().sum();ref_loss.backward()
                gathered=[None]*4;dist.all_gather_object(gathered,[v.grad.detach().cpu() for v in reference.parameters()])
                expected=[]
                for i in range(2):
                    full=sum(row[i].double() for row in gathered).div(4).float()
                    expected.append(full[rank*32:(rank+1)*32].contiguous())
                mode=TritonRuntimeMode();profile=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) if rank==0 and step==4 else None
                if profile is not None:profile.__enter__()
                with mode:
                    optimizer.zero_grad(set_to_none=True)
                    out=model(x);loss=out.float().sum();loss.backward();optimizer.step()
                    global_loss=loss.detach().clone();comm.all_reduce(global_loss,'sum');global_loss/=4
                    finite=all(bool(torch.isfinite(local(t)).all()) for t in parameters)
                    torch.cuda.synchronize()
                if profile is not None:
                    profile.__exit__(None,None,None);path=a.output/f'fsdp_retain{int(retain)}_trace.json';profile.export_chrome_trace(str(path))
                    counts=Counter(e['name'] for e in json.loads(path.read_text())['traceEvents'] if e.get('cat')=='kernel')
                    unknown={k:v for k,v in counts.items() if k!='_gemm' and not k.startswith(('_runtime_','_adamw_'))}
                    assert counts and not unknown,unknown
                    trace=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),only_Triton=True,kernels=dict(counts))
                else:trace=None
                assert finite
                output_check=compare(out,ref_out);input_check=compare(x.grad,ref_x.grad)
                grad_checks=[compare(local(t.grad),ref) for t,ref in zip(parameters,expected)]
                for master,grad in zip(masters,expected):master.grad=grad.cuda()
                native.step();parameter_checks=[compare(local(t),master,atol=1e-8) for t,master in zip(parameters,masters)]
                moment_checks=[{key:compare(optimizer.state[t][key],native.state[master][key],atol=1e-12 if key=='exp_avg' else 1e-14) for key in ['exp_avg','exp_avg_sq']} for t,master in zip(parameters,masters)]
                steps.append(dict(step=step,output=output_check,input_gradient=input_check,all_shard_gradients=grad_checks,
                    parameters=parameter_checks,moments=moment_checks,candidate_only_trace=trace))
                modes.append(mode.report())
                # Reconstruct reference weights from independently updated
                # native optimizer shards, not from candidate parameters.
                gathered=[None]*4;dist.all_gather_object(gathered,[v.detach().cpu() for v in masters])
                with torch.no_grad():
                    for i,param in enumerate(reference.parameters()):param.copy_(torch.cat([row[i] for row in gathered]).bfloat16().cuda())
                del out,loss,ref_out,ref_loss,x,ref_x
            policies.append(dict(retain=retain,passed=True,steps=steps,runtime_dispatch=modes,
                hooks=dict(units=hooks['units'],all_gather_calls=hooks['all_gather'].calls,reduce_scatter_calls=hooks['reduce_scatter'].calls)))
            del model,reference,optimizer,native,parameters,masters
            torch.cuda.empty_cache();dist.barrier()
        report=dict(rank=rank,complete=True,passed=True,policies=policies,communication=comm.report(),scope=__doc__)
        reports=[None]*4;dist.all_gather_object(reports,report)
        if rank==0:(a.output/'check.json').write_text(json.dumps(dict(complete=True,passed=True,kind='fsdp',ranks=reports),indent=2)+'\n')
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
