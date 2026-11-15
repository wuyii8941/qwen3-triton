"""Screen true GEMM/round/mean fusion on full8B vocabulary-shard shapes."""
from pathlib import Path
import argparse,json,statistics
import torch
from qwen3_triton.kernels.linear import matmul
from qwen3_triton.kernels.vocab_ce import accumulate_weight_gradient
from qwen3_triton.kernels.vocab_dw import weight_gradient_mean
from qwen3_triton.audit import NoFallback


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    assert not args.output.exists();torch.manual_seed(1115);torch.cuda.set_device(0);torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    dz=torch.randn((2048,37984),device='cuda',dtype=torch.bfloat16)
    hidden=torch.randn((2048,4096),device='cuda',dtype=torch.bfloat16)
    def old():
        out=torch.empty((37984,4096),device='cuda',dtype=torch.float32)
        for rank in range(4):
            partial=matmul(dz[rank*512:(rank+1)*512].t(),hidden[rank*512:(rank+1)*512],optimized=True)
            accumulate_weight_gradient(partial,out,4,rank==0)
        return out
    configs=[(64,64,32,8,4,3),(64,128,32,8,8,3),(32,128,32,8,4,3),(64,64,64,8,4,3)]
    with NoFallback():reference=old()
    numerics=[]
    for cfg in configs:
        with NoFallback():got=weight_gradient_mean(dz,hidden,512,4,cfg)
        same=torch.equal(got,reference);error=float((got-reference).norm()/reference.norm())
        assert same if cfg[2]==32 else error<.005
        numerics.append(dict(config=cfg,bitwise_equal=same,relative_l2=error));del got
    del reference
    functions=[('unfused',old)]+[(str(i),lambda cfg=cfg:weight_gradient_mean(dz,hidden,512,4,cfg)) for i,cfg in enumerate(configs)]
    timings={}
    for order,rows in [(1,functions),(2,list(reversed(functions)))]:
        for name,fn in rows:
            samples=[]
            for i in range(13):
                torch.cuda.synchronize();begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                begin.record();result=fn();end.record();end.synchronize()
                if i>=3:samples.append(begin.elapsed_time(end))
                del result
            timings[f'{name}_r{order}']=dict(median_ms=statistics.median(samples),samples_ms=samples)
    eligible=[]
    for i,cfg in enumerate(configs):
        reductions=[1-timings[f'{i}_r{o}']['median_ms']/timings[f'unfused_r{o}']['median_ms'] for o in [1,2]]
        numerics[i]['reductions']=reductions
        if min(reductions)>=.03 and numerics[i]['bitwise_equal']:eligible.append(i)
    selected=min(eligible,key=lambda i:sum(timings[f'{i}_r{o}']['median_ms'] for o in [1,2])) if eligible else None
    report=dict(complete=True,full_model_qualified=False,shape=dict(V=37984,H=4096,rows_per_rank=512,world_size=4),
        numerics=numerics,timings=timings,selected_config=configs[selected] if selected is not None else None,
        scope='One-GPU CUDA-event module benchmark including allocations/fills. Every output coordinate compared; full-model qualification remains separate.')
    with args.output.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(complete=True,numerics=numerics,selected_config=report['selected_config']),indent=2),flush=True)


if __name__=='__main__':main()
