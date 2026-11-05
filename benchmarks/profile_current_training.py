"""Short profiling of current compiled HF + FlashOptim; no checkpoints."""
import json,time,statistics
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from benchmarks.screen_research_candidates import load_model
from benchmarks.research_candidates import make_optimizer

def kernel_summary(root):
    from collections import defaultdict
    trace=json.loads((root/'trace.json').read_text())
    kernels=defaultdict(lambda:dict(count=0,total_us=0.))
    # CUDA user ranges are not leaf kernels. Use Chrome categories, not just
    # device_type, to exclude CompiledFxGraph/optimizer annotation durations.
    for event in trace['traceEvents']:
        if event.get('cat')=='kernel':
            kernels[event['name']]['count']+=1
            kernels[event['name']]['total_us']+=event.get('dur',0.)
    rows=sorted((dict(name=k,**v) for k,v in kernels.items()),key=lambda r:r['total_us'],reverse=True)
    report=json.loads((root/'profile.json').read_text())
    report['cuda_kernels']=rows;report['kernel_summary_source']='Chrome category kernel only'
    (root/'profile.json').write_text(json.dumps(report,indent=2))
    return rows

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=['baseline','consumed'],default='baseline')
    parser.add_argument('--tag',default='current_training_profile_v1')
    parser.add_argument('--seq',type=int,default=512)
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args()
    root=Path('artifacts')/args.tag
    if args.summarize_only:
        print(json.dumps(kernel_summary(root)[:12],indent=2));return
    root.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(20261102)
    model=load_model()
    if args.variant=='consumed':
        from training.tied_embedding import patch_tied_embedding
        patch_tied_embedding(model,consume_proxy=True)
    optimizer=make_optimizer(model,'flashoptim',1e-5)
    ids=np.memmap('artifacts/convergence_corpus_v1/validation.bin',dtype='<u4',mode='r')
    batch=torch.from_numpy(ids[:args.seq].astype(np.int64)).unsqueeze(0).cuda()
    def loss_fn(x):return model(x,labels=x,use_cache=False).loss
    fn=torch.compile(loss_fn,fullgraph=True,dynamic=False)
    def step(profile=False):
        optimizer.zero_grad(set_to_none=True)
        with torch.profiler.record_function('forward'):loss=fn(batch)
        with torch.profiler.record_function('backward'):loss.backward()
        with torch.profiler.record_function('optimizer'):optimizer.step()
        return loss
    times=[]
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for i in range(20):
            torch.cuda.synchronize();tick=time.perf_counter();loss=step();torch.cuda.synchronize()
            if i>=4:times.append((time.perf_counter()-tick)*1000)
        print('WARMUP_AND_TIMING_COMPLETE',statistics.median(times),flush=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True,profile_memory=True) as prof:
            for _ in range(2):step();prof.step()
            torch.cuda.synchronize()
    rows=[]
    for event in prof.key_averages(group_by_input_shape=True):
        rows.append(dict(name=event.key,shapes=event.input_shapes,count=event.count,
            self_device_us=event.self_device_time_total,device_us=event.device_time_total,
            self_cpu_us=event.self_cpu_time_total))
    rows.sort(key=lambda r:r['self_device_us'],reverse=True)
    report=dict(torch=torch.__version__,seq=args.seq,batch=1,variant=args.variant,dtype='bf16',optimizer='FlashOptim no master',
                step_ms=times,median_step_ms=statistics.median(times),profile_steps=2,
                events=rows,cuda_kernels=[],loss=float(loss.detach()),scope='Unprofiled steady timing separate from profiler; fixed GPU input, shared machine.')
    (root/'profile.json').write_text(json.dumps(report,indent=2))
    prof.export_chrome_trace(str(root/'trace.json'))
    kernel_summary(root)
    print(json.dumps(rows[:15],indent=2),flush=True)

if __name__=='__main__':main()
