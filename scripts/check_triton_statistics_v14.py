"""Real-GPU FP64 validation accumulation/division against independent CPU floats.

Includes32 changed batch losses, unequal token weights, zero-valid NaN batch,
FP64 scalar/tensor division and exact factory fill precision. Not a model run.
"""
import argparse,json,math,os
from pathlib import Path
from contextlib import nullcontext
from datetime import timedelta
import torch
import torch.distributed as dist
from qwen3_triton.runtime.dispatch import TritonRuntimeMode
from qwen3_triton.runtime.metrics import accumulate_validation
from qwen3_triton.gradient_archive import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rank=int(os.environ['RANK']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=True
    dist.init_process_group('gloo',timeout=timedelta(minutes=5))
    try:
        assert dist.get_world_size()==4
        # CPU FP32 round first, as model loss does, then independent Python
        # FP64 multiply/add in the exact order of the original validation core.
        inputs=[(float(torch.tensor((i+1)*(rank+1)/7.+(1e-7 if i%2 else 0),dtype=torch.float32)),(1+i%4)*511) for i in range(32)]
        expected_nll=0.;expected_count=0
        for loss,tokens in inputs:expected_nll+=loss*tokens;expected_count+=tokens
        mode=TritonRuntimeMode();captured=rank==0
        profile=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) if captured else nullcontext()
        exact_fill=math.pi+1e-15
        with profile as trace:
            with mode:
                seed=torch.tensor(0.,dtype=torch.float64).cuda();totals=seed.new_zeros((2,))
                for loss,tokens in inputs:
                    value=torch.tensor(loss,dtype=torch.float32).cuda();accumulate_validation(totals,value,tokens)
                invalid=torch.tensor(float('nan'),dtype=torch.float32).cuda();accumulate_validation(totals,invalid,0)
                mean_tensor=totals[0]/totals[1];mean_scalar=totals[0]/float(expected_count)
                filled=totals.new_full((2,),exact_fill);ones=totals.new_ones((3,))
                empty=totals.new_empty_strided((2,3),(1,2))
                output=totals.cpu();mt=float(mean_tensor);ms=float(mean_scalar)
                fill_cpu=filled.cpu();ones_cpu=ones.cpu();empty_cpu=empty.cpu()
                torch.cuda.synchronize()
        assert output.tolist()==[expected_nll,float(expected_count)]
        assert mt==ms==expected_nll/expected_count
        assert fill_cpu.tolist()==[exact_fill]*2 and ones_cpu.tolist()==[1.]*3
        assert tuple(empty.stride())==(1,2) and torch.isnan(empty_cpu).all()
        for bad in [-1]:
            try:accumulate_validation(totals,value,bad)
            except ValueError:pass
            else:raise AssertionError('Negative token count accepted')
        audit=None
        if captured:
            path=a.output/'statistics_trace.json.gz';trace.export_chrome_trace(str(path))
            import gzip
            from collections import Counter
            events=[e for e in json.loads(gzip.decompress(path.read_bytes()))['traceEvents'] if e.get('cat')=='kernel']
            counts=Counter(e['name'] for e in events)
            allowed={'_runtime_fill','_runtime_typed_fill','_runtime_strided_copy','_runtime_validation_accumulate','_runtime_binary'}
            assert counts['_runtime_validation_accumulate']==32 and counts['_runtime_binary']==2 and not (set(counts)-allowed),counts
            audit=dict(sha256=sha(path),kernels=dict(counts),only_Triton=True)
        row=dict(rank=rank,complete=True,passed=True,changed_loss_batches=32,independent_CPU_FP64_bitwise_equal=True,
            scalar_and_tensor_division_exact=True,zero_valid_NaN_skipped=True,FP64_factory_precision_exact=True,
            deterministic_noncontiguous_empty_NaN=True,expected_tokens=expected_count,expected_nll=expected_nll,mean=mt,
            dispatch=mode.report(),candidate_only_trace=audit)
        rows=[None]*4;dist.all_gather_object(rows,row)
        if rank==0:
            d=dict(complete=True,passed=True,kind='statistics',ranks=rows,full_model_qualified=False,scope=__doc__)
            (a.output/'check.json').write_text(json.dumps(d,indent=2)+'\n')
            print(json.dumps(dict(complete=True,passed=True,kind='statistics',full_model_qualified=False)),flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
