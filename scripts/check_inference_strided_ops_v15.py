"""Independent exact arithmetic oracle for small-row positive-stride CUDA views."""
import json,torch
from qwen3_triton.inference import kernels
from qwen3_triton.inference.dispatch import InferenceRuntimeMode

@torch.inference_mode()
def run(path):
 rows=[];runtime=InferenceRuntimeMode()
 for m in [1,8]:
  n=4096 if m==1 else 64;k=4096
  xb=(torch.arange(m*k*2,device='cuda').reshape(m,k*2)%7).to(torch.bfloat16)
  wb=((torch.arange(n*k*2,device='cuda').reshape(n,k*2)%5)-2).to(torch.bfloat16)
  for layout in ['input','weight','both']:
   x=xb[:,1::2] if layout in ['input','both'] else xb[:,:k].contiguous()
   w=wb[:,1::2] if layout in ['weight','both'] else wb[:,:k].contiguous()
   expected=(x.float()@w.float().t()).to(x.dtype)
   for reference in [False,True]:
    for sliced in [False,True]:
     with runtime:actual=kernels.linear(x,w,reference_accumulation=reference,sliced=sliced)
     row=dict(m=m,n=n,k=k,layout=layout,input_stride=list(x.stride()),weight_stride=list(w.stride()),input_storage_offset=x.storage_offset(),weight_storage_offset=w.storage_offset(),reference_accumulation=reference,sliced=sliced,bitwise_equal_to_independent_FP32_oracle=bool(torch.equal(actual,expected)),max_abs=float((actual.float()-expected.float()).abs().max()))
     rows.append(row);assert row['bitwise_equal_to_independent_FP32_oracle'],row
 result=dict(complete=True,cases=len(rows),all_passed=all(r['bitwise_equal_to_independent_FP32_oracle'] for r in rows),rows=rows,strict_GPU_dispatch=runtime.report(),oracle='Dyadic/integer inputs with exact FP32 sum below2^24; native FP32 matrix multiply then BF16 cast. Candidate runs under strict Triton runtime. Positive-stride input/weight views include nonzero storage offsets; no default-route oracle.')
 path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(strided_cases=len(rows),all_passed=result['all_passed'])),flush=True)
