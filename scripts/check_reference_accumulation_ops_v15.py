"""Monitored real8B local checks for the actual reference-accumulation module."""
import argparse,gc,json,os,subprocess,threading
from pathlib import Path
from unittest.mock import patch
import torch
from qwen3_triton.inference import accumulation,kernels
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from scripts import diagnose_qwen3_native_order_v15 as local
from scripts.verify_qwen3_inference_v15 import sha
ROOT=Path(__file__).resolve().parents[1]

@torch.inference_mode()
def boundary_checks():
 rows=[];torch.manual_seed(20261117)
 for label in ['strided_input','strided_weight','unsupported_M']:
  m=34 if label=='unsupported_M' else 65;n=k=4096
  x=torch.randn((m,k*2 if label=='strided_input' else k),device='cuda',dtype=torch.bfloat16)
  w=torch.randn((n,k*2 if label=='strided_weight' else k),device='cuda',dtype=torch.bfloat16)
  if label=='strided_input':x=x[:,::2]
  if label=='strided_weight':w=w[:,::2]
  assert accumulation.linear(x,w) is None,label
  expected=kernels.linear(x,w);actual=kernels.linear(x,w,reference_accumulation=True)
  row=dict(case=label,bitwise_equal_to_default=bool(torch.equal(expected,actual)))
  assert row['bitwise_equal_to_default'];rows.append(row)
 del x,w,expected,actual
 original=accumulation.raw_empty
 def poisoned(shape,**kw):
  out=original(shape,**kw)
  if len(shape)==3 and kw['dtype']==torch.float32:out.fill_(float('nan'))
  return out
 for m,n,k in [(17,12288,4096),(65,4096,4096),(65,1024,4096),(65,4096,12288)]:
  x=torch.zeros((m,k),device='cuda',dtype=torch.bfloat16);x[:,-1]=1
  w=torch.zeros((n,k),device='cuda',dtype=torch.bfloat16);w[:,-1]=1
  with patch.object(accumulation,'raw_empty',side_effect=poisoned):actual=accumulation.linear(x,w)
  row=dict(case='last_K_and_poisoned_partials',m=m,n=n,k=k,all_outputs_one=bool((actual==1).all()))
  assert row['all_outputs_one'],row;rows.append(row)
 del x,w,actual
 x=torch.ones((1,4096),device='cuda',dtype=torch.bfloat16);w=torch.ones((1024,4096),device='cuda',dtype=torch.bfloat16)
 assert accumulation.linear(x,w,gemv=False) is None
 expected=kernels.linear(x,w,gemv=False);actual=kernels.linear(x,w,gemv=False,reference_accumulation=True)
 assert torch.equal(expected,actual);rows.append(dict(case='disabled_GEMV_preference',bitwise_equal_to_default=True))
 return rows

@torch.inference_mode()
def run(out,foreign):
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 files=[*sorted((ROOT/'qwen3_triton').rglob('*.py')),Path(__file__),ROOT/'scripts/diagnose_qwen3_native_order_v15.py']
 hashes={str(p.relative_to(ROOT)):sha(p) for p in files};boundaries=boundary_checks();gc.collect();torch.cuda.empty_cache()
 original_match=local.matched_linear;original_route=local.route
 def actual(x,w,**kw):return kernels.linear(x,w,reference_accumulation=True,**kw)
 def route(m,n,k):return str(accumulation.strategy(m,n,k)) if accumulation.strategy(m,n,k) is not None else 'default'
 local.matched_linear=actual;local.route=route
 try:rows=local.local_checks()
 finally:local.matched_linear=original_match;local.route=original_route
 for p in files:assert sha(p)==hashes[str(p.relative_to(ROOT))],p
 result=dict(complete=True,actual_exported_module=True,physical_GPU=1,GPU=torch.cuda.get_device_name(),torch=torch.__version__,CUDA=torch.version.cuda,boundaries=boundaries,local_rows=rows,all_local_calls=len(rows),all_local_unequal=sum(v['matched_unequal'] for v in rows),all_local_passed=all(v['matched_unequal']==0 for v in rows),source_sha256=hashes,foreign_context_observations=list(foreign),resource_clean=not foreign,limitations='Local identical-HF-input checks and layout/tail guards only. Full exported-model gate/audit and matched request timings are separate requirements.')
 raw=json.dumps(result,indent=2)+'\n';assert len(raw.encode())<=4*2**20;(out/'result.json').write_text(raw)
 assert len(rows)==2520 and result['all_local_passed'] and result['resource_clean']
 print(json.dumps({k:v for k,v in result.items() if k not in ['local_rows','source_sha256']}),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and out.is_relative_to(Path('/data1/tzh')) and not out.exists();out.mkdir(parents=True)
 foreign=[];stop=threading.Event()
 with stable_lease([1]):
  uuid=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
  def monitor():
   while not stop.wait(.5):
    try:
     raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=10)
     foreign.extend(line for line in raw.splitlines() if line.split(',')[0].strip()==uuid and int(line.split(',')[1])!=os.getpid())
    except Exception as exc:foreign.append(repr(exc))
  thread=threading.Thread(target=monitor,daemon=True);thread.start()
  try:run(out,foreign)
  except BaseException as exc:(out/'failure.json').write_text(json.dumps(dict(error=repr(exc),foreign_context_observations=foreign),indent=2)+'\n');raise
  finally:stop.set();thread.join()
if __name__=='__main__':main()
