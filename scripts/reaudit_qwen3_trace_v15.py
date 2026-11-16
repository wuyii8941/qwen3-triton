"""Separate complete phase recording; preserves earlier incomplete trace failure.

Collector-bootstrap kernels are retained in the raw trace and labeled separately.
Every actual phase launch must have a GPU activity event and exact model counts.
"""
import argparse,hashlib,json,os,subprocess,threading,torch
from pathlib import Path
from qwen3_triton.inference import TritonQwen3Inference
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from verify_qwen3_inference_v15 import trace,sha

@torch.inference_mode()
def run(out,foreign):
 torch.set_num_threads(2);torch.manual_seed(20261116)
 root=Path(__file__).resolve().parents[1];files=[*sorted((root/'qwen3_triton').rglob('*.py')),Path(__file__),root/'scripts/verify_qwen3_inference_v15.py']
 sources={str(p.relative_to(root)):sha(p) for p in files}
 model,reports_setup=trace(out,'setup',lambda:TritonQwen3Inference.from_pretrained('/data1/tzh/models/Qwen/Qwen3-8B'),0,warmup_collection=True)
 ids=torch.randint(1,100000,(1,65),device='cuda');token=torch.tensor([[50]],device='cuda');cache=model.new_cache(1,80)
 _,prefill=trace(out,'prefill',lambda:model(ids,past_key_values=cache),36,warmup_collection=True)
 _,decode=trace(out,'decode',lambda:model(token,past_key_values=cache),36,warmup_collection=True)
 _,generate=trace(out,'generate',lambda:model.generate(ids,max_new_tokens=3,eos_token_id=[],pad_token_id=0),108,warmup_collection=True)
 reports=dict(setup=reports_setup,prefill=prefill,decode=decode,generate=generate)
 for p in files:assert sha(p)==sources[str(p.relative_to(root))],p
 result=dict(complete=True,phase_coverage_passed=all(x['passed'] for x in reports.values()),resource_clean=not foreign,foreign_process_observations=list(foreign),traces=reports,source_sha256=sources,model='Qwen/Qwen3-8B',layers=36,physical_GPU=1,
  HF_numerical_qualification=False,scope='Fresh separately labeled setup/prefill/decode/3token-generation recording with launch-to-GPU activity matching and complete operation counts. Does not rewrite full_validation_008 or reclassify its incomplete decode trace. Bootstrap is64Triton position kernels outside actual annotations; raw events retained.')
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','traces']}),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();assert out.is_relative_to(Path('/data1/tzh')) and not out.exists();out.mkdir(parents=True)
 foreign=[];stop=threading.Event()
 with stable_lease([1]):
  uuid=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
  def monitor():
   while not stop.wait(.5):
    try:
     raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=10)
     for s in raw.splitlines():
      f=s.split(',')
      if f[0].strip()==uuid and int(f[1])!=os.getpid():foreign.append(s)
    except Exception as exc:foreign.append(repr(exc))
  worker=threading.Thread(target=monitor,daemon=True);worker.start()
  try:run(out,foreign)
  except BaseException as exc:(out/'failure.json').write_text(json.dumps(dict(error=repr(exc),foreign_process_observations=foreign),indent=2)+'\n');raise
  finally:stop.set();worker.join()
if __name__=='__main__':main()
