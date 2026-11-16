"""CPU regression: missing decoder compute must fail even when36appends remain."""
import gzip,json,tempfile
from pathlib import Path
from unittest.mock import patch
from scripts.verify_qwen3_inference_v15 import trace
COUNTS={'_inference_positions':1,'_lookup':1,'_tables':1,'_inference_gemv':253,'_hf_norm_forward':73,'_hf_norm_rope_forward':72,'_inference_cache_append':36,'_inference_cached_attention':36,'_add':72,'_hf_swiglu_forward':36}
class Profiler:
 def __init__(self,events):self.events=events
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def export_chrome_trace(self,path):Path(path).write_bytes(gzip.compress(json.dumps({'traceEvents':self.events}).encode()))
def events(counts):
 out=[];correlation=0
 for name,n in counts.items():
  for _ in range(n):
   correlation+=1;out.extend([dict(cat='cuda_driver',name='cuLaunchKernelEx',ts=correlation,args={'correlation':correlation}),dict(cat='kernel',name=name,ts=correlation,args={'correlation':correlation})])
 return out

def run():
 with tempfile.TemporaryDirectory(dir='/data1/tzh/proj/qwen_operator_lab/.runtime-research/tmp') as tmp:
  for scenario in ['complete','missing_compute','missing_activity']:
   missing=scenario=='missing_compute'
   counts=dict(COUNTS)
   if missing:counts['_inference_gemv']-=1
   data=events(counts)
   if scenario=='missing_activity':data.append(dict(cat='cuda_driver',name='cuLaunchKernelEx',ts=99999,args={'correlation':99999}))
   with patch('torch.profiler.profile',return_value=Profiler(data)),patch('torch.cuda.synchronize'):
    _,report=trace(Path(tmp),'decode',lambda:None,36)
   assert report['passed']==(scenario=='complete'),report
 print('Complete decode accepted; missing-GEMV and missing-GPU-activity decode rejected')
if __name__=='__main__':run()
