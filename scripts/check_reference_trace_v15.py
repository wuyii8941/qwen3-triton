"""CPU trace regression for reference-order phases and missing GPU work."""
import tempfile
from pathlib import Path
from unittest.mock import patch
from scripts.check_inference_trace_integrity_v15 import Profiler,events
from scripts.verify_qwen3_inference_v15 import trace
BASE={'_inference_positions':1,'_lookup':1,'_tables':1,'_hf_norm_forward':73,'_hf_norm_rope_forward':72,'_inference_cache_append':36,'_inference_cached_attention':36,'_add':72}
P65={**BASE,'_split_gemm':180,'_inference_reduce_parts':180,'_gate_up':36,'_inference_gemv':1}
P17={**BASE,'_gemm':108,'_inference_interleaved_linear':72,'_split_gemm':72,'_inference_reduce_parts':72,'_hf_swiglu_forward':36,'_inference_gemv':1}
DECODE={**BASE,'_hf_norm_forward':0,'_inference_reference_norm':73,'_inference_interleaved_linear':108,'_inference_reference_gemv':144,'_inference_gemv':1,'_hf_swiglu_forward':36}
GENERATE={key:value*3 for key,value in BASE.items()}
GENERATE.update(_hf_norm_forward=73,_inference_reference_norm=146,_split_gemm=180,_inference_reduce_parts=180,_gate_up=36,_inference_gemv=3,_inference_interleaved_linear=216,_inference_reference_gemv=288,_hf_swiglu_forward=72,_inference_argmax_parts=3,_inference_select=3)

def run():
 with tempfile.TemporaryDirectory(dir='/data1/tzh/proj/qwen_operator_lab/.runtime-research/tmp') as tmp:
  # Counts alone cannot establish that the intended phase was actually captured.
  old={**BASE,'_inference_gemv':253,'_hf_swiglu_forward':36}
  with patch('torch.profiler.profile',return_value=Profiler(events(old))),patch('torch.cuda.synchronize'):
   _,report=trace(Path(tmp),'decode',lambda:None,36)
  assert not report['passed'],'Trace without an actual-phase annotation was accepted'
  for phase,counts,length,appends in [('prefill',P65,65,36),('prefill17',P17,17,36),('decode',DECODE,65,36),('generate',GENERATE,65,108)]:
   scenarios=['complete','missing_compute','missing_activity','missing_scope','native_compute','wrong_route']
   if counts.get('_inference_reduce_parts'):scenarios+=['missing_partial','missing_reduction']
   for scenario in scenarios:
    data=events(counts)
    if scenario in ['missing_compute','missing_partial','missing_reduction']:
     missing='_split_gemm' if scenario=='missing_partial' else '_inference_reduce_parts' if scenario=='missing_reduction' else '_inference_gemv'
     reduced=dict(counts);reduced[missing]-=1;data=events(reduced)
    if scenario=='wrong_route':
     if phase=='decode':wrong={**BASE,'_inference_gemv':253,'_hf_swiglu_forward':36}
     elif phase=='generate':
      wrong={key:value*3 for key,value in BASE.items()};wrong.update(_gemm=180,_gate_up=36,_inference_gemv=507,_hf_swiglu_forward=72,_inference_argmax_parts=3,_inference_select=3)
     else:wrong={**BASE,'_gemm':180,'_gate_up':36,'_inference_gemv':1}
     data=events(wrong)
    if scenario=='missing_activity':data.append(dict(cat='cuda_driver',name='cuLaunchKernelEx',ts=.5,args={'correlation':99999}))
    if scenario=='native_compute':
     data.extend([dict(cat='cuda_driver',name='cuLaunchKernelEx',ts=.5,args={'correlation':99999}),dict(cat='kernel',name='ampere_bf16_s16816gemm_bf16_native',ts=.5,args={'correlation':99999})])
    if scenario!='missing_scope':data.append(dict(cat='user_annotation',name='actual_'+phase,ts=0,dur=100000))
    with patch('torch.profiler.profile',return_value=Profiler(data)),patch('torch.cuda.synchronize'):
     _,report=trace(Path(tmp),phase,lambda:None,appends,reference_accumulation=True,prompt_tokens=length)
    assert report['passed']==(scenario=='complete'),(phase,scenario,report)
 print('Reference phases accepted; missing scope/compute/partials/reductions/activity and native fallback rejected')
if __name__=='__main__':run()
