"""One monitored lease for norm regression, failed-request regression and full gates."""
import contextlib,gc,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from scripts import diagnose_qwen3_error_growth_v15 as harness
from scripts import diagnose_qwen3_reference_request_v15 as request
from scripts import check_reference_norm_v15 as norm
from scripts import verify_qwen3_inference_v15 as verifier
from scripts import check_inference_strided_ops_v15 as strided
from scripts import compare_qwen3_graph_v15 as paired

ROOT=Path(__file__).resolve().parents[1]
FILES=[Path(__file__),Path(request.__file__),Path(norm.__file__),Path(harness.__file__),Path(strided.__file__),Path(paired.__file__)]
HASHES={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in FILES}

def run(a,out,foreign):
 # Outer harness already owns the stable physicalGPU1 lease and monitor.
 strided.run(out/'strided.json')
 with patch.object(norm,'stable_lease',lambda devices:contextlib.nullcontext()),patch.object(sys,'argv',['check_reference_norm','--output',str(out/'norm.json')]):norm.main()
 target=out/'request';target.mkdir();request.run(a,target,foreign)
 gc.collect();torch.cuda.empty_cache()
 target=out/'full';target.mkdir()
 args=SimpleNamespace(model='/data1/tzh/models/Qwen/Qwen3-8B',identity='/data1/tzh/proj/qwen_operator_lab/artifacts/official_qwen3_inputs_v1.json',reference_accumulation=True,prefill_tiling=True,sliced_decode=False,warmup_collection=True,diagnostic=False,physical_device=1,corpus_blocks=512,corpus_sequence=512)
 verifier.run(args,target,foreign)
 gc.collect();torch.cuda.empty_cache()
 benchmark=out/'paired';benchmark.mkdir()
 options=SimpleNamespace(include_reference=True,include_sliced=False,reference_prefill_tiling=True,qualification=target/'result.json')
 paired.run(options,benchmark,foreign)
 for p in FILES:assert hashlib.sha256(p.read_bytes()).hexdigest()==HASHES[str(p.relative_to(ROOT))],p
 (out/'source_receipt.json').write_text(json.dumps(dict(source_sha256=HASHES,foreign_context_observations=list(foreign)),indent=2)+'\n')

if __name__=='__main__':harness.run=run;harness.main()
