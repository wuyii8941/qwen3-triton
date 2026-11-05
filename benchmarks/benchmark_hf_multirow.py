"""Verify optional multi-row Q/K norms on a real checkpoint (prefill only)."""
from contextlib import nullcontext
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch
from transformers import AutoModelForCausalLM
from torch.nn.attention import sdpa_kernel,SDPBackend
from model.hf_inference import qwen3_triton_inference
from benchmark_hf_inference import check


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='hf_multirow_ablation.json')
    args=parser.parse_args()
    torch.manual_seed(20260928)
    root=Path(__file__).resolve().parents[1]
    report=dict(manifest=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        model='/data1/tzh/models/Qwen/Qwen3-1.7B',seed=20260928,
        attention='forced Flash SDPA',timing='alternating synchronized wall-time, 20 samples',
        sources={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
                 ('model/hf_inference.py','ops/triton/rowwise_tuning.py','benchmarks/benchmark_hf_multirow.py')}),cases=[])
    for dtype in (torch.float16,torch.bfloat16):
        model=AutoModelForCausalLM.from_pretrained(report['manifest']['model'],local_files_only=True,
            torch_dtype=dtype,attn_implementation='sdpa').cuda().eval()
        def ctx(name):
            return nullcontext() if name=='hf' else qwen3_triton_inference(model,multirow_norm=name=='multirow')
        for b,s in ((1,128),(2,128),(1,1024)):
            ids=torch.randint(model.config.vocab_size,(b,s),device='cuda')
            case=dict(dtype=str(dtype),batch=b,seq=s,parity={},timings={})
            names=['hf','triton','multirow']
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                ref=None
                for name in names:
                    with ctx(name):
                        out=model(ids,use_cache=True,logits_to_keep=1,output_hidden_states=True)
                        if ref is None: ref=out
                        case['parity'][name]=dict(logits=check(out.logits,ref.logits),
                            hidden=[check(a,b) for a,b in zip(out.hidden_states,ref.hidden_states)],
                            argmax=bool((out.logits.argmax(-1)==ref.logits.argmax(-1)).all()))
                del ref,out
                samples={n:[] for n in names}
                for repeat in range(24):
                    for name in (names if repeat%2==0 else names[::-1]):
                        with ctx(name):
                            torch.cuda.synchronize()
                            start=time.perf_counter()
                            out=model(ids,use_cache=True,logits_to_keep=1)
                            torch.cuda.synchronize()
                            ms=(time.perf_counter()-start)*1000
                            if repeat>=4: samples[name].append(ms)
                            del out
                case['timings']={n:dict(median_ms=statistics.median(v),samples_ms=v) for n,v in samples.items()}
            print(dtype,b,s,{n:round(t['median_ms'],3) for n,t in case['timings'].items()},flush=True)
            report['cases'].append(case)
        del model
        torch.cuda.empty_cache()
    (root/'artifacts'/args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
