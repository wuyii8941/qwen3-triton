"""Native vs exact Triton vs W8A16 on the checkpoint's real tied LM head."""
import hashlib
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from torch.nn.attention import sdpa_kernel,SDPBackend
from ops.triton.linear_decode import linear_decode
from ops.triton.weight_only_gemv import quantize_groupwise
from benchmarks.benchmark_kernel_tuning import run_case


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--shared',action='store_true')
    args=parser.parse_args()
    target=Path('artifacts/lm_head_tuning_shared.json' if args.shared else 'artifacts/lm_head_tuning.json')
    if target.exists(): raise FileExistsError(target)
    torch.manual_seed(20261005)
    model=AutoModelForCausalLM.from_pretrained('/data1/tzh/models/Qwen/Qwen3-1.7B',
        local_files_only=True,torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
    weight=model.lm_head.weight.detach()
    snapshots={g:quantize_groupwise(weight,g) for g in (16,32,64)}
    rows=[]
    errors=[]
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for batch in (1,2,4):
            ids=torch.randint(model.config.vocab_size,(batch,32),device='cuda')
            x=model.model(ids,use_cache=False).last_hidden_state[:,-1].contiguous()
            ref=F.linear(x,weight)
            variants={'native':lambda:F.linear(x,weight)}
            for bn in (1,2,4,8,16):
                for warps in (4,8):
                    variants[f'fp16_n{bn}_w{warps}']=lambda bn=bn,warps=warps:linear_decode(x,weight,block_n=bn,num_warps=warps,share_batch=args.shared)
            rows+=run_case(variants,ref,dict(batch=batch,scope='original_fp16',n=weight.shape[0],k=weight.shape[1]))
            for g,packed in snapshots.items():
                reconstructed=packed.dequantize()
                oracle=F.linear(x,reconstructed)
                errors.append(dict(batch=batch,group=g,
                    relative_l2=float((oracle.float()-ref.float()).norm()/ref.float().norm()),
                    max_abs=float((oracle-ref).abs().max()),argmax_matches=bool(torch.equal(oracle.argmax(-1),ref.argmax(-1)))))
                del reconstructed
                variants={f'w8g{g}_n{bn}_w{warps}':
                    (lambda bn=bn,warps=warps,packed=packed:linear_decode(x,packed,block_n=bn,num_warps=warps,share_batch=args.shared))
                    for bn in (1,2,4,8,16) for warps in (4,8)}
                rows+=run_case(variants,oracle,dict(batch=batch,scope='dequantized_oracle',group=g,n=weight.shape[0],k=weight.shape[1]))
            target.write_text(json.dumps(dict(seed=20261005,shared_batch=args.shared,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                rows=rows,quantization_error=errors,extra_snapshot_bytes={g:w.nbytes for g,w in snapshots.items()},
                source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                    ['ops/triton/linear_decode.py','ops/triton/weight_only_gemv.py',__file__]}),indent=2)+'\n')
    for batch in (1,2,4):
        print(batch,[(r['backend'],round(r['median_us'],2)) for r in sorted(
            (r for r in rows if r['batch']==batch),key=lambda r:r['median_us'])[:8]],flush=True)


if __name__=='__main__': main()
