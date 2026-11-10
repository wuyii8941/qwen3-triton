"""Ablate last-token view strides, including copy cost and operator dispatch."""
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.profiler import profile, ProfilerActivity
from transformers import AutoModelForCausalLM
from ops.triton.linear_decode import linear_decode
from benchmarks.benchmark_kernel_tuning import run_case


@torch.inference_mode()
def main():
    target = Path('artifacts/lm_head_layout.json')
    if target.exists():
        raise FileExistsError(target)
    torch.manual_seed(20261008)
    model = AutoModelForCausalLM.from_pretrained('/data1/tzh/models/Qwen/Qwen3-1.7B',
        local_files_only=True, torch_dtype=torch.float16, attn_implementation='sdpa').cuda().eval()
    weight = model.lm_head.weight
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, seed=20261008,
        scope='LM head only; same original weights; last-token view from real HF hidden states; copy cost included',
        rows=[], dispatch={}, source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in [__file__, 'ops/triton/linear_decode.py']})
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for batch in (1, 2, 4):
            ids = torch.randint(model.config.vocab_size, (batch, 128), device='cuda')
            hidden = model.model(ids, use_cache=False).last_hidden_state
            x = hidden[:, -1:]
            reference = F.linear(x, weight)
            variants = {
                'native_strided': lambda:F.linear(x, weight),
                'native_contiguous': lambda:F.linear(x.contiguous(), weight),
                'triton_fp16': lambda:linear_decode(x.reshape(batch, -1).contiguous(), weight,
                    block_n=1, num_warps=4 if batch>=4 else 8, share_batch=True).view_as(reference),
            }
            report['rows'] += run_case(variants, reference,
                dict(batch=batch, seq=128, input_stride=list(x.stride()), input_contiguous=x.is_contiguous()))
            report['dispatch'][batch] = {}
            for name, fn in variants.items():
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    fn(); torch.cuda.synchronize()
                report['dispatch'][batch][name] = {
                    event.key:event.count for event in prof.key_averages()
                    if event.key in ('aten::mm', 'aten::bmm', 'aten::clone', 'aten::copy_', 'aten::contiguous')}
            target.write_text(json.dumps(report, indent=2)+'\n')
    print([(r['batch'], r['backend'], round(r['median_us'], 2)) for r in report['rows']], flush=True)
    print(target, flush=True)


if __name__ == '__main__':
    main()
