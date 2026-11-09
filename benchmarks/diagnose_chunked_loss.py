"""Small single-GPU arithmetic diagnostic, NOT FSDP or performance proof."""
import argparse
import copy
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from training.chunked_loss import chunked_linear_cross_entropy


def error(actual, expected):
    a, b = actual.float(), expected.float()
    delta = (a-b).abs()
    return dict(max_abs=delta.max().item(),
        relative_l2=((a-b).norm()/b.norm().clamp_min(1e-12)).item(),
        mismatched_at_gate=int((delta > 2e-5+2e-3*b.abs()).sum().item()),
        elements=a.numel())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qwen', action='store_true', help='Reproduce rank0 D128 Qwen arithmetic without FSDP')
    parser.add_argument('--input-seed', type=int, default=400)
    parser.add_argument('--backward-impl', choices=['default','native_ce','native_ce_padded'], default='default')
    parser.add_argument('--chunk-size', type=int, default=7)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous diagnostic')
    if args.qwen:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from training.fsdp_loss import install_chunked_qwen_loss
        torch.manual_seed(8)
        config = Qwen3Config(vocab_size=97, hidden_size=512, intermediate_size=768,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=128, tie_word_embeddings=False, attention_dropout=0.)
        config._attn_implementation = 'sdpa'
        reference = Qwen3ForCausalLM(config).cuda().train()
        model = install_chunked_qwen_loss(copy.deepcopy(reference), chunk_size=args.chunk_size,
                                         backward_impl=args.backward_impl)
        captured = {}
        def capture(name):
            def hook(module, args):
                captured[name] = args[0]
                args[0].retain_grad()
            return hook
        reference.lm_head.register_forward_pre_hook(capture('native'))
        model.lm_head.register_forward_pre_hook(capture('chunked'))
        torch.manual_seed(args.input_seed)
        ids = torch.randint(97, (2, 11), device='cuda')
        outputs = []
        for instance in (reference, model):
            casts = {n: p.to(torch.bfloat16) for n, p in instance.named_parameters()}
            output = torch.func.functional_call(instance, casts, (ids,), dict(labels=ids, use_cache=False))
            output.loss.backward()
            outputs.append(output)
        rows = [dict(name=n, **error(p.grad, reference.get_parameter(n).grad))
                for n, p in model.named_parameters()]
        result = dict(complete=True, device=torch.cuda.get_device_name(), input_seed=args.input_seed,
            backward_impl=args.backward_impl, chunk_size=args.chunk_size, rows=rows,
            loss_abs=float((outputs[0].loss-outputs[1].loss).detach().abs()),
            head_hidden=error(captured['chunked'], captured['native'][:, :-1]),
            head_hidden_gradient=error(captured['chunked'].grad, captured['native'].grad[:, :-1]),
            scope='Rank0-input D128 tiny Qwen BF16 arithmetic only; no collectives or optimizer')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
        return
    torch.manual_seed(8)
    x0 = torch.randn(2, 11, 512, device='cuda', dtype=torch.bfloat16)*.1
    w0 = torch.randn(97, 512, device='cuda', dtype=torch.bfloat16)*.04
    labels = torch.randint(97, (2, 11), device='cuda')
    rows = []
    for upstream in (1., .37):
        reference_logits = F.linear(x0, w0)
        target = labels[:, 1:].reshape(-1)

        def evaluate(mode, chunk=0):
            x = x0.clone().requires_grad_(); w = w0.clone().requires_grad_()
            hidden = x[:, :-1].contiguous().reshape(-1, 512)
            if mode == 'hf_full_positions':
                logits = F.linear(x, w)[:, :-1].contiguous().reshape(-1, 97)
                loss = F.cross_entropy(logits.float(), target)
            elif mode == 'native_shifted_positions':
                logits = F.linear(hidden, w)
                loss = F.cross_entropy(logits.float(), target)
            else:
                loss = chunked_linear_cross_entropy(hidden, w, target, chunk_size=chunk)
            dx, dw = torch.autograd.grad(loss*upstream, (x, w))
            return loss.detach(), dx, dw

        expected = evaluate('hf_full_positions')
        for mode, chunk in [('native_shifted_positions', 0), ('chunked', 1), ('chunked', 7), ('chunked', 128)]:
            loss, dx, dw = evaluate(mode, chunk)
            hidden = x0[:, :-1].contiguous().reshape(-1, 512)
            size = chunk or len(hidden)
            logits = torch.cat([F.linear(hidden[i:i+size], w0) for i in range(0, len(hidden), size)])
            rows.append(dict(mode=mode, chunk_size=chunk, upstream=upstream,
                loss_abs=float((loss-expected[0]).abs()),
                logits=error(logits, reference_logits[:, :-1].reshape(-1, 97)),
                dx=error(dx, expected[1]), dw=error(dw, expected[2])))
    result = dict(complete=True, device=torch.cuda.get_device_name(), rows=rows,
        scope='Single-GPU BF16 arithmetic isolation; no FSDP, no optimizer or speedup claim')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
