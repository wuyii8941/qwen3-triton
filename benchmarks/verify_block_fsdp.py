"""Multi-rank Qwen3 block-sharding parity, tied and untied weights."""
import os
import copy
import json
import argparse
import tempfile
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
from transformers import Qwen3Config, Qwen3ForCausalLM
from training.block_fsdp import shard_qwen, BlockFSDP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--checkpoint-test', action='store_true')
    parser.add_argument('--export-test', action='store_true')
    options = parser.parse_args()
    if options.export_test and not options.bf16: parser.error('Export comparison requires --bf16')
    dtype = torch.bfloat16 if options.bf16 else torch.float32
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl', timeout=timedelta(seconds=180))
    try:
        for tied in (False, True):
            torch.manual_seed(8)
            config = Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=96,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                head_dim=16, tie_word_embeddings=tied, attention_dropout=0.)
            config._attn_implementation = 'sdpa'
            cpu = Qwen3ForCausalLM(config)
            reference = copy.deepcopy(cpu).cuda().train()
            model = shard_qwen(cpu).train()
            for module in model.modules():
                if isinstance(module, BlockFSDP): module.compute_dtype = dtype
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
            opt_ref = torch.optim.AdamW(reference.parameters(), lr=1e-3, foreach=False)
            for step in range(3):
                torch.manual_seed(400 + dist.get_rank()*20 + step)
                ids = torch.randint(0, 97, (2, 11), device='cuda')
                opt.zero_grad(set_to_none=True); opt_ref.zero_grad(set_to_none=True)
                actual = model(ids, labels=ids, use_cache=False)
                # Match per-unit BF16 casts: tied embed/head share an FP32
                # master, but each use casts separately before backward.
                compute_parameters = {n:p.to(dtype) for n,p in reference.named_parameters(remove_duplicate=False)}
                expected = torch.func.functional_call(reference,compute_parameters,(ids,),
                    dict(labels=ids,use_cache=False),tie_weights=False)
                torch.testing.assert_close(actual.logits, expected.logits, atol=2e-5, rtol=2e-4)
                torch.testing.assert_close(actual.loss, expected.loss)
                actual.loss.backward(); expected.loss.backward()
                for p in reference.parameters():
                    dist.all_reduce(p.grad); p.grad.div_(dist.get_world_size())
                for name, unit in model.named_modules():
                    if not isinstance(unit, BlockFSDP): continue
                    ref_unit = reference.get_submodule(name)
                    target = torch.cat([p.grad.reshape(-1) for p in ref_unit.parameters()])
                    # Gradients/master values must be inspected without BF16 casting.
                    old_dtype = unit.compute_dtype; unit.compute_dtype = torch.float32
                    observed = unit.gather(unit.shard.grad)[:unit.numel]
                    unit.compute_dtype = old_dtype
                    torch.testing.assert_close(observed, target, atol=2e-5, rtol=2e-3)
                opt.step(); opt_ref.step()
                if options.checkpoint_test and step == 0:
                    from training.sharded_checkpoint import save_sharded, load_sharded
                    location = [tempfile.mkdtemp(prefix='fsdp-checkpoint-') if dist.get_rank()==0 else None]
                    dist.broadcast_object_list(location,src=0)
                    checkpoint = Path(location[0])/'step1'
                    metadata = dict(tied=tied,dtype=str(dtype))
                    save_sharded(checkpoint,model,opt,1,metadata)
                    with torch.no_grad():
                        for p in model.parameters(): p.add_(5)
                    opt.state.clear()
                    assert load_sharded(checkpoint,model,opt,metadata) == 1
                for name, unit in model.named_modules():
                    if not isinstance(unit, BlockFSDP): continue
                    target = torch.cat([p.detach().reshape(-1) for p in reference.get_submodule(name).parameters()])
                    old_dtype = unit.compute_dtype; unit.compute_dtype = torch.float32
                    observed = unit.gather(unit.shard.detach())[:unit.numel]
                    unit.compute_dtype = old_dtype
                    torch.testing.assert_close(observed, target, atol=3e-5, rtol=3e-3)
            if options.export_test:
                from training.export_sharded_hf import export_hf
                location = [tempfile.mkdtemp(prefix='fsdp-export-') if dist.get_rank()==0 else None]
                dist.broadcast_object_list(location,src=0)
                destination = Path(location[0])/'hf'
                export_hf(model,destination,Path(location[0]))
                reloaded = Qwen3ForCausalLM.from_pretrained(destination,torch_dtype=dtype,
                    attn_implementation='sdpa').cuda().eval()
                with torch.no_grad():
                    model.eval()
                    torch.testing.assert_close(model(ids,use_cache=False).logits,
                        reloaded(ids,use_cache=False).logits,atol=2e-5,rtol=2e-4)
                del reloaded
            if dist.get_rank() == 0:
                print(json.dumps(dict(passed=True,tied=tied,dtype=str(dtype),world=dist.get_world_size(),
                    checkpoint=options.checkpoint_test,export=options.export_test,steps=3)),flush=True)
            del model, reference, opt, opt_ref
    finally:
        dist.destroy_process_group()


if __name__ == '__main__': main()
