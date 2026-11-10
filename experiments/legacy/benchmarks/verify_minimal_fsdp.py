"""Run using torch.distributed.run --standalone --nproc_per_node=2."""
import os
import copy
import json
from datetime import timedelta
import torch
import torch.distributed as dist
from training.minimal_fsdp import MinimalFSDP


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(42)
        reference = torch.nn.Sequential(torch.nn.Linear(13, 23), torch.nn.SiLU(),
                                        torch.nn.Linear(23, 7)).cuda()
        wrapped = MinimalFSDP(copy.deepcopy(reference))
        opt_ref = torch.optim.AdamW(reference.parameters(), lr=1e-3, foreach=False)
        opt = torch.optim.AdamW(wrapped.parameters(), lr=1e-3, foreach=False)
        maximum = 0.
        for step in range(3):
            torch.manual_seed(100 + rank * 10 + step)
            x = torch.randn(5, 13, device='cuda', requires_grad=True)
            xr = x.detach().clone().requires_grad_(True)
            target = torch.randn(5, 7, device='cuda')
            opt.zero_grad(set_to_none=True); opt_ref.zero_grad(set_to_none=True)
            y, yr = wrapped(x), reference(xr)
            torch.testing.assert_close(y, yr, atol=2e-6, rtol=2e-5)
            loss, loss_ref = (y-target).square().mean(), (yr-target).square().mean()
            torch.testing.assert_close(loss, loss_ref)
            loss.backward(); loss_ref.backward()
            torch.testing.assert_close(x.grad, xr.grad, atol=2e-6, rtol=2e-5)
            for p in reference.parameters():
                dist.all_reduce(p.grad)
                p.grad.div_(dist.get_world_size())
            expected = torch.cat([p.grad.flatten() for p in reference.parameters()])
            actual = wrapped.gather(wrapped.shard.grad)[:wrapped.numel]
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            opt.step(); opt_ref.step()
            expected = torch.cat([p.detach().flatten() for p in reference.parameters()])
            actual = wrapped.gather(wrapped.shard.detach())[:wrapped.numel]
            maximum = max(maximum, (actual-expected).abs().max().item())
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        if rank == 0:
            print(json.dumps(dict(passed=True, world_size=dist.get_world_size(), steps=3,
                parameter_count=wrapped.numel, local_parameter_count=wrapped.shard.numel(),
                max_parameter_error=maximum)), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
