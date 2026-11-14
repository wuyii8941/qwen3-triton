"""Fail immediately if explicit Triton model arithmetic falls back to ATen.

Allocation, views, layout copies, parameter casts, autograd accumulation and
the shared FSDP/NCCL infrastructure remain framework operations. Pair this
dispatch guard with actual CUDA traces; the guard alone is not a trace audit.
"""
from torch.utils._python_dispatch import TorchDispatchMode

FORBIDDEN = {
    'aten::mm', 'aten::bmm', 'aten::addmm', 'aten::matmul', 'aten::linear',
    'aten::embedding', 'aten::embedding_dense_backward', 'aten::silu',
    'aten::silu_backward', 'aten::rsqrt', 'aten::pow', 'aten::mean',
    'aten::sum', 'aten::sin', 'aten::cos', 'aten::mul', 'aten::div',
    'aten::_softmax', 'aten::_log_softmax', 'aten::nll_loss_forward',
    'aten::nll_loss_backward', 'aten::_scaled_dot_product_flash_attention',
    'aten::_scaled_dot_product_efficient_attention',
    'aten::_scaled_dot_product_attention_math', 'aten::scaled_dot_product_attention',
}

class NoFallback(TorchDispatchMode):
    def __init__(self, backward=False):
        super().__init__()
        self.backward = backward

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func._schema.name
        if name in FORBIDDEN or (not self.backward and name in {'aten::add', 'aten::add_'}):
            raise AssertionError(f'Model compute fallback: {func}')
        return func(*args, **(kwargs or {}))
