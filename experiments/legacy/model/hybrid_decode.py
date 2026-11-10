"""Optional split prefill/decode execution, with one shared static KV cache.

Both model contexts must remain active. The two module trees must share the
same original parameters; this is inference-only, single-request orchestration,
not simultaneous use of either runner or a serving scheduler.
"""
from model.static_decode import StaticDecodeRunner


class HybridDecodeRunner:
    def __init__(self, prefill_model, decode_model, batch_size, capacity):
        left = dict(prefill_model.named_parameters())
        right = dict(decode_model.named_parameters())
        if left.keys() != right.keys() or any(left[k] is not right[k] for k in left):
            raise ValueError('Prefill/decode module trees must share original parameters')
        if prefill_model.config.to_dict() != decode_model.config.to_dict():
            raise ValueError('Identical model configs required')
        self.decode_runner = StaticDecodeRunner(decode_model, batch_size, capacity)
        self.prefill_runner = StaticDecodeRunner(prefill_model, batch_size, capacity)
        # Replace the just-created empty cache before compilation/capture. The
        # prefill writes exactly the addresses consumed by decode; no KV copy.
        self.prefill_runner.cache = self.decode_runner.cache

    def compile_prefill(self, *, backend):
        import torch
        if self.decode_runner.graph is not None:
            raise ValueError('Configure compilation before capture')
        self.prefill_runner._forward = torch.compile(
            self.prefill_runner._forward, backend=backend, fullgraph=True, dynamic=False)

    def prefill(self, tokens):
        logits = self.prefill_runner.prefill(tokens)
        self.decode_runner.length = self.prefill_runner.length
        return logits

    def capture(self, sample_token):
        self.decode_runner.capture(sample_token)

    def decode(self, token):
        return self.decode_runner.decode(token)

    @property
    def cache(self):
        return self.decode_runner.cache
