"""Fixed-shape, single-stream training prototype: graph F/B, optimizer outside.

The optimizer and its step counter run normally on EVERY replay. No frozen
bias-correction value is captured. Gradient buffers are produced by a captured
backward whose .grad was None at capture; replay overwrites rather than adds.
No accumulation/checkpointing/dropout/cache support is claimed here.
"""
import torch


class GraphedForwardBackward:
    def __init__(self, model, loss_fn, example):
        if example.device.type!='cuda': raise ValueError('CUDA inputs required')
        if getattr(model,'is_gradient_checkpointing',False):
            raise ValueError('Activation checkpointing is not validated with this graph prototype')
        if getattr(model.config,'attention_dropout',0):
            raise ValueError('Nonzero dropout is not validated')
        self.model=model
        self.static_input=example.clone()
        self.params=[p for p in model.parameters() if p.requires_grad]
        head=getattr(model,'lm_head',None)
        if head is not None and hasattr(head,'_experimental_tied_proxy'):
            del head._experimental_tied_proxy
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                model.zero_grad(set_to_none=True)
                loss_fn(self.static_input).backward()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        model.zero_grad(set_to_none=True)
        # The experimental tied adapter caches a view with an autograd edge.
        # Release the warmup edge before changing from warmup to capture stream.
        head=getattr(model,'lm_head',None)
        if head is not None and hasattr(head,'_experimental_tied_proxy'):
            del head._experimental_tied_proxy
        torch.cuda.empty_cache()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss=loss_fn(self.static_input)
            self.loss.backward()
        self.loss=self.loss.detach()
        if head is not None and hasattr(head,'_experimental_tied_proxy'):
            del head._experimental_tied_proxy
        if any(p.grad is None for p in self.params):
            raise ValueError('All trainable parameters must receive gradients in the capture')
        self.grad_pointers=[p.grad.data_ptr() for p in self.params]
        self.param_pointers=[p.data_ptr() for p in self.params]

    def backward(self, batch):
        if not self.model.training: raise RuntimeError('Captured training graph requires model.train()')
        if (batch.shape!=self.static_input.shape or batch.dtype!=self.static_input.dtype
                or batch.device!=self.static_input.device):
            raise ValueError('Graph replay requires the captured input shape/dtype/device')
        if any(p.grad is None or p.grad.data_ptr()!=ptr for p,ptr in zip(self.params,self.grad_pointers)):
            raise RuntimeError('Captured gradient buffers replaced; do not zero_grad(set_to_none=True) between replays')
        if any(p.data_ptr()!=ptr for p,ptr in zip(self.params,self.param_pointers)):
            raise RuntimeError('Captured parameter buffers replaced')
        self.static_input.copy_(batch)
        self.graph.replay()
        return self.loss
