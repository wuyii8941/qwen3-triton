"""Opt-in candidate adapters, used only by independent research screens."""
import importlib.util
from pathlib import Path

import torch


def partition_parameters(model):
    """Deduplicate tied embedding/head weights; orthogonalize hidden matrices only."""
    excluded={id(p) for module in (model.get_input_embeddings(),model.get_output_embeddings())
              if module is not None for p in module.parameters()}
    hidden=[];aux=[];seen=set()
    for p in model.parameters():
        if id(p) in seen or not p.requires_grad:continue
        seen.add(id(p))
        (hidden if p.ndim==2 and id(p) not in excluded else aux).append(p)
    assert not ({id(p) for p in hidden}&{id(p) for p in aux})
    return hidden,aux


def tensor_storage_bytes(value):
    seen=set()
    def visit(x):
        if isinstance(x,torch.Tensor):
            storage=x.untyped_storage();key=(str(x.device),storage.data_ptr())
            if key in seen:return 0
            seen.add(key);return storage.nbytes()
        if isinstance(x,dict):return sum(visit(v) for v in x.values())
        if isinstance(x,(tuple,list)):return sum(visit(v) for v in x)
        return 0
    return visit(value)


class OptimizerPair:
    def __init__(self,*optimizers):self.optimizers=optimizers
    def step(self):
        for optimizer in self.optimizers:optimizer.step()
    def zero_grad(self,set_to_none=True):
        for optimizer in self.optimizers:optimizer.zero_grad(set_to_none=set_to_none)
    def state_dict(self):return [o.state_dict() for o in self.optimizers]


def make_optimizer(model,variant,lr,aux_lr=1e-5):
    if variant=='triton':
        from training.triton_adamw import TritonAdamW
        return TritonAdamW(model.parameters(),lr=lr,math_mode='rounded')
    if variant=='flashoptim':
        from flashoptim import FlashAdamW
        # Match current BF16/no-master baseline. Do not silently add 24-bit
        # master weights, cast normalization/head to FP32, or release gradients.
        return FlashAdamW(model.parameters(),lr=lr,betas=(.9,.999),eps=1e-8,
                          weight_decay=.01,master_weight_bits=None,decouple_lr=False)
    if variant in ('muon','normuon'):
        filename=('Muon/muon.py' if variant=='muon' else 'NorMuon/normuon.py')
        path=Path('artifacts/research_sources_v1')/filename
        spec=importlib.util.spec_from_file_location(f'official_{variant}',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        hidden,aux=partition_parameters(model)
        cls=module.SingleDeviceMuon if variant=='muon' else module.SingleDeviceNorMuon
        return OptimizerPair(cls(hidden,lr=lr,weight_decay=.01,momentum=.95),
            torch.optim.AdamW(aux,lr=aux_lr,betas=(.9,.999),eps=1e-8,weight_decay=.01,fused=True))
    return torch.optim.AdamW(model.parameters(),lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=.01,fused=True)


def linear_loss(variant,e,w,labels):
    if variant=='cce':
        from cut_cross_entropy import linear_cross_entropy
        return linear_cross_entropy(e,w,labels,filter_eps=None)
    if variant=='liger':
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        return LigerFusedLinearCrossEntropyLoss()(w,e,labels)
    return torch.nn.functional.cross_entropy(torch.nn.functional.linear(e,w).float(),labels)


def model_loss(model,tokens,variant):
    if variant not in ('cce','liger'):return model(tokens,labels=tokens,use_cache=False).loss
    hidden=model.model(tokens,use_cache=False).last_hidden_state
    e=hidden[:,:-1,:].reshape(-1,hidden.shape[-1])
    labels=tokens[:,1:].reshape(-1)
    return linear_loss(variant,e,model.get_output_embeddings().weight,labels)
