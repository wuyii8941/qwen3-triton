"""Reusable CPU best-parameter storage, not an optimizer/resume checkpoint."""
import torch


def local_tensor(parameter):
    return parameter.to_local() if hasattr(parameter,'to_local') else parameter


class BestParameters:
    def __init__(self):
        self.values={}

    @torch.no_grad()
    def save(self,model):
        for name,parameter in model.named_parameters():
            local=local_tensor(parameter)
            if name not in self.values:
                self.values[name]=torch.empty_like(local,device='cpu')
            self.values[name].copy_(local.detach())

    @torch.no_grad()
    def restore(self,model):
        if set(self.values)!=set(dict(model.named_parameters())):
            raise ValueError('Best parameter schema mismatch')
        for name,parameter in model.named_parameters():
            if local_tensor(parameter).shape!=self.values[name].shape:
                raise ValueError('Best parameter shape mismatch')
        for name,parameter in model.named_parameters():
            local_tensor(parameter).copy_(self.values[name])
