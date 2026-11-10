"""Independent compiled training ablations; no checkpoint compatibility claim.

Flat optimizer includes gradient concatenation cost. Quantization block
boundaries change, so it is an approximate candidate, not exact launch fusion.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
from types import MethodType
import torch


class FlatFlash:
    def __init__(self,model,lr):
        from flashoptim import FlashAdamW
        self.params=[p for p in model.parameters() if p.requires_grad]
        self.flat=torch.nn.Parameter(torch.cat([p.detach().reshape(-1) for p in self.params]))
        offset=0
        for p in self.params:
            p.data=self.flat.data[offset:offset+p.numel()].view_as(p);offset+=p.numel()
        self.optimizer=FlashAdamW([self.flat],lr=lr,master_weight_bits=None,weight_decay=.01)
    def zero_grad(self,set_to_none=True):
        for p in self.params:p.grad=None
        self.flat.grad=None
    def step(self):
        if any(p.grad is None for p in self.params):raise RuntimeError('Flat experiment requires all gradients')
        self.flat.grad=torch.cat([p.grad.reshape(-1) for p in self.params])
        self.optimizer.step()
    def state_dict(self):return self.optimizer.state_dict()


def packed_forward(self,x):
    gate,up=self.packed_gate_up(x).chunk(2,dim=-1)
    return self.down_proj(torch.nn.functional.silu(gate)*up)


def swiglu_forward(self,x):
    from training.hf_rounded import activation_triton_backward
    return self.down_proj(activation_triton_backward(self.gate_proj(x),self.up_proj(x)))


def patch_model(model,variant):
    for layer in model.model.layers:
        module=layer.mlp
        if variant=='packed':
            gate,up=module.gate_proj,module.up_proj
            if gate.bias is not None or up.bias is not None:raise ValueError('Bias unsupported')
            linear=torch.nn.Linear(gate.in_features,2*gate.out_features,bias=False,
                                   device=gate.weight.device,dtype=gate.weight.dtype)
            with torch.no_grad():linear.weight.copy_(torch.cat([gate.weight,up.weight]))
            module.packed_gate_up=linear
            del module.gate_proj;del module.up_proj
            module.forward=MethodType(packed_forward,module)
        elif variant=='swiglu':module.forward=MethodType(swiglu_forward,module)
    return model


def worker(variant,output):
    import benchmarks.screen_research_candidates as screen
    from torch.nn.attention import sdpa_kernel,SDPBackend
    original_load=screen.load_model;original_opt=screen.make_optimizer
    screen.load_model=lambda:patch_model(original_load(),variant)
    if variant=='flat':screen.make_optimizer=lambda model,v,lr:FlatFlash(model,lr)
    torch.set_num_threads(4);torch.manual_seed(20261106)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        result=screen.train_screen('compile_flashoptim',1e-5,512,64)
    result.update(ablation=variant,warning='Short trajectory only; compare full validation curve before adopting. Flat changes quantization groups.')
    with output.open('x') as out:json.dump(result,out,indent=2)


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=['baseline','flat','packed','swiglu'])
    p.add_argument('--output',type=Path);a=p.parse_args()
    if a.worker:return worker(a.worker,a.output)
    root=Path('artifacts/training_fusion_ablation_v1');root.mkdir(exist_ok=False)
    plan=dict(rounds=3,steps=64,variants=['baseline','flat','packed','swiglu'],
              validation_abs_gate=.01,scope='Compiled HF+FlashOptim, short screen, gradient-cat included; no long jobs or checkpoints')
    (root/'plan.json').write_text(json.dumps(plan,indent=2));records=[]
    for round_id in range(3):
        order=plan['variants'][round_id:]+plan['variants'][:round_id]
        for variant in order:
            name=f'r{round_id}_{variant}';print('START',name,flush=True);tick=time.perf_counter()
            with (root/f'{name}.log').open('x') as log:
                code=subprocess.run([sys.executable,'-m','benchmarks.test_training_fusions','--worker',variant,
                    '--output',str(root/f'{name}.json')],stdout=log,stderr=subprocess.STDOUT).returncode
            records.append(dict(name=name,returncode=code,wall_seconds=time.perf_counter()-tick))
            (root/'status.json').write_text(json.dumps(dict(records=records,complete=False),indent=2))
            print('END',name,code,flush=True)
    (root/'status.json').write_text(json.dumps(dict(records=records,complete=True),indent=2))

if __name__=='__main__':main()
