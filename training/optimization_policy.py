"""Explicit research switches; never silently replace the default training path."""
import json
from pathlib import Path

MODES=('off','embedding','embedding_consumed','multi','all','multi_embedding','all_embedding')


def prepare(model, *, mode, optimizer_name, lr, seq, embedding_gate=None,batch=1,accumulation=1):
    if mode not in MODES: raise ValueError(f'Unknown optimization mode: {mode}')
    if mode!='off' and optimizer_name!='flashoptim':
        raise ValueError('These experiments were validated only with FlashOptim')
    undo=lambda:None
    if mode in ('embedding','embedding_consumed','multi_embedding','all_embedding'):
        if embedding_gate is None: raise ValueError('Embedding mode requires an explicit correctness-gate JSON')
        gate=json.loads(Path(embedding_gate).read_text())
        if not (gate.get('passed') and gate.get('deterministic') and
                gate.get('seq')==seq and gate.get('batch',1)==batch):
            raise ValueError('Embedding gate does not validate this batch/seq training shape')
        if accumulation!=gate.get('accumulation',1):
            raise ValueError('Embedding gate does not validate this accumulation setting')
        if mode=='embedding_consumed' and not gate.get('consume_proxy'):
            raise ValueError('Consumed embedding mode requires its own consume_proxy gate')
        # Promoted from the independently tested research harness; still opt-in.
        from training.tied_embedding import patch_tied_embedding as patch
        undo=patch(model,consume_proxy=mode=='embedding_consumed')
    if mode in ('multi','all','multi_embedding','all_embedding'):
        from training.multi_tensor_flashoptim import MultiTensorFlashAdamW
        all_tensors=mode in ('all','all_embedding')
        optimizer=MultiTensorFlashAdamW(model.parameters(),lr=lr,betas=(.9,.999),eps=1e-8,
            weight_decay=.01,master_weight_bits=None,decouple_lr=False,
            max_numel=2**31-1 if all_tensors else 65536,persistent=all_tensors)
    else:
        from benchmarks.research_candidates import make_optimizer
        optimizer=make_optimizer(model,optimizer_name,lr)
    return optimizer,undo
