"""Short varied-data/held-out validation gate, NOT convergence or test selection."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.data import DataLoader
from benchmarks.check_tied_embedding import patch
from benchmarks.research_candidates import make_optimizer
from benchmarks.screen_research_candidates import load_model
from training.flashoptim_convergence_job import Blocks
from training.multi_tensor_flashoptim import MultiTensorFlashAdamW


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=['baseline','both','graph_embedding','consumed_embedding'])
    parser.add_argument('--output')
    parser.add_argument('--graph',action='store_true')
    parser.add_argument('--consumed',action='store_true')
    args=parser.parse_args()
    root=Path('artifacts/consumed_embedding_trajectory_v1' if args.consumed else ('artifacts/graphed_trajectory_v1' if args.graph else 'artifacts/optimized_trajectory_v1'))
    if args.variant is None:
        root.mkdir(exist_ok=False)
        status=[]
        candidate_variant='consumed_embedding' if args.consumed else ('graph_embedding' if args.graph else 'both')
        for variant in ('baseline',candidate_variant):
            with (root/f'{variant}.log').open('w') as stream:
                result=subprocess.run([sys.executable,'-m','benchmarks.check_optimized_trajectory',
                    '--variant',variant,'--output',str(root/f'{variant}.json')],stdout=stream,stderr=subprocess.STDOUT)
            status.append(dict(variant=variant,returncode=result.returncode))
            (root/'status.json').write_text(json.dumps(status,indent=2))
            print('TRAJECTORY',variant,result.returncode,flush=True)
            if result.returncode: return
        baseline=json.loads((root/'baseline.json').read_text())
        candidate=json.loads((root/f'{candidate_variant}.json').read_text())
        differences=[abs(a['validation_loss']-b['validation_loss']) for a,b in zip(baseline['curve'],candidate['curve'])]
        result=dict(max_validation_abs_difference=max(differences),passed=max(differences)<=.01,
                    differences=differences,steps=256,eval_blocks=16,
                    scope='Single short trajectory pair; not convergence, no held-out test access or checkpoint files.')
        (root/'comparison.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
        return
    torch.set_num_threads(4);torch.manual_seed(20261102)
    model=load_model()
    undo=patch(model,consume_proxy=args.variant=='consumed_embedding') if args.variant in ('both','graph_embedding','consumed_embedding') else lambda:None
    options=dict(lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01,master_weight_bits=None,decouple_lr=False)
    optimizer=MultiTensorFlashAdamW(model.parameters(),**options) if args.variant=='both' else make_optimizer(model,'flashoptim',1e-5)
    def loss_fn(batch): return model(batch,labels=batch,use_cache=False).loss
    forward=torch.compile(loss_fn,fullgraph=True,dynamic=False)
    runner=None
    corpus=Path('artifacts/convergence_corpus_v1')
    train=DataLoader(Blocks(corpus/'train.bin',512,training=True,seed=20261102),batch_size=1,num_workers=2,pin_memory=True)
    validation=Blocks(corpus/'validation.bin',512)
    validation_batches=[validation[i].unsqueeze(0).cuda() for i in range(16)]
    report=dict(variant=args.variant,steps=256,seq=512,batch=1,curve=[],training_samples=[],torch=torch.__version__)
    start=time.perf_counter()
    def evaluate(step):
        model.eval()
        with torch.no_grad(): values=[float(loss_fn(batch)) for batch in validation_batches]
        model.train()
        report['curve'].append(dict(step=step,validation_loss=sum(values)/len(values),elapsed_seconds=time.perf_counter()-start))
        Path(args.output).write_text(json.dumps(report,indent=2))
        print('EVAL',args.variant,report['curve'][-1],flush=True)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        evaluate(0)
        for step,batch in enumerate(train,1):
            batch=batch.cuda(non_blocking=True)
            if args.variant=='graph_embedding':
                if runner is None:
                    from training.graphed_forward_backward import GraphedForwardBackward
                    runner=GraphedForwardBackward(model,forward,batch)
                loss=runner.backward(batch)
            else:
                optimizer.zero_grad(set_to_none=True)
                loss=forward(batch)
                loss.backward()
            optimizer.step()
            value=float(loss.detach())
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
            if step%16==0: report['training_samples'].append(dict(step=step,loss=value))
            if step%64==0: evaluate(step)
            if step==256: break
    report.update(complete=True,elapsed_seconds=time.perf_counter()-start)
    Path(args.output).write_text(json.dumps(report,indent=2))
    undo()


if __name__=='__main__': main()
