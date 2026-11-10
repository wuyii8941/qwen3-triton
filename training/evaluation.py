"""Exact full-block validation: identical collective count, no padded loss."""
import torch
import torch.nn.functional as F
import torch.distributed as dist


def validation_batches(dataset, rank, world, batch):
    if world<1 or batch<1 or not 0<=rank<world or len(dataset)<1:
        raise ValueError('Invalid validation partition')
    width=world*batch
    for start in range(0,len(dataset),width):
        indices=[start+rank*batch+j for j in range(batch)]
        valid=torch.tensor([index<len(dataset) for index in indices],dtype=torch.bool)
        # Every rank executes the same number and shape of forwards. Missing
        # blocks reuse a valid input but all their labels are ignored.
        ids=torch.stack([dataset[index] if index<len(dataset) else dataset[0] for index in indices])
        yield ids,valid


def shifted_nll(logits,ids,valid):
    labels=ids[:,1:].clone()
    labels[~valid]=-100
    total=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),
        labels.reshape(-1),ignore_index=-100,reduction='sum')
    count=valid.sum()*(ids.shape[1]-1)
    return total,count


@torch.no_grad()
def evaluate_full(model,dataset,batch=4):
    rank,world=dist.get_rank(),dist.get_world_size()
    training=model.training
    model.eval()
    totals=torch.zeros(2,device='cuda',dtype=torch.float64)
    try:
        for ids,valid in validation_batches(dataset,rank,world,batch):
            ids,valid=ids.cuda(),valid.cuda()
            logits=model(ids,use_cache=False).logits
            nll,count=shifted_nll(logits,ids,valid)
            totals[0]+=nll.double();totals[1]+=count.double()
        dist.all_reduce(totals)
        expected=len(dataset)*(dataset.seq-1)
        if totals[1].item()!=expected:raise RuntimeError('Validation token coverage mismatch')
        if not torch.isfinite(totals).all() or expected==0:raise RuntimeError('Invalid validation loss')
        return dict(loss=(totals[0]/totals[1]).item(),tokens=expected,blocks=len(dataset))
    finally:
        model.train(training)
