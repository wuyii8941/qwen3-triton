"""Explicit research-only runner; does not change StaticDecodeRunner defaults."""
from model.static_decode import StaticDecodeCache,StaticDecodeRunner
from ops.triton.block_skip_decode import block_skip_attention


class BlockSkipCache(StaticDecodeCache):
    def __init__(self,*args,threshold=1e-4,**kwargs):
        super().__init__(*args,**kwargs)
        if self.dim!=128 or not 0<=threshold<=1:
            raise ValueError('D=128 and threshold in [0,1] required')
        self.threshold=threshold

    def attend_decode(self,q,layer_idx,scale):
        if q.shape[2]!=1:raise ValueError('Decode query length must be 1')
        out,_=block_skip_attention(q[:,:,0],self.key_cache[layer_idx],self.value_cache[layer_idx],
            self.used,threshold=self.threshold,splits=16,block_n=128,scale=scale)
        return out.unsqueeze(1)


class BlockSkipRunner(StaticDecodeRunner):
    def __init__(self,model,batch_size,capacity,*,threshold=1e-4):
        super().__init__(model,batch_size,capacity)
        p=next(model.parameters())
        self.cache=BlockSkipCache(model.config,batch_size,capacity,device=p.device,dtype=p.dtype,threshold=threshold)
