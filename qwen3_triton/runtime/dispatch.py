"""Strict public TorchDispatchMode for runtime GPU data operations.

This is not a global monkeypatch. Known GPU operations are replaced by Triton;
unknown GPU operations fail explicitly. Views/host bookkeeping and actual
host-device DMA remain runtime operations. A real kernel trace is still
required because operations below the dispatcher can launch hidden kernels.
"""
from collections import Counter
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from . import ops
from .memory import empty,raw_empty
from .ops import fill_


_METADATA={
    'aten::view','aten::_unsafe_view','aten::_reshape_alias','aten::as_strided',
    'aten::detach','aten::detach_','aten::alias','aten::t','aten::transpose','aten::permute',
    'aten::slice','aten::select','aten::split','aten::split_with_sizes','aten::unbind',
    'aten::squeeze','aten::squeeze_','aten::unsqueeze','aten::unsqueeze_',
    'aten::expand','aten::set_','aten::is_same_size','aten::_has_compatible_shallow_copy_type',
    'aten::lift_fresh','aten::record_stream',
}


class TritonRuntimeMode(TorchDispatchMode):
    def __init__(self):
        super().__init__();self.replaced=Counter();self.metadata=Counter();self.host_dma=Counter()

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kw=kwargs or {};name=func._schema.name
        flat,_=tree_flatten((args,kw));tensors=[x for x in flat if isinstance(x,torch.Tensor)]
        device=kw.get('device');requested=torch.device(device) if device is not None else None
        cuda=any(t.is_cuda for t in tensors) or (requested is not None and requested.type=='cuda')
        if not cuda:return func(*args,**kw)
        if name in _METADATA:
            self.metadata[name]+=1;return func(*args,**kw)
        if name in ['aten::_local_scalar_dense','aten::is_nonzero']:
            self.host_dma['scalar_DtoH']+=1;return func(*args,**kw)
        # Local FSDP tensors are ordinary tensors. An unexpected DTensor
        # arithmetic route must not disable this mode around native kernels.
        if any(hasattr(t,'to_local') for t in tensors):
            raise RuntimeError(f'Unimplemented DTensor GPU runtime operation: {func}')
        self.replaced[name]+=1
        if name in ['aten::new_empty','aten::new_empty_strided','aten::new_zeros','aten::new_ones','aten::new_full']:
            source=args[0];target_device=requested or source.device
            if target_device.type!='cuda':return func(*args,**kw)
            shape=tuple(args[1] if len(args)>1 else kw['size']);dtype=kw.get('dtype') or source.dtype
            if name=='aten::new_empty_strided':
                stride=tuple(args[2] if len(args)>2 else kw['stride'])
                return ops.empty_strided(shape,stride,device=target_device,dtype=dtype)
            if name=='aten::new_empty':return empty(shape,device=target_device,dtype=dtype)
            value=(args[2] if len(args)>2 else kw['fill_value']) if name=='aten::new_full' else (1 if name=='aten::new_ones' else 0)
            return fill_(raw_empty(shape,device=target_device,dtype=dtype),value)
        if name in ['aten::empty','aten::empty_strided']:
            shape=tuple(args[0]);dtype=kw.get('dtype') or torch.get_default_dtype();device=requested or torch.get_default_device()
            stride=tuple(args[1]) if name=='aten::empty_strided' else ops.strides(shape)
            return ops.empty_strided(shape,stride,device=device,dtype=dtype)
        if name in ['aten::empty_like','aten::zeros_like','aten::ones_like']:
            source=args[0];dtype=kw.get('dtype') or source.dtype;device=requested or source.device
            if kw.get('memory_format')==torch.contiguous_format:stride=ops.strides(source.shape)
            else:
                from torch._prims_common import is_non_overlapping_and_dense
                stride=tuple(source.stride()) if is_non_overlapping_and_dense(source) else ops.strides(source.shape)
            out=ops.empty_strided(source.shape,stride,device=device,dtype=dtype,initialize=name=='aten::empty_like')
            if name!='aten::empty_like':
                # Fill the backing span, so non-contiguous dense layouts work.
                value=1 if name=='aten::ones_like' else 0
                return self._fill_strided(out,value)
            return out
        if name in ['aten::zeros','aten::ones','aten::full','aten::scalar_tensor']:
            if name=='aten::scalar_tensor':shape=();value=args[0]
            else:shape=tuple(args[0]);value=args[1] if name=='aten::full' else (1 if name=='aten::ones' else 0)
            dtype=kw.get('dtype') or (torch.get_default_dtype() if isinstance(value,float) or name in ['aten::zeros','aten::ones'] else torch.int64)
            return fill_(raw_empty(shape,device=requested or 'cuda',dtype=dtype),value)
        if name in ['aten::clone','aten::lift_fresh_copy']:return ops.clone(args[0],preserve=kw.get('memory_format')!=torch.contiguous_format)
        if name=='aten::reshape':
            try:return args[0].view(args[1])
            except RuntimeError:return ops.clone(args[0],preserve=False).view(args[1])
        if name=='aten::_to_copy':return self._to_copy(func,args[0],kw)
        if name=='aten::copy_':return self._copy(args[0],args[1],kw.get('non_blocking',False))
        if name=='aten::_foreach_copy_':
            for target,source in zip(args[0],args[1]):self._copy(target,source,kw.get('non_blocking',False))
            return None
        if name in ['aten::fill_','aten::zero_']:
            value=0 if name=='aten::zero_' else args[1]
            if isinstance(value,torch.Tensor):
                if value.is_cuda:raise RuntimeError('GPU-tensor fill scalar requires an explicit broadcast-copy implementation')
                value=value.item()
            return self._fill_strided(args[0],value)
        if name in ['aten::add','aten::add_','aten::sub','aten::sub_','aten::mul','aten::mul_','aten::div','aten::div_',
                    'aten::eq','aten::ne','aten::lt','aten::le','aten::gt','aten::ge','aten::bitwise_and','aten::bitwise_or',
                    'aten::__and__','aten::__or__']:
            op=name.split('::')[1].rstrip('_');op={'bitwise_and':'and','bitwise_or':'or','__and':'and','__or':'or'}.get(op,op)
            if kw.get('rounding_mode') is not None:raise RuntimeError('Explicit rounded division is not implemented')
            out=args[0] if name.endswith('_') else kw.get('out')
            return ops.binary(args[0],args[1],op,out=out,alpha=kw.get('alpha',1))
        if name in ['aten::abs','aten::neg','aten::isfinite','aten::bitwise_not','aten::__invert__']:
            return ops.unary(args[0],{'aten::bitwise_not':'not','aten::__invert__':'not'}.get(name,name.split('::')[1]),out=kw.get('out'))
        if name in ['aten::all','aten::any','aten::sum']:
            if len(args)>1 or kw.get('dim') is not None or kw.get('keepdim',False):
                raise RuntimeError(f'Unimplemented dimension-specific GPU reduction: {func}')
            return ops.reduce(args[0],name.split('::')[1],dtype=kw.get('dtype'))
        if name in ['fsdp::all_gather_copy_in']:
            inputs,output,sizes,n,rank=args;target=output.narrow(0,n*rank,n);start=0
            for source,size in zip(inputs,sizes):
                if not source.is_contiguous():source=ops.clone(source,preserve=False)
                self._copy(target.narrow(0,start,size),source.view(-1));start+=size
            if start!=n:raise ValueError('Incomplete FSDP gather copy-in')
            return target,output
        if name in ['fsdp::split_with_sizes_copy','aten::split_with_sizes_copy']:
            source,sizes=args[:2];dim=args[2] if len(args)>2 else kw.get('dim',0);outputs=kw['out'];start=0
            for size,target in zip(sizes,outputs):ops.copy_(target,source.narrow(dim,start,size));start+=size
            if start!=source.shape[dim]:raise ValueError('Incomplete FSDP gather split')
            return None if name.startswith('fsdp::') else outputs
        if name in ['fsdp::chunk_cat','aten::_chunk_cat']:
            tensors=args[0];dim=args[1] if len(args)>1 else kw['dim'];chunks=args[2] if len(args)>2 else kw['num_chunks']
            ops.chunk_cat(tensors,dim,chunks,kw['out']);return None if name.startswith('fsdp::') else kw['out']
        raise RuntimeError(f'Unimplemented GPU runtime operation: {func}; shapes/dtypes={[(tuple(t.shape),str(t.dtype)) for t in tensors]}')

    def _fill_strided(self,target,value):
        if target.is_contiguous():return fill_(target,value)
        scalar=fill_(raw_empty((),device=target.device,dtype=target.dtype),value)
        return ops.copy_(target,scalar.expand(target.shape))

    def _copy(self,target,source,non_blocking=False):
        if target.is_cuda and source.is_cuda:return ops.copy_(target,source)
        self.host_dma['host_device_copy']+=1
        if target.is_cuda:
            # Conversion/layout on host before DMA; no CUDA arithmetic is
            # delegated to ATen's generic copy kernel.
            host=source.to(dtype=target.dtype).expand(target.shape).contiguous()
            if target.is_contiguous():target.copy_(host,non_blocking=non_blocking);return target
            temporary=raw_empty(target.shape,device=target.device,dtype=target.dtype);temporary.copy_(host,non_blocking=non_blocking)
            return ops.copy_(target,temporary)
        device=source if source.is_contiguous() and source.dtype==target.dtype else ops.clone(source,dtype=target.dtype,preserve=False)
        target.copy_(device,non_blocking=non_blocking);return target

    def _to_copy(self,func,source,kw):
        target_device=torch.device(kw.get('device') or source.device);dtype=kw.get('dtype') or source.dtype
        if source.is_cuda and target_device==source.device:
            return ops.clone(source,dtype=dtype,preserve=kw.get('memory_format')!=torch.contiguous_format)
        if source.is_cuda and target_device.type=='cuda':raise RuntimeError('Cross-device Tensor.to is not a mapped peer collective')
        self.host_dma['to_host_or_device']+=1
        if source.is_cuda:
            temporary=source if source.is_contiguous() and source.dtype==dtype else ops.clone(source,dtype=dtype,preserve=False)
            return func(temporary,**kw)
        host=source.to(dtype=dtype).contiguous()
        out=raw_empty(host.shape,device=target_device,dtype=dtype);out.copy_(host,non_blocking=kw.get('non_blocking',False));return out

    def report(self):
        return dict(strict=True,replaced_GPU_operations=dict(self.replaced),host_metadata_operations=dict(self.metadata),host_DMA_operations=dict(self.host_dma),
            scope='Host views/autograd/FSDP scheduling/driver/DMA remain. Unknown GPU data operations fail. Full real-kernel trace must additionally prove no hidden native compute.')
