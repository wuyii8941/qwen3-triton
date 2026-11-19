"""Strict CUDA operation dispatch for inference."""
import torch
from ..runtime.dispatch import TritonRuntimeMode


class InferenceRuntimeMode(TritonRuntimeMode):
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kw=kwargs or {}
        if func._schema.name=='aten::contiguous' and args[0].is_cuda:
            source=args[0]
            if kw.get('memory_format',torch.contiguous_format)!=torch.contiguous_format:raise ValueError('Contiguous layout only')
            if source.is_contiguous():return source
            from ..runtime.ops import clone
            self.replaced['aten::contiguous']+=1
            return clone(source,preserve=False)
        if func._schema.name=='aten::to':
            source=args[0];overload=func._schema.overload_name
            if overload=='dtype_layout':
                options=dict(kw);copy=options.pop('copy',False)
            elif overload=='device':
                names=['device','dtype','non_blocking','copy','memory_format'];options=dict(zip(names,args[1:]));options.update(kw);copy=options.pop('copy',False)
            elif overload=='dtype':
                names=['dtype','non_blocking','copy','memory_format'];options=dict(zip(names,args[1:]));options.update(kw);copy=options.pop('copy',False)
            else:return super().__torch_dispatch__(func,types,args,kw)
            target=torch.device(options.get('device') or source.device)
            if not source.is_cuda and target.type!='cuda':return func(*args,**kw)
            if options.get('layout') not in (None,torch.strided) or options.get('pin_memory',False):raise ValueError('Strided unpinned CUDA conversion required')
            if not copy and target==source.device and options.get('dtype',source.dtype)==source.dtype and options.get('memory_format') in (None,torch.preserve_format):
                self.metadata['aten::to_identity']+=1;return source
            options={k:v for k,v in options.items() if v is not None and k not in ['layout','pin_memory']}
            self.replaced['aten::to']+=1
            return self._to_copy(torch.ops.aten._to_copy.default,source,options)
        return super().__torch_dispatch__(func,types,args,kw)
