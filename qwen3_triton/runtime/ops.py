"""Explicit Triton tensor utilities needed by the whole training data path."""
import math
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
from .memory import empty,raw_empty


@triton.jit
def _runtime_typed_fill(Y,N:tl.constexpr,VALUE:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    value=tl.full((),VALUE,Y.dtype.element_ty)
    tl.store(Y+i,value,i<N)


def fill_(target,value):
    if not target.is_contiguous():raise ValueError('Contiguous typed fill required')
    if target.numel():_runtime_typed_fill[(triton.cdiv(target.numel(),1024),)](target,target.numel(),value,1024)
    return target


def strides(shape):
    result=[];step=1
    for size in reversed(shape):result.append(step);step*=size
    return tuple(reversed(result))


def empty_strided(shape,stride,*,device,dtype,initialize=True):
    shape=tuple(shape);stride=tuple(stride)
    if len(shape)!=len(stride) or any(s<0 for s in stride):raise ValueError('Concrete nonnegative strides required')
    span=0 if math.prod(shape)==0 else 1+sum((n-1)*s for n,s in zip(shape,stride))
    alloc=empty if initialize else raw_empty
    base=alloc((span,),device=device,dtype=dtype)
    return base.as_strided(shape,stride)


@triton.jit
def _offset(I,SHAPE:tl.constexpr,STRIDE:tl.constexpr):
    remain=I;out=tl.full(I.shape,0,tl.int64)
    for axis in tl.static_range(len(SHAPE)-1,-1,-1):
        out+=(remain%SHAPE[axis])*STRIDE[axis];remain=remain//SHAPE[axis]
    return out


@triton.jit
def _runtime_strided_copy(X,Y,N:tl.constexpr,SHAPE:tl.constexpr,XS:tl.constexpr,YS:tl.constexpr,B:tl.constexpr):
    i=(tl.program_id(0)*B+tl.arange(0,B)).to(tl.int64)
    value=tl.load(X+_offset(i,SHAPE,XS),i<N,0)
    tl.store(Y+_offset(i,SHAPE,YS),value,i<N)


def copy_(target,source):
    if not target.is_cuda or not source.is_cuda or target.device!=source.device:raise ValueError('Same-device CUDA copy required')
    if source.shape!=target.shape:source=source.expand(target.shape)
    if target.numel():
        _runtime_strided_copy[(triton.cdiv(target.numel(),1024),)](source,target,target.numel(),tuple(target.shape),tuple(source.stride()),tuple(target.stride()),1024)
    return target


def clone(source,*,dtype=None,preserve=True):
    dtype=source.dtype if dtype is None else dtype
    from torch._prims_common import is_non_overlapping_and_dense
    stride=tuple(source.stride()) if preserve and is_non_overlapping_and_dense(source) else strides(source.shape)
    out=empty_strided(source.shape,stride,device=source.device,dtype=dtype)
    return copy_(out,source)


@triton.jit
def _runtime_binary(X,Y,OUT,N:tl.constexpr,SHAPE:tl.constexpr,XS:tl.constexpr,YS:tl.constexpr,OS:tl.constexpr,
                    TENSOR_Y:tl.constexpr,OP:tl.constexpr,ALPHA:tl.constexpr,B:tl.constexpr):
    i=(tl.program_id(0)*B+tl.arange(0,B)).to(tl.int64);valid=i<N
    dtype:tl.constexpr=OUT.dtype.element_ty
    calc:tl.constexpr=tl.float32 if (dtype==tl.float16 or dtype==tl.bfloat16) else dtype
    x=tl.load(X+_offset(i,SHAPE,XS),valid,0).to(calc)
    if TENSOR_Y:y=tl.load(Y+_offset(i,SHAPE,YS),valid,0).to(calc)
    else:y=tl.full((),Y,calc)
    if OP=='add':value=x+y*ALPHA
    elif OP=='sub':value=x-y*ALPHA
    elif OP=='mul':value=x*y
    elif OP=='div':
        if calc==tl.float64:value=libdevice.div_rn(x,y)
        else:value=tl.div_rn(x,y)
    elif OP=='eq':value=x==y
    elif OP=='ne':value=x!=y
    elif OP=='lt':value=x<y
    elif OP=='le':value=x<=y
    elif OP=='gt':value=x>y
    elif OP=='ge':value=x>=y
    elif OP=='and':value=x&y
    elif OP=='or':value=x|y
    else:tl.static_assert(False,'Unsupported binary operation')
    tl.store(OUT+_offset(i,SHAPE,OS),value,valid)


def binary(x,y,op,*,out=None,alpha=1):
    tensor_y=isinstance(y,torch.Tensor)
    if tensor_y and y.device.type=='cpu' and y.numel()==1:y=y.item();tensor_y=False
    if not isinstance(x,torch.Tensor) or not x.is_cuda or (tensor_y and y.device!=x.device):raise ValueError('CUDA left operand and matching right operand required')
    shape=torch.broadcast_shapes(x.shape,y.shape) if tensor_y else tuple(x.shape)
    dtype=torch.result_type(x,y)
    if op in ['eq','ne','lt','le','gt','ge']:dtype=torch.bool
    if op=='div' and not dtype.is_floating_point:dtype=torch.get_default_dtype()
    # Comparison inputs must retain their arithmetic precision rather than
    # being converted to bool before comparison.
    if op in ['eq','ne','lt','le','gt','ge']:
        return comparison(x,y,op,out=out)
    if out is None:out=empty(shape,device=x.device,dtype=dtype)
    if tuple(out.shape)!=tuple(shape):raise ValueError('Matching binary output shape required')
    xx=x.expand(shape);yy=y.expand(shape) if tensor_y else y
    if out.numel():_runtime_binary[(triton.cdiv(out.numel(),1024),)](xx,yy,out,out.numel(),tuple(shape),tuple(xx.stride()),tuple(yy.stride()) if tensor_y else (),tuple(out.stride()),tensor_y,op,alpha,1024,enable_fp_fusion=False)
    return out


@triton.jit
def _runtime_compare(X,Y,OUT,N:tl.constexpr,SHAPE:tl.constexpr,XS:tl.constexpr,YS:tl.constexpr,OS:tl.constexpr,TENSOR_Y:tl.constexpr,OP:tl.constexpr,B:tl.constexpr):
    i=(tl.program_id(0)*B+tl.arange(0,B)).to(tl.int64);valid=i<N
    x=tl.load(X+_offset(i,SHAPE,XS),valid,0)
    if TENSOR_Y:y=tl.load(Y+_offset(i,SHAPE,YS),valid,0)
    else:y=Y
    if OP=='eq':v=x==y
    elif OP=='ne':v=x!=y
    elif OP=='lt':v=x<y
    elif OP=='le':v=x<=y
    elif OP=='gt':v=x>y
    else:v=x>=y
    tl.store(OUT+_offset(i,SHAPE,OS),v,valid)


def comparison(x,y,op,*,out=None):
    tensor_y=isinstance(y,torch.Tensor)
    shape=torch.broadcast_shapes(x.shape,y.shape) if tensor_y else tuple(x.shape)
    if out is None:out=empty(shape,device=x.device,dtype=torch.bool)
    xx=x.expand(shape);yy=y.expand(shape) if tensor_y else y
    if out.numel():_runtime_compare[(triton.cdiv(out.numel(),1024),)](xx,yy,out,out.numel(),tuple(shape),tuple(xx.stride()),tuple(yy.stride()) if tensor_y else (),tuple(out.stride()),tensor_y,op,1024)
    return out


@triton.jit
def _runtime_unary(X,Y,N:tl.constexpr,SHAPE:tl.constexpr,XS:tl.constexpr,YS:tl.constexpr,OP:tl.constexpr,B:tl.constexpr):
    i=(tl.program_id(0)*B+tl.arange(0,B)).to(tl.int64);valid=i<N
    x=tl.load(X+_offset(i,SHAPE,XS),valid,0)
    if OP=='abs':value=tl.abs(x)
    elif OP=='neg':value=-x
    elif OP=='isfinite':value=(x==x)&(tl.abs(x)!=float('inf'))
    elif OP=='not':value=~x
    else:tl.static_assert(False,'Unsupported unary operation')
    tl.store(Y+_offset(i,SHAPE,YS),value,valid)


def unary(x,op,*,out=None):
    if out is None:out=empty(x.shape,device=x.device,dtype=torch.bool if op=='isfinite' else x.dtype)
    if out.numel():_runtime_unary[(triton.cdiv(out.numel(),1024),)](x,out,x.numel(),tuple(x.shape),tuple(x.stride()),tuple(out.stride()),op,1024)
    return out


@triton.jit
def _runtime_reduce(X,Y,N:tl.constexpr,KIND:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);valid=i<N
    if KIND=='all':value=tl.load(X+i,valid,1);reduced=tl.min((value!=0).to(tl.int32),0)
    elif KIND=='any':value=tl.load(X+i,valid,0);reduced=tl.max((value!=0).to(tl.int32),0)
    else:
        value=tl.load(X+i,valid,0).to(Y.dtype.element_ty);reduced=tl.sum(value,0)
    tl.store(Y+tl.program_id(0),reduced)


def reduce(x,kind,*,dtype=None):
    if not x.is_contiguous():x=clone(x,preserve=False)
    if kind in ['all','any']:dtype=torch.bool;acc=torch.bool
    else:
        dtype=dtype or (x.dtype if x.dtype.is_floating_point else torch.int64)
        acc=torch.float32 if dtype in [torch.bfloat16,torch.float16] else dtype
    if not x.numel():
        out=raw_empty((),device=x.device,dtype=dtype);return fill_(out,1 if kind=='all' else 0)
    current=x
    while True:
        parts=triton.cdiv(current.numel(),1024);out=raw_empty((parts,),device=x.device,dtype=acc)
        _runtime_reduce[(parts,)](current,out,current.numel(),kind,1024)
        if parts==1:break
        current=out
    if out.dtype!=dtype:out=clone(out,dtype=dtype,preserve=False)
    return out.view(())


@triton.jit
def _runtime_chunk_cat_one(X,Y,N:tl.constexpr,LOCAL:tl.constexpr,OUT_STRIDE:tl.constexpr,START:tl.constexpr,
                           SHAPE:tl.constexpr,XS:tl.constexpr,B:tl.constexpr):
    i=(tl.program_id(0)*B+tl.arange(0,B)).to(tl.int64);valid=i<N
    value=tl.load(X+_offset(i,SHAPE,XS),valid,0)
    tl.store(Y+(i//LOCAL)*OUT_STRIDE+START+i%LOCAL,value,valid)


def chunk_cat(tensors,dim,num_chunks,out):
    if dim!=0 or num_chunks!=4 or not out.is_contiguous():raise ValueError('Audited four-rank dim0 chunk-cat required')
    sizes=[x.numel()//4 for x in tensors]
    if any(x.ndim<1 or x.shape[0]%4 for x in tensors):raise ValueError('Uneven dim0 shard requires a separately implemented padding policy')
    width=sum(sizes)
    if out.numel()!=4*width:raise ValueError('Complete packed output size mismatch')
    start=0
    for x,n in zip(tensors,sizes):
        if x.numel():_runtime_chunk_cat_one[(triton.cdiv(x.numel(),1024),)](x,out,x.numel(),n,width,start,tuple(x.shape),tuple(x.stride()),1024)
        start+=n
