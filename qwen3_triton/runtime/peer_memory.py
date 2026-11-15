"""CUDA IPC allocation/mapping only; GPU initialization is explicitly Triton.

Torch2.10 CUDASymmetricMemoryAllocator::alloc calls cudaMemset on each whole
allocation. That GPU data operation cannot be exempted as host allocation.
This scoped driver adapter avoids that allocator, without intercepting CUDA
functions or changing Torch globally. Gloo exchanges only CPU IPC metadata.
Call collective close after all GPU users finish, before owners free memory.
"""
import ctypes,math,socket
import torch
import torch.distributed as dist


class _IPCHandle(ctypes.Structure):
    _fields_=[('reserved',ctypes.c_ubyte*64)]


class _Driver:
    def __init__(self):
        self.lib=ctypes.CDLL('libcuda.so.1')
        signatures={
            'cuMemAlloc_v2':[ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t],
            'cuMemFree_v2':[ctypes.c_uint64],
            'cuIpcGetMemHandle':[ctypes.POINTER(_IPCHandle),ctypes.c_uint64],
            'cuIpcOpenMemHandle_v2':[ctypes.POINTER(ctypes.c_uint64),_IPCHandle,ctypes.c_uint],
            'cuIpcCloseMemHandle':[ctypes.c_uint64],
            'cuCtxGetCurrent':[ctypes.POINTER(ctypes.c_void_p)],
            'cuGetErrorString':[ctypes.c_int,ctypes.POINTER(ctypes.c_char_p)]}
        for name,args in signatures.items():
            fn=getattr(self.lib,name);fn.argtypes=args;fn.restype=ctypes.c_int

    def call(self,name,*args):
        result=getattr(self.lib,name)(*args)
        if result:
            message=ctypes.c_char_p();self.lib.cuGetErrorString(result,ctypes.byref(message))
            raise RuntimeError(f'{name} failed({result}): {message.value.decode() if message.value else "unknown CUDA driver error"}')


class DevicePointer:
    """Triton pointer argument; no Tensor/ATen allocation or data conversion."""
    def __init__(self,pool,address,shape,dtype):
        self.pool=pool;self.address=address;self.shape=tuple(shape);self.dtype=dtype;self.device=pool.device;self.is_cuda=True

    def data_ptr(self):
        if self.pool.closed:raise RuntimeError('Closed CUDA IPC mapping')
        return self.address

    def numel(self):return math.prod(self.shape)

    def is_contiguous(self):return True


class PeerMemory:
    """One local external allocation and three imported peer mappings."""
    def __init__(self,size_bytes,group):
        if dist.get_backend(group)!='gloo' or dist.get_process_group_ranks(group)!=[0,1,2,3]:
            raise ValueError('Canonical four-rank CPU Gloo bootstrap required')
        if size_bytes<1:raise ValueError('Nonempty allocation required')
        self.group=group;self.rank=dist.get_rank(group);self.device=torch.device('cuda',torch.cuda.current_device())
        self.size_bytes=size_bytes;self.closed=False;self.imports=[];self.driver=_Driver()
        context=ctypes.c_void_p();self.driver.call('cuCtxGetCurrent',ctypes.byref(context))
        if not context.value:raise RuntimeError('Initialize this rank CUDA primary context before peer allocation')
        address=ctypes.c_uint64();self.driver.call('cuMemAlloc_v2',ctypes.byref(address),size_bytes);self.local_address=address.value
        handle=_IPCHandle();self.driver.call('cuIpcGetMemHandle',ctypes.byref(handle),address.value)
        metadata=dict(host=socket.gethostname(),size_bytes=size_bytes,handle=bytes(handle.reserved))
        exchanged=[None]*4;dist.all_gather_object(exchanged,metadata,group=group)
        if any(x['host']!=metadata['host'] or x['size_bytes']!=size_bytes for x in exchanged):raise ValueError('Same-host equal-size CUDA IPC allocations required')
        self.addresses=[]
        for rank,item in enumerate(exchanged):
            if rank==self.rank:self.addresses.append(self.local_address);continue
            remote=_IPCHandle();ctypes.memmove(ctypes.byref(remote),item['handle'],64)
            pointer=ctypes.c_uint64()
            # CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS. Only mapping/context work;
            # no CUDA memcpy/memset or native GPU computation is issued.
            self.driver.call('cuIpcOpenMemHandle_v2',ctypes.byref(pointer),remote,1)
            self.addresses.append(pointer.value);self.imports.append(pointer.value)

    def pointer(self,rank,shape,dtype,*,offset_bytes=0):
        nbytes=math.prod(shape)*dtype.itemsize
        if self.closed or not 0<=rank<4 or offset_bytes<0 or offset_bytes%dtype.itemsize or offset_bytes+nbytes>self.size_bytes:
            raise ValueError('Invalid or closed typed peer view')
        return DevicePointer(self,self.addresses[rank]+offset_bytes,shape,dtype)

    def close(self):
        if self.closed:return
        # Owners must remain alive until every imported handle has closed.
        # Destructors intentionally do not free an owner asynchronously.
        torch.cuda.synchronize();dist.barrier(group=self.group)
        for pointer in self.imports:self.driver.call('cuIpcCloseMemHandle',pointer)
        dist.barrier(group=self.group)
        self.driver.call('cuMemFree_v2',self.local_address);self.closed=True
