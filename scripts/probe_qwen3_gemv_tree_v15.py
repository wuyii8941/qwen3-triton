"""Infer candidate native GEMV reduction trees from exact sparse-dot masks.

This is a reference-only experiment: no production dispatch/kernel changes.
A unit marker survives iff it joins after the large opposite pair cancels.
For a binary addition tree this is the complement of their LCA leaf set.
"""
import argparse,gzip,json,os,struct,subprocess,threading,time
from pathlib import Path
import torch
from qwen3_triton.inference import kernels
from qwen3_triton.benchmark_utils.stable_idle import stable_lease
from scripts.verify_qwen3_inference_v15 import sha
ROOT=Path(__file__).resolve().parents[1]

def build_tree(k,lanes,load,acc,mapping,vector_order,warp_order):
 parents=[-1]*k;masks=[1<<i for i in range(k)];children=[None]*k
 def add(a,b):
  assert parents[a]==parents[b]==-1 and not (masks[a]&masks[b])
  node=len(parents);parents[a]=parents[b]=node;parents.append(-1);masks.append(masks[a]|masks[b]);children.append((a,b));return node
 def fold(nodes,order):
  nodes=list(nodes)
  if order in ['forward','reverse']:
   if order=='reverse':nodes.reverse()
   total=nodes[0]
   for node in nodes[1:]:total=add(total,node)
   return total
  while len(nodes)>1:
   if order=='adjacent':nodes=[add(nodes[i],nodes[i+1]) for i in range(0,len(nodes),2)]
   else:
    half=len(nodes)//2;nodes=[add(nodes[i],nodes[i+half]) for i in range(half)]
  return nodes[0]
 lane_nodes=[]
 for lane in range(lanes):
  local=[]
  for register in range(acc):
   indices=[]
   for iteration in range(k//(lanes*load)):
    for v in range(register,load,acc):
     index=iteration*lanes*load+(lane*load+v if mapping=='packed' else v*lanes+lane)
     indices.append(index)
   local.append(fold(indices,'forward'))
  lane_nodes.append(fold(local,vector_order))
 root=fold(lane_nodes,warp_order);assert masks[root]==(1<<k)-1 and len(parents)==2*k-1
 return parents,masks,children

def lca_mask(parents,masks,a,b):
 ancestors=set()
 while a!=-1:ancestors.add(a);a=parents[a]
 while b not in ancestors:b=parents[b]
 return masks[b]

def cpu_check():
 parents,masks,children=build_tree(8,2,2,2,'packed','forward','halves')
 assert lca_mask(parents,masks,0,4)==(1<<0)|(1<<4)
 assert lca_mask(parents,masks,0,1)==sum(1<<v for v in [0,1,4,5])
 assert lca_mask(parents,masks,0,2)==255
 for positive,negative in [(0,4),(0,1),(0,2)]:
  lost=lca_mask(parents,masks,positive,negative)
  for marker in range(8):
   values=[0.]*len(parents);values[marker]=1.;values[positive]=2.**30;values[negative]=-(2.**30)
   for node in range(8,len(parents)):
    a,b=children[node];values[node]=struct.unpack('f',struct.pack('f',values[a]+values[b]))[0]
   assert values[-1]==(0. if lost&(1<<marker) else 1.)
 print('CPU LCA mask matches explicit FP32 sparse-dot evaluation',flush=True)

def fit(k,lanes,gold):
 exact=[];best=[];tested=0;start=time.perf_counter()
 for load in [1,2,4,8,16,32,64,128]:
  if lanes*load>k:continue
  for acc in [1,2,4,8,16,32,64,128]:
   if acc>load:continue
   for mapping in (['packed'] if load==1 else ['packed','strided']):
    for vector_order in (['forward'] if acc==1 else ['forward','reverse','adjacent','halves']):
     for warp_order in ['adjacent','halves']:
      parents,masks,_=build_tree(k,lanes,load,acc,mapping,vector_order,warp_order)
      error=sum((lca_mask(parents,masks,row['positive'],row['negative'])^int(row['lost_K_bitset_integer_hex'],16)).bit_count() for row in gold)
      item=dict(lanes=lanes,load=load,accumulators=acc,mapping=mapping,vector_order=vector_order,warp_order=warp_order,mask_mismatches=error)
      tested+=1
      if error==0:exact.append(item)
      best.append(item);best.sort(key=lambda x:x['mask_mismatches']);del best[12:]
 return dict(tested=tested,exact_count=len(exact),exact=exact,best=best,seconds=time.perf_counter()-start)

@torch.inference_mode()
def run(out,foreign):
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 files=[*sorted((ROOT/'qwen3_triton').rglob('*.py')),Path(__file__)]
 hashes={str(p.relative_to(ROOT)):sha(p) for p in files};previous_path=ROOT/'results/qwen3_triton_inference_v15_native_splitk_v1.json';previous=json.loads(previous_path.read_text())
 pairs=[(0,1),(0,2),(0,4),(0,8),(0,15),(0,16),(0,32),(0,64),(0,128),(0,256),(0,512),(16,48),(0,2048),(4064,4095),(7,1025),(31,2049),(37,2893),(1600,3504)]
 records=[];k=4096
 for kind,n,lanes in [('KV',1024,32),('gate_up',12288,16)]:
  x=torch.ones((1,1,k),device='cuda',dtype=torch.bfloat16);w=torch.zeros((n,k),device='cuda',dtype=torch.bfloat16);indices=torch.arange(n,device='cuda');gold=[];sample_rows=[]
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
   for _ in range(64):kernels.positions(1,1,0,torch.device('cuda'))
   torch.cuda.synchronize()
   with torch.profiler.record_function('native_gemv'):
    for _ in range(4):torch.nn.functional.linear(x,w)
    torch.cuda.synchronize()
  trace=out/(kind+'.json.gz');prof.export_chrome_trace(str(trace));assert trace.stat().st_size<=8*2**20
  events=json.loads(gzip.decompress(trace.read_bytes()))['traceEvents'];scope=next(e for e in events if e.get('name')=='native_gemv' and e.get('cat')=='user_annotation');start=scope['ts'];end=start+scope['dur']
  launches=[e for e in events if 'LaunchKernel' in e.get('name','') and start<=e.get('ts',-1)<=end];ids={e.get('args',{}).get('correlation') for e in launches}
  gpu=[e for e in events if e.get('cat')=='kernel' and e.get('args',{}).get('correlation') in ids]
  assert ids=={e.get('args',{}).get('correlation') for e in gpu}
  native=[]
  for e in gpu:
   item=dict(name=e['name'],grid=e.get('args',{}).get('grid'),block=e.get('args',{}).get('block'))
   if item not in native:native.append(item)
  expected=next(v for v in previous['native'] if v['m']==1 and v['kind']==kind)
  assert {v['name'] for v in native}=={v['name'] for v in expected['metadata']}
  offsets=range(4) if n<k else range(1)
  for positive,negative in pairs:
   known={};conflicts=set()
   for offset in offsets:
    marker=(indices*4+offset)%k if n<k else indices%k
    w.zero_();w[indices,marker]=1;w[:,positive]=2.**30;w[:,negative]=-(2.**30)
    y=torch.nn.functional.linear(x,w)[0,0];assert not bool(((y!=0)&(y!=1)).any())
    positions=marker.cpu().tolist();flags=(y==1).cpu().tolist();packed=bytearray((n+7)//8)
    for row,(position,flag) in enumerate(zip(positions,flags)):
     if flag:packed[row//8]|=1<<(row%8)
     if position in known and known[position]!=flag:conflicts.add(position)
     known.setdefault(position,flag)
    sample_rows.append(dict(positive=positive,negative=negative,offset=offset,ones=sum(flags),native_one_bits_lsb_first_hex=packed.hex()))
   assert len(known)==k
   lost=sum(1<<position for position,flag in known.items() if not flag)
   row=dict(positive=positive,negative=negative,lost_K_bitset_integer_hex=hex(lost)[2:],lost_markers=lost.bit_count(),inconsistent_repeated_K_markers=sorted(conflicts))
   gold.append(row);print(json.dumps(dict(kind=kind,positive=positive,negative=negative,lost_markers=row['lost_markers'],conflicts=len(conflicts))),flush=True)
  fits=fit(k,lanes,gold) if not any(v['inconsistent_repeated_K_markers'] for v in gold) else None
  record=dict(kind=kind,n=n,k=k,lanes_from_native_blockX=lanes,native=native,sparse_dot_masks=gold,raw_samples=sample_rows,candidate_tree_fit=fits,raw_trace=dict(file=trace.name,sha256=sha(trace),bytes=trace.stat().st_size))
  records.append(record);print(json.dumps(dict(kind=kind,fit=fits)),flush=True)
 for p in files:assert sha(p)==hashes[str(p.relative_to(ROOT))],p
 result=dict(complete=True,diagnostic_only=True,production_changed=False,physical_GPU=1,GPU=torch.cuda.get_device_name(),torch=torch.__version__,CUDA=torch.version.cuda,records=records,source_sha256=hashes,expected_native_report_sha256=sha(previous_path),foreign_context_observations=list(foreign),resource_clean=not foreign,limitations='Sparse cancellation data fits regular per-thread accumulator/vector/warp trees only. Zero mask mismatches do not prove full arithmetic identity. Any candidates require real HF operand checks and unchanged complete-model gates.')
 raw=json.dumps(result,indent=2)+'\n';assert len(raw.encode())<=2*2**20;(out/'result.json').write_text(raw)
 print(json.dumps(dict(complete=True,exact_tree_counts={v['kind']:None if v['candidate_tree_fit'] is None else v['candidate_tree_fit']['exact_count'] for v in records},resource_clean=not foreign)),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--cpu-check-only',action='store_true');a=p.parse_args();cpu_check()
 if a.cpu_check_only:return
 assert a.output is not None;out=a.output.resolve();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and out.is_relative_to(Path('/data1/tzh')) and not out.exists();out.mkdir(parents=True)
 foreign=[];stop=threading.Event()
 with stable_lease([1]):
  uuid=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
  def monitor():
   while not stop.wait(.5):
    try:
     raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=10)
     foreign.extend(line for line in raw.splitlines() if line.split(',')[0].strip()==uuid and int(line.split(',')[1])!=os.getpid())
    except Exception as exc:foreign.append(repr(exc))
  thread=threading.Thread(target=monitor,daemon=True);thread.start()
  try:run(out,foreign)
  except BaseException as exc:(out/'failure.json').write_text(json.dumps(dict(error=repr(exc),foreign_context_observations=foreign),indent=2)+'\n');raise
  finally:stop.set();thread.join()
if __name__=='__main__':main()
