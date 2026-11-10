"""FASA-inspired Qwen3 feasibility experiment, NOT a LongBench reproduction.

Separate calibration/held-out WikiText splits; all candidate timings include
selection. No benchmark result automatically changes production defaults.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import time
import torch
import transformers
import triton
from transformers import AutoModelForCausalLM,AutoTokenizer
from torch.nn.attention import sdpa_kernel,SDPBackend
from model.hf_inference import qwen3_triton_inference,pack_qkv_weights
from model.static_decode import StaticDecodeRunner,StaticDecodeCache
from model.frequency_sparse import FrequencySparseRunner
from ops.reference.frequency_sparse import contextual_agreement,dense_scores,proxy_scores as ref_scores,selection_metrics
from ops.triton.frequency_sparse import pack_proxy,proxy_scores,indexed_attention
from benchmarks.benchmark_hf_attention_e2e import bootstrap_speedup


ROOT=Path(__file__).resolve().parents[1]
MODEL=Path('/data1/tzh/models/Qwen/Qwen3-1.7B')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


@contextmanager
def observe(cache,callback):
    original=cache.attend_decode
    def wrapped(q,layer,scale):
        out=original(q,layer,scale)
        callback(q,layer,scale,out)
        return out
    cache.attend_decode=wrapped
    try:yield
    finally:del cache.attend_decode


def graph_us(fn,repeats=8,inner=20):
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=fn()
    samples=[]
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        start.record()
        for _ in range(inner):graph.replay()
        end.record(); end.synchronize()
        samples.append(start.elapsed_time(end)*1000/inner)
    return {'median_us':statistics.median(samples),'samples_us':samples}


def quality(actual,expected,labels):
    # CPU logits kept outside timed runs; same teacher-forced tokens for all.
    a,b=actual.float(),expected.float()
    loga,logb=a.log_softmax(-1),b.log_softmax(-1)
    kl=(logb.exp()*(logb-loga)).sum(-1)
    nlla=-loga.gather(-1,labels[:,None]).squeeze(-1)
    nllb=-logb.gather(-1,labels[:,None]).squeeze(-1)
    rel=(a-b).norm(dim=-1)/b.norm(dim=-1).clamp_min(1e-12)
    return dict(logits_relative_l2_mean=float(rel.mean()),logits_relative_l2_max=float(rel.max()),
        kl_mean=float(kl.mean()),kl_max=float(kl.max()),
        top1_agreement=float((a.argmax(-1)==b.argmax(-1)).float().mean()),
        dense_nll=float(nllb.mean()),candidate_nll=float(nlla.mean()),
        nll_delta=float((nlla-nllb).mean()),
        continuation_ppl_ratio=float(((nlla-nllb).mean()).exp()),
        finite=bool(torch.isfinite(a).all()))


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',default='v1')
    parser.add_argument('--data',type=Path,default=ROOT/'artifacts/frequency_wikitext_data.json')
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--calibration-docs',type=int,default=4)
    parser.add_argument('--calibration-seq',type=int,default=8192)
    parser.add_argument('--calibration-steps',type=int,default=8)
    parser.add_argument('--seqs',default='8192,16384,32768')
    parser.add_argument('--steps',type=int,default=128)
    parser.add_argument('--repeats',type=int,default=8)
    parser.add_argument('--test-offset',type=int,default=0)
    parser.add_argument('--skip-timing',action='store_true')
    args=parser.parse_args()
    if args.repeats<8 or args.steps<8:raise ValueError('At least 8 repeats/steps required')
    torch.set_num_threads(4)
    target=ROOT/'artifacts'/f'frequency_sparse_{args.tag}.json'
    calfile=ROOT/'artifacts'/f'frequency_calibration_{args.tag}.json'
    if target.exists() or (not args.calibration and calfile.exists()):
        raise FileExistsError('Choose a fresh tag; previous evidence is immutable')
    target.touch(exist_ok=False)
    corpus=json.loads(args.data.read_text())
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    # Tokenizer only; long sequences are then sliced below the actual limit.
    texts={s:torch.tensor(tokenizer(v['text'],add_special_tokens=False,verbose=False)['input_ids'])
           for s,v in corpus['splits'].items()}
    torch.manual_seed(20261101)
    rng=random.Random(20261101)
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,
        torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
    seqs=[int(s) for s in args.seqs.split(',')]
    if max(seqs)+args.steps+1>model.config.max_position_embeddings:
        raise ValueError('Context exceeds original checkpoint; no RoPE scaling change permitted')
    config_hash=sha(MODEL/'config.json')
    report=dict(manifest=dict(date='2026-11-01',model=str(MODEL),dtype='float16',
        torch=torch.__version__,transformers=transformers.__version__,triton=triton.__version__,
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),seed=20261101,
        compute_capability=torch.cuda.get_device_capability(),config_sha256=config_hash,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__).resolve(),
            ROOT/'ops/triton/frequency_sparse.py',ROOT/'ops/reference/frequency_sparse.py',
            ROOT/'model/frequency_sparse.py',ROOT/'model/static_decode.py',ROOT/'model/hf_inference.py']},
        data={s:{k:v for k,v in d.items() if k!='text'} for s,d in corpus['splits'].items()},
        dataset_revision=corpus['revision'],test_offset=args.test_offset,
        baseline='existing exact-weight Triton pipeline + native Flash static cache + CUDA Graph',
        constraints='B=1, equal lengths, default RoPE, inference only, all original FP16 weights',
        quality_scope='held-out WikiText continuation screening; NOT LongBench/task-quality acceptance',
        timing='randomized paired wall-time rounds; prefill+proxy packing+greedy decode; compile/capture excluded',
        driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip()),
        promotion_gate={'latency_speedup_min':1.05,'speedup_ci_lower_min':1.,'max_logit_l2':.02,
            'top1_agreement_min':.99,'kl_mean_max':.005,'ppl_ratio_max':1.01,
            'external_task_quality_required':True},cases=[],decision='pending; default dense unchanged')
    def save():target.write_text(json.dumps(report,indent=2))
    save()
    packed=pack_qkv_weights(model)
    # SAME optimizations for all variants; no weight quantization confounder.
    context=qwen3_triton_inference(model,fuse_qk=True,compact_gqa=True,decode_mlp=True,
        packed_qkv=True,packed_weights=packed,prefill_mlp=True,static_decode=True,
        direct_k_cache=True,direct_v_cache=True,fuse_residual=True)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION),context:
        if args.calibration:
            cal=json.loads(args.calibration.read_text())
            if cal['config_sha256']!=config_hash or cal['data_sha256']!=report['manifest']['data']['validation']['sha256']:
                raise ValueError('Calibration model/data mismatch')
            scores=torch.tensor(cal['agreement'],device='cuda')
        else:
            seq,steps=args.calibration_seq,args.calibration_steps
            runner=StaticDecodeRunner(model,1,seq+steps+1)
            scores=torch.zeros((model.config.num_hidden_layers,model.config.num_attention_heads,64),device='cuda')
            counts=[0]*model.config.num_hidden_layers
            for doc in range(args.calibration_docs):
                offset=doc*(seq+steps+1)
                ids=texts['validation'][offset:offset+seq+steps+1].unsqueeze(0).cuda()
                if ids.shape[1]!=seq+steps+1:raise ValueError('Not enough calibration tokens')
                runner.prefill(ids[:,:seq])
                def collect(q,layer,scale,out):
                    scores[layer].add_(contextual_agreement(q[:,:,0],runner.cache.key_cache[layer][:,:runner.cache.end],256))
                    counts[layer]+=1
                with observe(runner.cache,collect):
                    for pos in range(seq,seq+steps):runner.decode(ids[:,pos:pos+1],replay=False)
                print('calibration',doc+1,'/',args.calibration_docs,flush=True)
            scores/=torch.tensor(counts,device='cuda')[:,None,None]
            cal=dict(config_sha256=config_hash,data_sha256=report['manifest']['data']['validation']['sha256'],
                method='Eq4 top-256 overlap of split-half FC vs full-head scores, averaged over validation queries',
                docs=args.calibration_docs,seq=seq,steps=steps,query_counts=counts,agreement=scores.cpu().tolist())
            with calfile.open('x') as f:json.dump(cal,f)
            del runner
        report['calibration']=str(args.calibration or calfile)
        ranking=scores.argsort(-1,descending=True).int().contiguous()
        del scores
        save()
        for seq in seqs:
            print('case',seq,flush=True)
            steps=args.steps
            ids=texts['test'][args.test_offset:args.test_offset+seq+steps+1].unsqueeze(0).cuda()
            if ids.shape[1]!=seq+steps+1:raise ValueError('Not enough held-out tokens')
            prompt=ids[:,:seq].contiguous()
            capacity=seq+steps+1
            dense=StaticDecodeRunner(model,1,capacity)
            case=dict(prompt_tokens=seq,decode_steps=steps,batch=1,diagnostics=[],quality={},micro={},timing={})
            report['cases'].append(case)
            snapshots={}
            dense.prefill(prompt)
            reference=[]
            def snapshot(q,layer,scale,out):
                if layer in (0,7,14,21,27):snapshots[layer]=(q.clone(),out.clone(),scale)
            with observe(dense.cache,snapshot):
                for pos in range(seq,seq+steps):
                    out=dense.decode(ids[:,pos:pos+1],replay=False)
                    reference.append(out[0,0].cpu())
            reference=torch.stack(reference)
            valid=seq+steps
            # Last query has the same full dense KV prefix used for diagnostics.
            for layer,(q4,expected,scale) in snapshots.items():
                q=q4[:,:,0]
                k=dense.cache.key_cache[layer][:,:valid]
                v=dense.cache.value_cache[layer][:,:valid]
                full=dense_scores(q,k)
                for budget in (256,512,1024,2048):
                    for method in ('oracle','stream','random_fc','fasa16','fasa32'):
                        if method=='oracle':indices=full.topk(budget,-1).indices
                        elif method=='stream':
                            # Simple sink+recent baseline; not a claim to replicate StreamingLLM.
                            indices=torch.cat((torch.arange(4,device='cuda'),torch.arange(valid-budget+4,valid,device='cuda')))
                            indices=indices.expand(1,q.shape[1],-1).contiguous()
                        else:
                            if method=='random_fc':
                                gen=torch.Generator(device='cuda').manual_seed(92+layer)
                                pairs=torch.stack([torch.randperm(64,generator=gen,device='cuda')[:16] for _ in range(q.shape[1])])
                            else:pairs=ranking[layer,:,:int(method[4:])].contiguous()
                            indices=ref_scores(q,k,pairs).topk(budget,-1).indices
                        metrics=selection_metrics(q,k,v,indices,expected[:,0],scale)
                        case['diagnostics'].append(dict(layer=layer,budget=budget,method=method,**metrics))
                print('diagnostic',seq,layer,flush=True)
            save()
            runners={'dense_flash':dense}
            for f,budget in ((16,512),(32,1024)):
                name=f'fasa_f{f}_k{budget}'
                sparse=FrequencySparseRunner(model,1,capacity,pairs=ranking[:,:,:f].contiguous(),budget=budget)
                runners[name]=sparse
                sparse.prefill(prompt)
                actual=[]
                for pos in range(seq,seq+steps):
                    actual.append(sparse.decode(ids[:,pos:pos+1],replay=False)[0,0].cpu())
                metrics=quality(torch.stack(actual),reference,ids[0,seq+1:seq+steps+1].cpu())
                metrics['local_quality_gate']=(metrics['finite'] and metrics['logits_relative_l2_max']<.02
                    and metrics['top1_agreement']>=.99 and metrics['kl_mean']<=.005 and metrics['continuation_ppl_ratio']<=1.01)
                case['quality'][name]=metrics
                case.setdefault('memory',{})[name]=dict(full_kv_bytes=sparse.cache.nbytes,
                    extra_proxy_bytes=sparse.cache.proxy_nbytes)
                print('quality',seq,name,json.dumps(metrics),flush=True)
                save()
            # At identical dense-generated Q/K/V, isolate selection and FAC costs.
            for layer in (0,14,27):
                q4,expected,scale=snapshots[layer]
                q=q4[:,:,0]
                native=lambda:StaticDecodeCache.attend_decode(dense.cache,q4,layer,scale)
                case['micro'][str(layer)]={'native_flash':graph_us(native)}
                for name,sparse in list(runners.items())[1:]:
                    cache=sparse.cache
                    # For an apples-to-apples microbench, use the DENSE cache.
                    pairs=cache.pairs[layer]
                    proxy=cache.proxy[layer]
                    pack_proxy(dense.cache.key_cache[layer],pairs,torch.arange(valid,device='cuda'),proxy)
                    select=lambda:proxy_scores(q,proxy,pairs,dense.cache.used)
                    score=select()
                    idx=score.topk(cache.budget,-1,sorted=False).indices
                    topk=lambda:score.topk(cache.budget,-1,sorted=False).indices
                    fac=lambda:indexed_attention(q,dense.cache.key_cache[layer],dense.cache.value_cache[layer],idx,dense.cache.used,scale=scale)
                    def full():
                        pack_proxy(dense.cache.key_cache[layer],pairs,dense.cache.positions,proxy)
                        scores=select()
                        selected=scores.topk(cache.budget,-1,sorted=False).indices
                        return indexed_attention(q,dense.cache.key_cache[layer],dense.cache.value_cache[layer],selected,dense.cache.used,scale=scale)
                    case['micro'][str(layer)][name]={s:graph_us(fn) for s,fn in
                        [('score',select),('topk',topk),('indexed_fac',fac),('all_including_update',full)]}
                print('micro',seq,layer,json.dumps(case['micro'][str(layer)]),flush=True)
            save()
            if not args.skip_timing:
                for name,runner in runners.items():
                    runner.prefill(prompt)
                    runner.capture(ids[:,seq:seq+1])
                samples={n:dict(prefill_ms=[],decode_ms=[],pipeline_ms=[],peak_delta_bytes=[]) for n in runners}
                orders=[]
                for repeat in range(args.repeats+2):
                    order=list(runners);rng.shuffle(order)
                    if repeat>=2:orders.append(order)
                    for name in order:
                        runner=runners[name]
                        torch.cuda.synchronize()
                        base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                        start=time.perf_counter()
                        logits=runner.prefill(prompt)
                        torch.cuda.synchronize()
                        middle=time.perf_counter()
                        for step in range(steps):
                            token=logits[:,-1].argmax(-1,keepdim=True)
                            logits=runner.decode(token)
                        torch.cuda.synchronize()
                        finish=time.perf_counter()
                        if repeat>=2:
                            samples[name]['prefill_ms'].append((middle-start)*1000)
                            samples[name]['decode_ms'].append((finish-middle)*1000)
                            samples[name]['pipeline_ms'].append((finish-start)*1000)
                            samples[name]['peak_delta_bytes'].append(torch.cuda.max_memory_allocated()-base)
                    print('timing',seq,repeat,flush=True)
                case['timing']=dict(samples=samples,orders=orders,
                    medians={n:{k:statistics.median(v) for k,v in rows.items()} for n,rows in samples.items()},
                    paired={n:{p:bootstrap_speedup(samples['dense_flash'][p],samples[n][p])
                               for p in ('decode_ms','pipeline_ms')} for n in samples if n!='dense_flash'})
            case['promoted']=False
            case['decision']='Keep dense default. Approximate candidates require quality AND speed; no external task validation yet.'
            save()
            del runners,sparse,cache,dense,snapshots,reference,actual
            torch.cuda.empty_cache()
        report['decision']='No default change. See per-case quality/speed gates; approximate research code remains opt-in.'
        save()
    print('saved',target,flush=True)


if __name__=='__main__':main()
