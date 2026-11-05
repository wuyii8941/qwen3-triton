"""Independent test window confirms or rejects the operator-screen candidate.

Compare native Flash, the EXACT own dense kernel, and fixed 1e-4 threshold.
No thresholds chosen on this second window. All paths remain opt-in.
"""
import argparse
import json
import random
import statistics
import time
import torch
import transformers
import triton
from transformers import AutoModelForCausalLM,AutoTokenizer
from torch.nn.attention import sdpa_kernel,SDPBackend
from model.hf_inference import qwen3_triton_inference,pack_qkv_weights
from model.static_decode import StaticDecodeRunner
from model.block_skip import BlockSkipRunner
from benchmarks.benchmark_frequency_sparse import ROOT,MODEL,sha,quality
from benchmarks.benchmark_hf_attention_e2e import bootstrap_speedup


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',default='v1')
    parser.add_argument('--seqs',default='8192,32768')
    parser.add_argument('--steps',type=int,default=128)
    parser.add_argument('--repeats',type=int,default=8)
    parser.add_argument('--test-offset',type=int,default=65536)
    args=parser.parse_args()
    if args.repeats<8 or args.steps<8:raise ValueError('At least 8 repeats/steps required')
    target=ROOT/'artifacts'/f'block_skip_e2e_{args.tag}.json'
    if target.exists():raise FileExistsError(target)
    target.touch(exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(20261101)
    rng=random.Random(20261101)
    data=json.loads((ROOT/'artifacts/frequency_wikitext_data.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    tokens=torch.tensor(tokenizer(data['splits']['test']['text'],add_special_tokens=False,verbose=False)['input_ids'])
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,
        torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
    packed=pack_qkv_weights(model)
    report=dict(manifest=dict(date='2026-11-01',model=str(MODEL),gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,triton=triton.__version__,transformers=transformers.__version__,
        cuda=torch.version.cuda,dtype='float16',batch_size=1,test_offset=args.test_offset,
        seed=20261101,config_sha256=sha(MODEL/'config.json'),data_sha256=data['splits']['test']['sha256'],
        source_sha256={p:sha(ROOT/p) for p in ['ops/triton/block_skip_decode.py','model/block_skip.py',
            'model/static_decode.py','model/hf_inference.py','benchmarks/benchmark_block_skip_e2e.py']},
        choice='operator screening on offset0; fixed S16/BN128, threshold=1e-4; independent window offset65536',
        timing='8 randomized paired rounds after 2 warmup rounds; synchronized wall time; full prefill+128 greedy decode',
        quality='128 teacher-forced continuation steps, not LongBench; weights identical, no quantization',
        candidates={'own_dense':dict(threshold=0),'block_skip':dict(threshold=1e-4)},
        promotion_gate='>=5% pipeline gain, CI lower>1, max logit L2<2%, top1>=99%, KLmean<=.005, PPLratio<=1.01; external task eval additionally required'),cases=[])
    def save():target.write_text(json.dumps(report,indent=2))
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION),qwen3_triton_inference(model,fuse_qk=True,
            compact_gqa=True,decode_mlp=True,packed_qkv=True,packed_weights=packed,prefill_mlp=True,
            static_decode=True,direct_k_cache=True,direct_v_cache=True,fuse_residual=True):
        for seq in map(int,args.seqs.split(',')):
            if seq+args.steps+1>model.config.max_position_embeddings:raise ValueError('Context limit')
            ids=tokens[args.test_offset:args.test_offset+seq+args.steps+1].unsqueeze(0).cuda()
            if ids.shape[1]!=seq+args.steps+1:raise ValueError('Insufficient test tokens')
            prompt=ids[:,:seq].contiguous()
            cap=seq+args.steps+1
            runners={'dense_flash':StaticDecodeRunner(model,1,cap),
                'own_dense':BlockSkipRunner(model,1,cap,threshold=0),
                'block_skip':BlockSkipRunner(model,1,cap,threshold=1e-4)}
            case=dict(seq_len=seq,decode_steps=args.steps,quality={},timing={})
            report['cases'].append(case)
            for name,runner in runners.items():
                runner.prefill(prompt)
                logits=[]
                for pos in range(seq,seq+args.steps):
                    logits.append(runner.decode(ids[:,pos:pos+1],replay=False)[0,0].cpu())
                actual=torch.stack(logits)
                if name=='dense_flash':reference=actual
                else:
                    case['quality'][name]=quality(actual,reference,ids[0,seq+1:seq+args.steps+1].cpu())
                    print('quality',seq,name,case['quality'][name],flush=True)
                runner.prefill(prompt)
                runner.capture(ids[:,seq:seq+1])
            save()
            samples={n:{p:[] for p in ('prefill_ms','decode_ms','pipeline_ms','peak_delta_bytes')} for n in runners}
            orders=[]
            for repeat in range(args.repeats+2):
                order=list(runners);rng.shuffle(order)
                if repeat>=2:orders.append(order)
                for name in order:
                    runner=runners[name]
                    torch.cuda.synchronize()
                    base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                    start=time.perf_counter()
                    out=runner.prefill(prompt)
                    torch.cuda.synchronize();middle=time.perf_counter()
                    for _ in range(args.steps):
                        token=out[:,-1].argmax(-1,keepdim=True)
                        out=runner.decode(token)
                    torch.cuda.synchronize();end=time.perf_counter()
                    if repeat>=2:
                        samples[name]['prefill_ms'].append((middle-start)*1000)
                        samples[name]['decode_ms'].append((end-middle)*1000)
                        samples[name]['pipeline_ms'].append((end-start)*1000)
                        samples[name]['peak_delta_bytes'].append(torch.cuda.max_memory_allocated()-base)
                print('timing',seq,repeat,flush=True)
            case['timing']=dict(samples=samples,orders=orders,
                medians={n:{p:statistics.median(v) for p,v in rows.items()} for n,rows in samples.items()},
                paired={n:{p:bootstrap_speedup(samples['dense_flash'][p],samples[n][p])
                           for p in ('decode_ms','pipeline_ms')} for n in samples if n!='dense_flash'})
            case['promoted']=False
            save()
            del runners,runner,reference,actual,logits,out
            torch.cuda.empty_cache()
    report['decision']='Default dense unchanged; explicit opt-in experiment only. See both numerical and speed gates.'
    save()
    print('saved',target,flush=True)


if __name__=='__main__':main()
