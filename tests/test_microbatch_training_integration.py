"""Actual trainer, tiny checkpoints; verify sample cursor and batch resume guards."""
import gc
import json
import sys
import numpy as np
import pytest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM

pytest.importorskip('flashoptim')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_microbatch_accumulation_checkpoint_resume(tmp_path,monkeypatch):
    import training.flashoptim_convergence_job as job
    corpus=tmp_path/'corpus';corpus.mkdir();rng=np.random.default_rng(18)
    for split,n in (('train',256),('validation',32)):
        (corpus/f'{split}.bin').write_bytes(rng.integers(0,97,n,dtype='<u4').tobytes())
    cfg=Qwen3Config(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,tie_word_embeddings=True)
    cfg._attn_implementation='sdpa'
    monkeypatch.setattr(job.AutoModelForCausalLM,'from_pretrained',lambda *a,**k:Qwen3ForCausalLM(cfg).to(dtype=torch.bfloat16))
    protocol=tmp_path/'protocol.json';protocol.write_text(json.dumps(dict(model='fixture',corpus=str(corpus),
        seq=16,seed=8,lr=1e-5,plateau=dict(min_delta=.002,patience=8,min_checks=10),max_seconds=180,eval_every=1,log_every=1)))
    gate=tmp_path/'gate.json';gate.write_text(json.dumps(dict(passed=True,deterministic=True,seq=16,batch=2,
        consume_proxy=True,accumulation=2)))
    output=tmp_path/'output'
    argv=['job','--protocol',str(protocol),'--output',str(output),'--optimizer','flashoptim','--execution','compile',
        '--optimizations','embedding_consumed','--embedding-gate',str(gate),'--microbatch','2','--accumulation','2']
    monkeypatch.setattr(sys,'argv',argv+['--interrupt-after','2']);job.main();gc.collect()
    monkeypatch.setattr(sys,'argv',argv+['--resume','--interrupt-after','3']);job.main();gc.collect()
    state=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
    assert (state['step'],state['microbatch'],state['accumulation'])==(3,2,2)
    train=[json.loads(s) for s in (output/'events.jsonl').read_text().splitlines() if json.loads(s)['event']=='train']
    assert [r['samples_seen'] for r in train]==[4,8,12]
    other_gate=tmp_path/'other_gate.json';other_gate.write_text(json.dumps(dict(passed=True,deterministic=True,
        seq=16,batch=2,consume_proxy=True,accumulation=1)))
    changed=list(argv);changed[-1]='1';changed[changed.index(str(gate))]=str(other_gate)
    monkeypatch.setattr(sys,'argv',changed+['--resume','--interrupt-after','4'])
    with pytest.raises(ValueError,match='batch/accumulation mismatch'):job.main()
    assert torch.load(output/'last.pt',map_location='cpu',weights_only=False)['step']==3
