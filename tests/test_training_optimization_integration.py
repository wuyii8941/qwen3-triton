"""Exercise the actual trainer CLI with tiny synthetic fixtures, not a benchmark."""
import gc
import json
import sys
import numpy as np
import pytest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM

pytest.importorskip('flashoptim')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('embedding',[False,True,'consumed'])
def test_graph_trainer_interrupt_checkpoint_resume(tmp_path,monkeypatch,embedding):
    import training.flashoptim_convergence_job as job
    corpus=tmp_path/'corpus';corpus.mkdir()
    rng=np.random.default_rng(5)
    for split,n in (('train',128),('validation',32)):
        (corpus/f'{split}.bin').write_bytes(rng.integers(0,97,n,dtype='<u4').tobytes())
    cfg=Qwen3Config(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,tie_word_embeddings=True)
    cfg._attn_implementation='sdpa'
    def fake_load(*args,**kwargs):return Qwen3ForCausalLM(cfg).to(dtype=torch.bfloat16)
    monkeypatch.setattr(job.AutoModelForCausalLM,'from_pretrained',fake_load)
    protocol=dict(model='synthetic-fixture',corpus=str(corpus),seq=16,seed=7,lr=1e-5,
        plateau=dict(min_delta=.002,patience=8,min_checks=10),max_seconds=120,eval_every=1,log_every=1)
    protocol_path=tmp_path/'protocol.json';protocol_path.write_text(json.dumps(protocol))
    # Fixture-only gate; production gates are separate real-Qwen artifacts.
    gate=tmp_path/'fixture_gate.json';gate.write_text(json.dumps(dict(passed=True,deterministic=True,seq=16,batch=1,consume_proxy=embedding=='consumed')))
    output=tmp_path/'output'
    base=['job','--protocol',str(protocol_path),'--output',str(output),'--optimizer','flashoptim','--execution','graph']
    mode='embedding_consumed' if embedding=='consumed' else ('embedding' if embedding else 'off')
    if embedding:base+=['--optimizations',mode,'--embedding-gate',str(gate)]
    monkeypatch.setattr(sys,'argv',base+['--interrupt-after','2'])
    job.main();gc.collect()
    saved=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
    assert saved['step']==2 and saved['execution']=='graph'
    assert saved['optimizations']==mode
    del saved
    monkeypatch.setattr(sys,'argv',base+['--resume','--interrupt-after','3'])
    job.main();gc.collect()
    saved=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
    assert saved['step']==3
    events=[json.loads(line) for line in (output/'events.jsonl').read_text().splitlines()]
    assert any(e['event']=='preflight_interrupted' and e['step']==3 for e in events)
    # Mode mismatch must be rejected before touching the saved checkpoint.
    changed=list(base);changed[changed.index('graph')]='compile'
    monkeypatch.setattr(sys,'argv',changed+['--resume','--interrupt-after','4'])
    with pytest.raises(ValueError,match='execution mismatch'):job.main()
    assert torch.load(output/'last.pt',map_location='cpu',weights_only=False)['step']==3
