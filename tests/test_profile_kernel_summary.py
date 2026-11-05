import json
from benchmarks.profile_current_training import kernel_summary


def test_only_leaf_kernel_categories_are_summed(tmp_path):
    trace=dict(traceEvents=[dict(cat='gpu_user_annotation',name='CompiledFxGraph',dur=500),
        dict(cat='cpu_op',name='aten::mm',dur=300),dict(cat='kernel',name='gemm',dur=10),
        dict(cat='kernel',name='gemm',dur=11),dict(cat='gpu_memcpy',name='copy',dur=5)])
    (tmp_path/'trace.json').write_text(json.dumps(trace))
    (tmp_path/'profile.json').write_text(json.dumps(dict(profile_steps=2,events=[])))
    rows=kernel_summary(tmp_path)
    assert rows==[dict(name='gemm',count=2,total_us=21.)]
