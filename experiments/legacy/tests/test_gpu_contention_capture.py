from scripts.capture_gpu_contention import descendants,parse_pmon


def test_descendants_not_unrelated_processes():
    assert descendants(10,[(11,10),(12,11),(13,12),(99,1)])=={10,11,12,13}


def test_unknown_utilization_not_silently_zero():
    text='# gpu pid type sm mem\n0 - - - -\n1 11 C 70 20\n1 99 C - -\n2 98 C 25 10\n'
    rows=parse_pmon(text,{10,11})
    assert rows==[dict(gpu=1,pid=11,sm_percent=70,belongs_to_experiment=True),
        dict(gpu=1,pid=99,sm_percent=None,belongs_to_experiment=False),
        dict(gpu=2,pid=98,sm_percent=25,belongs_to_experiment=False)]
