from scripts.report_fsdp_1p7b_accumulation import pair_ratios


def test_pair_ratios_not_averaged_or_reversed():
    runs={(0,'large'):{'median_step_seconds':1.},(1,'large'):{'median_step_seconds':2.},
        (0,'each'):{'median_step_seconds':1.5},(1,'each'):{'median_step_seconds':4.}}
    assert pair_ratios(runs,'large','each')==[1.5,2.]


def test_missing_or_oom_run_not_a_speedup():
    runs={(0,'large'):{'median_step_seconds':1.},(0,'each'):{'median_step_seconds':2.}}
    assert pair_ratios(runs,'large','each')==[2.,None]
