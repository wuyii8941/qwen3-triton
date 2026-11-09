import pytest

from scripts.audit_fsdp_convergence import audit


def example():
    protocol=dict(sequence_length=4,validation=dict(every_updates=2),
        safety=dict(max_updates=100),plateau=dict(min_delta=.002,patience=1,min_checks=2))
    rows=[dict(step=s,loss=2.,elapsed_seconds=s+1,blocks=2,tokens=6) for s in (0,2)]
    return dict(protocol=protocol,backend='official',complete=True,converged=True,
        stop_reason='validation_plateau',best_parameters_restored=True,
        training=[dict(step=s,loss=2.,seconds=1.,max_rank_peak_allocated_gib=1.,
            max_rank_peak_reserved_gib=2.) for s in (1,2)],validation=rows,
        best_step=0,best_validation=2.,plateau_reference_loss=2.,elapsed_seconds=5.,
        test=dict(loss=2.1,blocks=3,tokens=9))


def test_complete_history():
    r=example();assert audit(r,r['protocol'],2,3)['history_consistent']


@pytest.mark.parametrize('fault',['protocol','nonfinite','gap','coverage','best','test','elapsed',
    'first_plateau','not_plateau','reference','missing_validation','restore'])
def test_reject_incomplete_or_inconsistent_claim(fault):
    r=example();protocol=r['protocol']
    if fault=='protocol':r['protocol']={}
    elif fault=='nonfinite':r['training'][0]['loss']=float('nan')
    elif fault=='gap':r['training'][0]['step']=2
    elif fault=='coverage':r['validation'][0]['tokens']=3
    elif fault=='best':r['best_step']=2
    elif fault=='test':r['test']['blocks']=1
    elif fault=='elapsed':r['elapsed_seconds']=1.
    elif fault=='first_plateau':
        r['training'] += [dict(r['training'][0],step=s) for s in (3,4)]
        r['validation'].append(dict(r['validation'][-1],step=4))
    elif fault=='not_plateau':r['validation'][-1]['loss']=1.
    elif fault=='reference':r['plateau_reference_loss']=3.
    elif fault=='missing_validation':r['validation'].pop(0)
    else:r['best_parameters_restored']=False
    with pytest.raises(ValueError):audit(r,protocol,2,3)
