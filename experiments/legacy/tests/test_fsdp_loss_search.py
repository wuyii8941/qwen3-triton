import pytest
from training.fsdp_loss_search import summarize_loss_screen
from tests.test_fsdp_search import row


def loss_row(chunk, seconds):
    r = row(8, seconds)
    r.pop('keep_full_layers')
    r['chunk_size'] = chunk
    r['result']['optimization']['loss_chunk_size'] = chunk
    return r


def test_loss_screen_selects_only_matched_successful_cases():
    rows = [loss_row(0, 7), loss_row(128, 6.8), dict(chunk_size=512, status='oom')]
    assert summarize_loss_screen(rows)['best_chunk_size'] == 128
    assert summarize_loss_screen([loss_row(0, 7), loss_row(128, 8)])['best_chunk_size'] == 0


@pytest.mark.parametrize('fault', ['input', 'retention', 'label', 'nan', 'incomplete', 'missing', 'failed', 'memory', 'sources'])
def test_rejects_unmatched_or_invalid_loss_results(fault):
    rows = [loss_row(0, 7), loss_row(128, 6)]
    r = rows[1]['result']
    if fault == 'input': r['manifest']['input_sha256'] = 'wrong'
    elif fault == 'sources': r['manifest']['implementation_sources'] = {'sha256': 'different'}
    elif fault == 'retention': r['optimization']['keep_full_layers'] = 0
    elif fault == 'label': r['optimization']['loss_chunk_size'] = 32
    elif fault == 'nan': r['median_step_seconds'] = float('nan')
    elif fault == 'incomplete': r['complete'] = False
    elif fault == 'missing': rows = rows[1:]
    elif fault == 'failed': rows[1]['status'] = 'failed'
    elif fault == 'memory': r['records'][0]['peak_allocated_gib'] = 43
    with pytest.raises(ValueError): summarize_loss_screen(rows)
