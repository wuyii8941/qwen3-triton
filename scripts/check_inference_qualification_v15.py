"""CPU acceptance regression: functional checks cannot replace fixed validation."""
from scripts import verify_qwen3_inference_v15 as verifier

def run():
 args=dict(diagnostic=False,numerics=True,greedy=True,traces={name:{'passed':True} for name in ['setup','prefill17','prefill','decode','generate']},foreign=[],corpus_rows=[dict(block=i,passed=True) for i in range(512)],corpus_blocks=512,corpus_sequence=512)
 result=verifier.qualification(**args)
 assert result['qualified'] and result['functional_qualified'] and result['corpus_numerics'] and result['full_validation']
 for changed in [dict(corpus_rows=[],corpus_blocks=0),dict(corpus_rows=args['corpus_rows'][:-1]),dict(corpus_rows=[dict(block=0,passed=True)]*512),dict(corpus_rows=[dict(block=i+512,passed=True) for i in range(512)]),dict(corpus_rows=[dict(block=i,passed=i!=300) for i in range(512)]),dict(corpus_sequence=128),dict(foreign=['foreignGPUprocess']),dict(diagnostic=True),dict(numerics=False),dict(greedy=False),dict(traces={})]:
  test={**args,**changed};assert not verifier.qualification(**test)['qualified'],changed
 missing={**args['traces']};missing.pop('prefill17');assert not verifier.qualification(**{**args,'traces':missing})['functional_qualified']
 assert verifier.qualification(**{**args,'corpus_rows':[],'corpus_blocks':0})['functional_qualified']
 print('Full acceptance requires complete unique fixed data, numerical/greedy gates, all phase audits and clean monitoring')
if __name__=='__main__':run()
