"""Check releasing the ephemeral shared-weight proxy after the LM head uses it."""
import json
from pathlib import Path
import sys
import benchmarks.check_tied_embedding as harness
from training.tied_embedding import patch_tied_embedding


if __name__=='__main__':
    harness.patch=lambda model:patch_tied_embedding(model,consume_proxy=True)
    harness.main()
    path=Path('artifacts')/sys.argv[sys.argv.index('--tag')+1]/'result.json'
    result=json.loads(path.read_text());result['consume_proxy']=True
    path.write_text(json.dumps(result,indent=2))
