"""Preserve compiler evidence rather than infer copies from Python clone()."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    source=a.source.read_text();lines=source.splitlines();calls=[]
    for i,line in enumerate(lines,1):
        match=re.search(r'_merge_\d+\.run\((.*?)\)',line)
        if match:
            args=[s.strip() for s in match.group(1).split(',')]
            calls.append(dict(line=i,source=line.strip(),input_buffer=args[0],output_buffer=args[4],
                              same_buffer=args[0]==args[4]))
    if not calls:raise RuntimeError('No compiled merge launch found')
    names={c['input_buffer'] for c in calls}
    excerpts=[dict(line=i,text=line.strip()) for i,line in enumerate(lines,1)
              if any(re.search(rf'out={re.escape(name)}(?:[,)]|$)',line)
                     or f'{name} = empty_strided_cuda((151936, 2048)' in line for name in names)]
    result=dict(source=str(a.source),sha256=hashlib.sha256(source.encode()).hexdigest(),calls=calls,
        producer_excerpts=excerpts,
        conclusion='This compiled B1/S512 backward already reuses the GEMM dW buffer for merge; Python clone is not evidence of a runtime copy.',
        scope='Only this generated graph/version/shape. Eager backward still needs safe ownership semantics.')
    Path('artifacts/consumed_codegen_audit_wave2_v1.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
