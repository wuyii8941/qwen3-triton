"""Inspect generated PTX, not inferred DRAM traffic or a throughput claim."""
import json
from pathlib import Path
import re
import torch
from training.multi_tensor_flashoptim import MultiTensorFlashAdamW,_batched_update,_batched_update_vectorized


def main():
    rows=[]
    for vectorized,kernel in ((False,_batched_update),(True,_batched_update_vectorized)):
        p=torch.nn.Parameter(torch.randn(2048*2048,device='cuda',dtype=torch.bfloat16))
        p.grad=torch.randn_like(p)
        opt=MultiTensorFlashAdamW([p],max_numel=2**31-1,persistent=True,vectorized=vectorized,
                                 master_weight_bits=None,lr=1e-5,decouple_lr=False)
        opt.step();torch.cuda.synchronize()
        compiled=list(kernel.device_caches[0][0].values())[-1]
        ptx=compiled.asm['ptx']
        rows.append(dict(vectorized=vectorized,registers=compiled.n_regs,
            global_load_instructions=re.findall(r'ld\.global[^;]+;',ptx),
            global_store_instructions=re.findall(r'st\.global[^;]+;',ptx)))
    Path('artifacts/optimizer_vectorization_audit_v1.json').write_text(json.dumps(rows,indent=2))
    print([(r['vectorized'],r['registers'],len(r['global_load_instructions']),
            [x for x in r['global_load_instructions'] if '.v' in x]) for r in rows],flush=True)


if __name__=='__main__':main()
