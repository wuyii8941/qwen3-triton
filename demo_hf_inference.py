"""Greedy text generation smoke demo, outside performance timing."""
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.hf_inference import qwen3_triton_inference


@torch.inference_mode()
def main():
    path = '/data1/tzh/models/Qwen/Qwen3-1.7B'
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    prompt = tokenizer.apply_chat_template(
        [{'role':'user','content':'Explain GPU kernel fusion in one sentence.'}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    options = dict(max_new_tokens=32, do_sample=False, use_cache=True,
                   pad_token_id=tokenizer.eos_token_id)
    before = model.generate(**inputs, **options)
    with qwen3_triton_inference(model):
        after = model.generate(**inputs, **options)
    n = inputs.input_ids.shape[1]
    report = dict(prompt=prompt, baseline=tokenizer.decode(before[0,n:], skip_special_tokens=True),
                  triton=tokenizer.decode(after[0,n:], skip_special_tokens=True),
                  baseline_ids=before[0,n:].tolist(), triton_ids=after[0,n:].tolist(),
                  exact_token_match=torch.equal(before,after), max_new_tokens=32,
                  scope='one prompt smoke test, not a generation quality evaluation')
    target = Path(__file__).parent/'artifacts/hf_generation_demo.json'
    target.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
