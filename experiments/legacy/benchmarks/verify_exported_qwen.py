"""Single-GPU stock-HF reload parity and short greedy-generation smoke."""
import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--reference',required=True)
    p.add_argument('--output',required=True)
    args = p.parse_args()
    saved = torch.load(args.reference,map_location='cpu',weights_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    with torch.no_grad():
        actual = model(saved['input_ids'].cuda(),use_cache=False).logits[:,-1].cpu()
        torch.testing.assert_close(actual,saved['last_logits'],atol=.03,rtol=.01)
        tokenizer = AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        prompt = tokenizer.apply_chat_template([dict(role='user',content='What is 2 + 2?')],
            tokenize=False,add_generation_prompt=True)
        inputs = tokenizer(prompt,return_tensors='pt').to('cuda')
        generated = model.generate(**inputs,max_new_tokens=32,do_sample=False,
            pad_token_id=tokenizer.eos_token_id)
        result = dict(passed=True,max_abs=(actual.float()-saved['last_logits'].float()).abs().max().item(),
            generated_tokens=generated.shape[1]-inputs.input_ids.shape[1],
            generated_text=tokenizer.decode(generated[0,inputs.input_ids.shape[1]:],skip_special_tokens=True),
            note='Generation smoke only, not a model quality evaluation')
    with open(args.output,'w') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(result),flush=True)


if __name__ == '__main__': main()
