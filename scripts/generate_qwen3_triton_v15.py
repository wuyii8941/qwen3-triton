"""Generate text from local official Qwen3 BF16 weights using Triton GPU math."""
import argparse,json
from transformers import AutoTokenizer
from qwen3_triton.inference import TritonQwen3Inference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--prompt',required=True)
    p.add_argument('--max-new-tokens',type=int,default=32);p.add_argument('--chat',action='store_true')
    p.add_argument('--enable-thinking',action='store_true');a=p.parse_args()
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    text=tokenizer.apply_chat_template([dict(role='user',content=a.prompt)],tokenize=False,add_generation_prompt=True,enable_thinking=a.enable_thinking) if a.chat else a.prompt
    ids=tokenizer(text,return_tensors='pt')['input_ids']
    model=TritonQwen3Inference.from_pretrained(a.model)
    with model.runtime:ids=ids.to(model.device)
    out=model.generate(ids,max_new_tokens=a.max_new_tokens,eos_token_id=model.generation_config.eos_token_id,pad_token_id=model.generation_config.pad_token_id)
    tokens=out[0,ids.shape[1]:].cpu().tolist()
    print(json.dumps(dict(generated_text=tokenizer.decode(tokens,skip_special_tokens=True),token_ids=tokens),ensure_ascii=False))


if __name__=='__main__':main()
