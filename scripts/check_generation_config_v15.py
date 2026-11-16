"""CPU regression: locally pinned generation stop IDs survive inference loading."""
import contextlib,json,tempfile,torch
from pathlib import Path
from unittest.mock import patch
from transformers import Qwen3Config,Qwen3ForCausalLM
from qwen3_triton.inference import TritonQwen3Inference
from scripts import generate_qwen3_triton_v15 as cli

def main():
 with tempfile.TemporaryDirectory(dir='/data1/tzh/proj/qwen_operator_lab/.runtime-research/tmp') as tmp:
  c=Qwen3Config(vocab_size=16,hidden_size=128,intermediate_size=128,num_hidden_layers=1,num_attention_heads=1,num_key_value_heads=1,head_dim=128,max_position_embeddings=128,eos_token_id=2,pad_token_id=0)
  hf=Qwen3ForCausalLM(c).to(torch.bfloat16);hf.save_pretrained(tmp)
  p=Path(tmp)/'generation_config.json';d=json.loads(p.read_text());d.update(eos_token_id=[2,3],pad_token_id=0);p.write_text(json.dumps(d))
  model=TritonQwen3Inference.from_pretrained(tmp,device='cpu')
  assert getattr(model,'generation_config',None) is not None,'Inference model discarded local generation_config.json'
  assert model.generation_config.eos_token_id==[2,3]
  calls=[]
  def generate(ids,**kw):calls.append(kw);return torch.tensor([[5,6]])
  model.generate=generate
  class Tokenizer:
   eos_token_id=2;pad_token_id=7
   def __call__(self,*a,**kw):return {'input_ids':torch.tensor([[5]])}
   def decode(self,*a,**kw):return 'fixture'
  with patch.object(cli.AutoTokenizer,'from_pretrained',return_value=Tokenizer()),patch.object(cli.TritonQwen3Inference,'from_pretrained',return_value=model) as loader,patch('sys.argv',['generate','--model',tmp,'--prompt','fixture','--prefill-tiling']):cli.main()
  assert loader.call_args.kwargs.get('prefill_tiling') is True,'CLI discarded explicit prefill tiling option'
  assert calls[0].get('eos_token_id')==[2,3] and calls[0].get('pad_token_id')==0,calls
 print('Local EOS list/padding retained and used by CLI')
if __name__=='__main__':main()
