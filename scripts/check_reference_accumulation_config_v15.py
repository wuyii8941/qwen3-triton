"""CPU regression for the explicit inference reference-accumulation option."""
import json,tempfile,torch
from unittest.mock import patch
from transformers import Qwen3Config,Qwen3ForCausalLM
from qwen3_triton.inference import TritonQwen3Inference
from scripts import generate_qwen3_triton_v15 as cli

def main():
 with tempfile.TemporaryDirectory(dir='/data1/tzh/proj/qwen_operator_lab/.runtime-research/tmp') as tmp:
  config=Qwen3Config(vocab_size=16,hidden_size=128,intermediate_size=256,num_hidden_layers=1,num_attention_heads=1,num_key_value_heads=1,head_dim=128,max_position_embeddings=128,eos_token_id=2,pad_token_id=0)
  hf=Qwen3ForCausalLM(config).to(torch.bfloat16);hf.save_pretrained(tmp)
  model=TritonQwen3Inference.from_pretrained(tmp,device='cpu',reference_accumulation=True,prefill_tiling=True)
  assert model.reference_accumulation is True,'Explicit accumulation choice discarded'
  assert model.prefill_tiling is True and model.sliced_decode is False,'Other option values changed'
  default=TritonQwen3Inference.from_pretrained(tmp,device='cpu')
  assert default.reference_accumulation is False,'Reference arithmetic must remain explicit'
  for name,value in model.state_dict().items():assert torch.equal(value,hf.state_dict()[name]),name
  calls=[]
  def generate(ids,**kw):calls.append(kw);return torch.tensor([[5,6]])
  model.generate=generate
  class Tokenizer:
   def __call__(self,*args,**kw):return {'input_ids':torch.tensor([[5]])}
   def decode(self,*args,**kw):return 'fixture'
  with patch.object(cli.AutoTokenizer,'from_pretrained',return_value=Tokenizer()),patch.object(cli.TritonQwen3Inference,'from_pretrained',return_value=model) as loader,patch('sys.argv',['generate','--model',tmp,'--prompt','fixture','--reference-accumulation']):cli.main()
  assert loader.call_args.kwargs.get('reference_accumulation') is True,'CLI discarded explicit reference arithmetic'
  assert calls[0]['eos_token_id']==2 and calls[0]['pad_token_id']==0,'Generation stop metadata changed'
 print('Reference accumulation survives loading/CLI; default, independent flags and weights retained')
if __name__=='__main__':main()
