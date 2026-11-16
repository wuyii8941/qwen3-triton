"""CPU regression: inference tile choice survives local checkpoint loading."""
import tempfile
from pathlib import Path
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from qwen3_triton.inference import TritonQwen3Inference

def main():
 with tempfile.TemporaryDirectory(dir='/data1/tzh/proj/qwen_operator_lab/.runtime-research/tmp') as tmp:
  config=Qwen3Config(vocab_size=16,hidden_size=128,intermediate_size=256,num_hidden_layers=1,num_attention_heads=1,num_key_value_heads=1,head_dim=128,max_position_embeddings=128)
  hf=Qwen3ForCausalLM(config).to(torch.bfloat16);hf.save_pretrained(tmp)
  model=TritonQwen3Inference.from_pretrained(tmp,device='cpu',prefill_tiling=True)
  assert model.prefill_tiling is True,'Explicit inference tile choice discarded by loader'
  assert model.sliced_decode is False,'Independent decode experiment was accidentally enabled'
  default=TritonQwen3Inference.from_pretrained(tmp,device='cpu')
  assert default.prefill_tiling is False,'Research tiling must be explicit'
  for name,weight in model.state_dict().items():assert torch.equal(weight,hf.state_dict()[name]),name
 print('Explicit prefill tiling survives loading; default/decode configuration and weights retained')
if __name__=='__main__':main()
