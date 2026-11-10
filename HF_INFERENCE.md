# 真实 HF Qwen3 推理接入

推理原型已统一移入 `experiments/legacy/`，不混入当前训练算子库。本文源码相对路径以归档为根，命令在仓库根目录通过 `scripts/legacy_python.sh` 启动。见 [目录导航](REPOSITORY_LAYOUT.md)。

## 接口

model.hf_inference.qwen3_triton_inference 是可恢复的 scoped 替换。默认保留HF模型、Linear、attention与cache API，只替换norm、RoPE和SwiGLU。锁定Transformers4.51.0，CUDA FP16/BF16、同设备同dtype。

~~~python
import torch
from transformers import AutoModelForCausalLM
from model.hf_inference import qwen3_triton_inference

model = AutoModelForCausalLM.from_pretrained(
    "/path/to/Qwen3-1.7B", local_files_only=True,
    torch_dtype=torch.bfloat16, attn_implementation="sdpa",
).cuda().eval()

# input_ids / next_token_ids 为同设备、合法词表范围的token张量。
with torch.inference_mode(), qwen3_triton_inference(model):
    prefill = model(input_ids, use_cache=True, logits_to_keep=1)
    decode = model(next_token_ids, past_key_values=prefill.past_key_values,
                   use_cache=True, logits_to_keep=1)
~~~

context正常或异常退出都会恢复原方法；RoPE函数替换在context内是模块全局行为，只用于单线程实验，不适合并发服务或训练。未经验证的量化、分片/offload和其他Transformers版本不在默认接入范围。

## 数值语义

HF专用kernel保留低精度cast边界：norm乘权重前cast，RoPE两项乘积cast后相加，SwiGLU的SiLU先cast再乘up。它与独立数学kernel的FP32中间计算不完全相同，不声称逐位等价。

HF RoPE直接支持其非连续BHSD view，head_dim方向连续；Q/K head数可不同。split-half配对不能误写成相邻元素interleaving。

## 实验开关

fuse_qk、compact_gqa、fuse_residual、multirow_norm、decode_mlp、packed_qkv、prefill_mlp等均默认关闭。fuse_qk/compact_gqa的FP16完整模型有收益，但BF16扩展输入有超过2%门槛的反例，不能全面放行。

静态cache/Graph使用 model.static_decode.StaticDecodeRunner，不直接传给HF generate；限定固定batch、无padding、等长prompt、连续decode，并依赖旧版Torch2.5.1私有Flash varlen接口。direct_k_cache/direct_v_cache为独立实验。W8A16、频率稀疏等近似方案不默认启用。

## 复现

使用匹配的推理环境；不能将私有静态decode API直接视为Torch2.10已验证。

~~~bash
bash scripts/legacy_python.sh -m pytest -q tests/test_hf_inference.py
python benchmarks/benchmark_hf_inference.py --model /path/to/Qwen3-1.7B --dtype float16 --profile
python benchmarks/benchmark_hf_inference.py --model /path/to/Qwen3-1.7B --dtype bfloat16 --profile
~~~

CPU回归和独立核心GPU矩阵不替代上述真实checkpoint接入门禁。性能口径与日期见 [推理证据](INFERENCE_EVIDENCE.md)，可选路径由其benchmark单独验证，不自动叠加。
