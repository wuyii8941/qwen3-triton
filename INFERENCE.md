# 推理

## 支持范围

V15 提供单卡 BF16 的 dense prefill、分块 prefill、请求独立 KV cache、单 token decode、贪心生成及固定请求 CUDA Graph。Embedding、Linear、归一化/RoPE、Attention、缓存写入和选词等 GPU 计算使用 Triton。Python/PyTorch 负责请求控制、内存分配和 CUDA Graph 等主机服务。

当前不提供 padded/ragged 批处理、服务调度、采样、beam search、量化或张量并行。验证不能推广到未测硬件和任意输入。

## 使用

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/research_python.sh \
  scripts/generate_qwen3_triton_v15.py \
  --model /data1/tzh/models/Qwen/Qwen3-8B \
  --reference-accumulation --prefill-tiling \
  --chat --prompt 'What is the capital of France?' --max-new-tokens 32
```

将 `--model` 路径替换为本机官方权重目录。上面的命令是 Eager 生成。两个选项默认关闭；性能测试使用它们并使用 CUDA Graph。接口示例：

```python
from qwen3_triton.inference import TritonQwen3Inference, CapturedGreedyRequest

model = TritonQwen3Inference.from_pretrained(
    '/data1/tzh/models/Qwen/Qwen3-8B',
    reference_accumulation=True, prefill_tiling=True)
# ids 为无 padding、连续的 CUDA 整数张量，形状 [batch, sequence]。
request = CapturedGreedyRequest(model, ids, max_new_tokens=8)
tokens = request(ids)
```

Graph 固定输入形状和生成 token 数，不提前停止；返回结果由请求对象持有，下一次 replay 会覆盖它。每个对象仅限单流、非并发使用。手动调用 `forward` 做连续 decode 时，应先用 `model.new_cache(batch_size, max_length)` 分配足够容量。

## 验证与性能

官方 Qwen3-8B：24 项原数值检查、两条 16-token 生成对照、完整 512 块 S512 验证及五段 GPU trace 检查通过；验证语料 logits 逐位一致。额外逐步文本 logits 诊断仍有 23 个超阈值项，不能声称任意输入均与 HF 数值等价。

单 A6000、B1/S65、生成 8 token 的两轮固定 Graph 请求中，Triton 为 202.1/202.2 ms，HF 为 220.3/220.2 ms；Eager 耗时仍约为 HF 的 3.1–3.2 倍。编译、加载和 Graph 创建不计入。完整样本、历史失败和限制见 [结果汇总](docs/RESULTS.md) 与 [详细推理记录](docs/history/INFERENCE_V15.md)。
