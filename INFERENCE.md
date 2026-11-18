# Qwen3-8B 推理

## 支持范围

最终推理路径支持单卡 BF16、dense/分块 prefill、请求独立 KV cache、单 token decode、贪心生成和固定请求 CUDA Graph。当前不包含服务调度、采样、beam search、量化、TP 或 PP。

## 生成

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/research_python.sh scripts/generate.py \
  --model weights/Qwen3-8B \
  --prompt 'What is the capital of France?' \
  --max-new-tokens 32 --chat \
  --reference-accumulation --prefill-tiling
```

`--model` 必须指向本地官方 Qwen3-8B 权重。两个优化开关默认关闭，便于显式比较。

## 验证

`scripts/verify_inference.py` 对固定模型做数值审计和相反执行顺序的延迟比较，并把小型汇总写到指定的新目录：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/research_python.sh scripts/verify_inference.py \
  --model weights/Qwen3-8B \
  --identity artifacts/inputs/input_manifest.json \
  --prefill-tiling --reference-accumulation \
  --output artifacts/inference_validation
```

训练所用的固定输入定义位于 `configs/input_recipe.json`，推理/训练共同使用的 GEMM 计划位于 `configs/gemm_plan.json`。Graph 固定输入形状和生成 token 数，不提前停止；请求对象不可跨流并发复用。

历史 A6000 数据见 [docs/RESULTS.md](docs/RESULTS.md)。这些数据只覆盖固定形状和设备，不能推广到任意输入、批量或服务场景。
