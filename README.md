# Qwen3-8B Triton 训练与推理

四卡 FSDP2 训练和单卡 BF16 推理，支持固定请求 CUDA Graph。训练使用 Triton 算子和 AdamW，通信由 Triton reduce-scatter 与 NCCL 协作完成。

## 安装

在 Linux CUDA 环境使用 Python 3.11，并按 `requirements-qwen3-triton.txt` 安装锁定版本：

```bash
python3.11 -m venv .venv-research
.venv-research/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.10.0+cu128'
.venv-research/bin/python -m pip install -r requirements-qwen3-triton.txt
```

通过 `bash scripts/research_python.sh ...` 启动项目命令。权重和语料使用本地路径。训练与推理验证的输出目录须位于 `/data1/tzh` 下。输入校验规则在 `configs/input_recipe.json`，训练 GEMM 计划在 `configs/gemm_plan.json`。

## 运行

| 工作 | 入口 |
| --- | --- |
| 准备固定语料 | `scripts/prepare_corpus.py` |
| 绑定本地权重和语料 | `scripts/configure_inputs.py` |
| 导出 HF 训练参考 | `scripts/export_training_reference.py` |
| 四卡训练 | `scripts/train.py` |
| 单卡生成 | `scripts/generate.py` |
| 验证推理数值与性能 | `scripts/verify_inference.py` |

完整命令见[训练指南](TRAINING.md)和[推理指南](INFERENCE.md)，实测收益和适用条件见[结果汇总](docs/RESULTS.md)。

## 代码结构

- `qwen3_triton/training/` 训练模型、数据、输入校验与 FSDP。
- `qwen3_triton/inference/` 推理模型、KV cache 和 CUDA Graph。
- `qwen3_triton/kernels/` 计算算子。
- `qwen3_triton/runtime/` CUDA 调度、显存、通信与优化器。
- `qwen3_triton/validation/` 验证与参考比较。

第三方来源及其许可证见[来源说明](qwen3_triton/PROVENANCE.md)。
