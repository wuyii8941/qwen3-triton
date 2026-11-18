# Qwen3-8B Triton 训练与推理

本仓库只保留一套可运行的 Qwen3-8B 实现：四卡 FSDP 训练与单卡 BF16 推理。历史版本、对照分支和原始实验产物均在 Git 历史中，不再放在当前源码树。

## 稳定入口

| 工作 | 命令 |
| --- | --- |
| 准备固定语料 | `scripts/prepare_corpus.py` |
| 绑定本地权重与语料 | `scripts/configure_inputs.py` |
| 导出 HF 训练参考 | `scripts/export_training_reference.py` |
| 运行完整 Triton 训练 | `scripts/train.py` |
| 单卡生成 | `scripts/generate.py` |
| 推理正确性与性能验证 | `scripts/verify_inference.py` |

依赖安装：`python -m pip install -r requirements-qwen3-triton.txt`。研究环境统一通过 `bash scripts/research_python.sh ...` 启动，使缓存和临时文件留在项目的 `.runtime-research` 中。

训练输入的固定校验规则在 `configs/input_recipe.json`，矩阵乘计划在 `configs/gemm_plan.json`。权重、语料、运行输出和 profiler trace 不随仓库发布。

- 完整训练步骤见 [TRAINING.md](TRAINING.md)
- 推理和验证见 [INFERENCE.md](INFERENCE.md)
- 已有性能结论及适用范围见 [docs/RESULTS.md](docs/RESULTS.md)
- 第三方来源见 [qwen3_triton/PROVENANCE.md](qwen3_triton/PROVENANCE.md)

当前保留的实现仍让 PyTorch 负责 autograd、FSDP 主机调度、CUDA 资源管理和请求控制；模型 GPU 数学、优化器与完整运行时的数据操作由 Triton 路径执行。历史性能不能替代在目标机器上的重新验收。
