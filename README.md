# Complete Triton Qwen3-8B

官方 **Qwen/Qwen3-8B（36 层）**的可训练 Triton 模型。
Embedding、所有 Linear、RMSNorm、QK-Norm/RoPE、causal GQA attention、
SwiGLU、residual 和交叉熵的模型计算及反向均由 Triton 实现。
参数、autograd、分配器、FSDP2/NCCL 和 AdamW 使用共同的 PyTorch 基础设施。

GitHub main 的首次发布固定已验收的 V9 实现与小型证据；本研究分支继续以此优化。
这不是完整预训练框架，也不包含推理/KV cache 接口。

## 实测结果

官方 8B 权重，4×RTX A6000，每卡 batch=1、sequence=512，
BF16 计算、FP32 master/梯度/AdamW state，相同 FSDP2 和逐层重计算协议。
数字是两种运行顺序下的完整 optimizer step 中位数，编译和验证不计入。

| 实现 | 顺序一（秒/步） | 顺序二（秒/步） |
|---|---:|---:|
| HF + Flash | 7.0273 | 7.0304 |
| 完整 Triton 基础版 | 7.4540 | 7.4705 |
| 完整 Triton 优化版 | 7.1585 | 7.1529 |

优化版相对 Triton 基础版耗时减少 **3.96% / 4.25%**，
仍比 HF + Flash 慢 **1.87% / 1.74%**，整步 allocated 峰值约 34.14 GiB/卡。
五项双顺序消融中，Linear/GEMM tiling 的整体收益最明确；
其他四项的整步变化很小且方向不一致，不将模块加速当作整体收益。

16 组有效运行均完成 27 次实际更新；全部 399 参数的梯度容差检查、
初末全量验证和模型 CUDA 内核覆盖检查通过。受外部作业干扰的一次记录
已排除，结果披露中断及恢复。数值容差通过不等于与 HF 梯度逐位相同，
短程质量检查不代表完整训练收敛。

## V10：SM86 GEMM 调优研究

本分支增加按形状和布局选择 GEMM 配置的可选路径，并完成 8 组整模型对照。
全量调优相对 V9 两轮快 **0.51% / 0.88%**，仍比同轮 HF 慢 **1.45% / 0.87%**；
仅调优 MLP/head 的方案没有稳定整步收益。数值、短程质量和内核检查全部通过，
两种方案均未达到事先设定的性能采用门槛，**V9 仍为默认配置**。

详见 [实验、2026 年方法评估与复现](OPTIMIZATION_V10.md) 和
[证据索引](results/qwen3_triton_v10_evidence_index_v1.json)。配置保持显式启用；
原 V9 校验器固定历史源码哈希，本分支使用 `scripts/check_qwen3_tuned_source.py`。

## 代码与复现

- [实现、协议、消融和限制](TRITON_QWEN3.md)
- [环境、权重、语料和运行命令](QWEN3_TRITON_REPRODUCE.md)
- [模型](qwen3_triton/model.py)；[算子](qwen3_triton/kernels/)
- [训练入口](scripts/train_qwen3_triton_8b.py)
- [三组整体对照与五项消融](scripts/run_qwen3_triton_suite.py)
- [模块对照](scripts/run_qwen3_triton_module_benchmark.py)
- [整体结果](results/qwen3_triton_v9_paired_training_v1.json)
- [模块结果](results/qwen3_triton_v9_module_benchmark_v1.json)
- [数据及运行验收](results/qwen3_triton_v9_data_acceptance_v1.json)

验收环境：Python 3.11.16、PyTorch 2.10.0+cu128、Triton 3.6.0、
Transformers 4.51.0。依赖版本见 [版本清单](requirements-qwen3-triton.txt)；
克隆后需自行准备匹配环境、官方 8B 权重和语料，不能直接使用本机绝对路径。

仓库同时保留早期 Tiny-Qwen、训练和推理实验，以及
[历史源码](experiments/legacy/README.md)。其性能数字使用各自的协议；
本次完整 Triton 8B 的入口和验收以以上链接为准。

## 源码与数据范围

只提交源码、配置、文档和小型结果。权重、语料、环境、缓存、checkpoint、
原始梯度及 profiler trace 不上传。报告中的本机 artifacts 路径不会随 Git 分发，
复核原始记录需要另行提供这些文件。

第三方来源和许可证保存在 [PROVENANCE](qwen3_triton/PROVENANCE.md)
及 [第三方目录](qwen3_triton/third_party/)。项目自有源码的许可证尚未选定；
本次是代码快照发布，不宣称已完成开源许可配置。
