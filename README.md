# Complete Triton Qwen3-8B

官方 **Qwen/Qwen3-8B（36 层）**的可训练 Triton 模型。
Embedding、所有 Linear、RMSNorm、QK-Norm/RoPE、causal GQA attention、
SwiGLU、residual 和交叉熵的模型计算及反向均由 Triton 实现。
V9–V13 的参数、autograd、分配器、FSDP2/NCCL 和 AdamW 使用共同的 PyTorch 基础设施。

GitHub main 的首次发布固定已验收的 V9 实现与小型证据；本研究分支继续以此优化。
训练版本的验收范围不包含完整预训练框架。新增 [V15 推理实现](INFERENCE_V15.md)
提供单卡 BF16 prefill、KV cache、decode、贪心生成和固定请求 CUDA Graph；
显式 `reference_accumulation=True, prefill_tiling=True` 路径已通过原24项数值检查、
两条16-token生成对照、完整512块S512验证和五段GPU审计；验证语料logits逐位一致。
额外逐步logits诊断仍有超阈值项，因此结论限于已测协议，两个选项均默认关闭。
旧 `prefill_tiling` 单独开启的5.1% Graph收益属于未通过完整质量检查的历史路径，
不能当作新验收路径的性能数字。最新验收含短行RMSNorm修正，原先失败的新增提示回归也已通过。
[正式证据](results/qwen3_triton_inference_v15_reference_api_qualified_v2.json)。
新路径在单A6000、B1S65、生成8token的完整Graph请求中为 **201.4 / 201.2 ms**，
同轮HF Graph为 **218.8 / 219.0 ms**，耗时减少 **7.95% / 8.12%**。
三条更换输入的生成对照均通过。Eager仍约比HF慢3.63倍，收益限于上述Graph场景。
[完整配对样本与限制](results/qwen3_triton_inference_v15_reference_paired_request_v1.json)。

本分支最新通过完整验收的是 [V13](OPTIMIZATION_V13.md)：
两轮为 **4.5148 / 4.5176 秒/步**，
相对同轮 HF＋Flash 耗时减少 **14.75% / 13.80%**，
相对已有 V12 减少 **7.95% / 7.85%**。下文保留各代各自的实验协议与结果。

[V14 完整 GPU 运行时](TRITON_RUNTIME_V14.md) 已通过完整8B功能验收：四rank、
全399参数、16步质量检查、初末512块验证、显存和完整GPU事件审核均通过。
完整 GPU 数据通路由 Triton 实现，CPU FSDP/autograd 调度与 CUDA 驱动服务保留。
但同一正式对照中 V14 为 **5.927 秒/步**，比 HF＋Flash 慢 **12.3%**，比 V13
慢 **31.1%**，因此没有采用为性能默认。新[运行时入口](scripts/train_qwen3_runtime_v14.py)
仍默认 `--runtime full` 以复现完整实现；性能推荐继续使用已验收的 V13。
完整协议与审核摘要见[正式证据索引](results/qwen3_triton_runtime_v14_evidence_index_v1.json)。

## V9：首次完整实现的实测结果

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

## V11：保留参数、减少训练通信

本分支新增可选训练策略：前向后保留 36 层及 LM head 的完整参数，跳过
embedding 反向不需要的权重聚合，逐层重计算保持不变，并显式使用 V10 GEMM 计划。
六组完整 8B 对照及独立审查通过；两轮整步耗时从 **7.141 / 7.195 秒**降至
**5.249 / 5.214 秒**，分别减少 **26.50% / 27.54%**。

实际 AllGather 从 78 降到 40 次，ReduceScatter 仍为 39 次。代价是计时训练区间
PyTorch allocated 峰值从约 **34.14 增至 40.18 GiB/卡**，allocated/reserved 均在
原 41.875 GiB 预算内。相同新策略的 HF 为 **5.246 / 5.251 秒**，与 Triton
总体持平。这是组合训练策略的收益，不能全部归因于 Triton 算子。

详见 [实验与显式使用方法](OPTIMIZATION_V11.md)、
[策略参数](results/qwen3_triton_v11_strategy_v1.json) 和
[独立审查及证据索引](results/qwen3_triton_v11_evidence_index_v1.json)。
原入口默认值不变；结果限于上述硬件与形状，16 次质量更新不证明长期收敛，
本轮没有测量推理性能。

## V12：词表分片输出层与融合梯度

本分支新增完整 Triton 训练的可选词表分片投影/CE 路径，改造输出层参数所有权、
通信对象及分布式反向；decoder 的 FSDP、逐层重计算和 native fused AdamW
与基线相同。它需要定制计算图；HF 也可以移植，不能称为 HF 无法实现的算法。

| 同轮完整 8B 训练 | 顺序一，秒/步 | 逆序，秒/步 |
| --- | ---: | ---: |
| 原生 HF＋Flash，共有 V11 策略 | 5.285977 | 5.249324 |
| V11 Triton 控制 | 5.212518 | 5.208927 |
| Triton 词表分片 head | 4.909584 | 4.899512 |
| Triton 词表分片 head＋融合 dW | **4.902588** | **4.887280** |

推荐融合版相对同轮 HF 耗时减少 **7.25% / 6.90%**，相对 V11 Triton
减少 **5.95% / 6.17%**；allocated 峰值 **37.468 GiB/卡**，HF 为 40.248。
dW 融合自身整步增量仅 **0.14% / 0.25%**，另省 0.301 GiB/卡。

推荐方案及基线通过全部 399 参数首步梯度、16 次质量更新、初末全量验证、
CUDA 覆盖和资源审查。辅助 HF 定制 head 两轮均有一项梯度检查失败，拒绝其
性能结论；原始全队列报告保留失败，逐方法审核没有放宽容差。
短程质量检查不代表长期收敛；结果限于以上硬件与形状，未测推理性能。

详见 [实现、完整对照、失败记录及下一轮方向](OPTIMIZATION_V12.md)、
[逐方法审查](results/qwen3_triton_v12_method_review_v1.json)、
[原始全队列报告](results/qwen3_triton_v12_cohort_review_v1.json)、
[显式策略](results/qwen3_triton_v12_strategy_v1.json) 和
[证据索引](results/qwen3_triton_v12_evidence_index_v1.json)。新入口显式启用，
原默认配置不变；不能将本轮增量与 V11 的公共策略收益重复计算。

## V13：无损梯度传输与 Triton FP32 规约

本分支通过 CUDA Symmetric Memory，以 BF16 传输已能由 BF16 精确表示的本地
梯度值，再由 Triton 以 FP32 累加和输出。每次传输的每个坐标均检查位模式往返，
没有放宽原数值门槛。36 层 decoder 的大梯度规约使用新算子，其余通信仍走 NCCL。

| 同轮完整 8B 训练 | 顺序一，秒/步 | 逆序，秒/步 |
| --- | ---: | ---: |
| 原生 HF＋Flash，共有 V11 策略 | 5.296109 | 5.240922 |
| V12 Triton，词表分片 head＋融合 dW | 4.904509 | 4.902587 |
| HF＋Flash＋同一 peer 通信 | 4.974458 | 4.940806 |
| 完整 Triton＋peer 通信 | 4.514766 | 4.517596 |

相对 V12 的本轮增量为 **7.95% / 7.85%**；相对同轮原生 HF 为
**14.75% / 13.80%**，相对采用相同通信算法的 HF 为 **9.24% / 8.57%**。
HF 也可使用这一通信算子，不能将其称为 Triton 独占算法。
有效 allocated/reserved 峰值最高 **37.966 / 41.871 GiB/卡**，
包含外部缓冲的 0.5 GiB 预留，原总预算和验证 batch/样本均未减少。

八组完整运行及独立审查通过原 399 参数梯度、短程质量、全量验证、FP32 状态、
显存及 CUDA 覆盖门槛。负结果和旧验证 OOM 均保留；只推荐通过完整验收的
peer 方案。固定硬件、形状与软件版本，16 次质量更新不证明长期收敛，未测推理。
新入口显式启用，原默认配置不变。

详见 [实现、完整结果与复现](OPTIMIZATION_V13.md)、
[独立审查](results/qwen3_triton_v13_training_review_v1.json)、
[显式策略](results/qwen3_triton_v13_strategy_v1.json) 和
[证据索引](results/qwen3_triton_v13_evidence_index_v1.json)。

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
