# Qwen 优化总表：做了什么、什么有效、什么不该启用

## 最新训练侧筛选（2026-11-05，独立研究环境）

下方历史表仍对应Torch2.5推理环境；本轮训练使用Torch2.10/Triton3.6、HF4.51、
FlashOptim0.1.4、Qwen3-1.7B BF16/A6000，不能混用旧倍率。
完整说明与接入命令：[训练优化筛选](TRAINING_REPLICATION_PLAN.md#最终强基线复核与推荐以本段为准)。

| 方案 | 当前决定 |
|---|---|
| 共享embedding/head梯度合并 + 消费临时proxy | B1/S512两轮完整step耗时少5.5–6.6%，peak allocated少约1.29 GiB；已接入可选训练模式 |
| 普通embedding合并的shape矩阵 | 4组shape×4路径×2轮；S2048有显存回退，不普遍启用 |
| 小张量FlashOptim launch合并 | 310→199次，保持逐参数量化/状态；仅部分小shape有效，不盲目叠加 |
| 全参数persistent、对齐、异步metadata | 正确性/汇编/性能都做过，无稳定额外step收益，留作实验 |
| 训练F/B CUDA Graph | 完整梯度与恢复通过；去掉额外workspace限制后，S512额外收益不足1%，S2048无收益，不默认推荐 |
| 原生SDPA GQA去复制 | 两版均未过BF16完整梯度门槛，不接入训练 |
| MLP dX addmm | 无加速，不采用 |

最终目标回归 **67 passed**；三组256步变动数据/validation短轨迹均保留证据，
并非新的收敛结论。没有改默认训练、没有新建大模型checkpoint或修改系统Python。
统一证据：`artifacts/optimization_wave_final_v1.json`。

## 历史推理与训练工作

范围：本地 Qwen3-1.7B、RTX A6000、PyTorch 2.5.1、Triton 3.1.0、Transformers 4.51.0。
当前完整回归：**278 passed，2 warnings**。最新完整训练结果见
[收敛作业比较](TRAINING_CONVERGENCE_RESULTS.md)；早期论文实验见 [论文算法可行性筛选](RESEARCH_OPTIMIZATIONS.md#6-论文算法可行性实验2026-11-01fasa)。
此前有效优化的报告仍为 [LM head / compile / 混合路径](LM_HEAD_COMPILE.md)，本轮失败候选没有替换原默认链路。
真实模型主验收使用 FP16。以下不是不同阶段 speedup 的累加表：算子 GPU 时间、
随机 block 时间和完整模型 wall-time 必须分开看；历史数据对应各自 JSON 的源码指纹。

## 一张总表

| # | 优化工作 | 解决的问题 / 具体做法 | 当前证据与结论 | 接口或证据位置 |
|---:|---|---|---|---|
| 1 | Triton RMSNorm | FP32 reduction，融合平方/均值/rsqrt/缩放 | 已接入 HF；独立算子 BF16 指定 shape 与 compile 基本持平，明显优于 eager | `model/hf_inference.py::_norm`；`INFERENCE_EVIDENCE.md` |
| 2 | Triton split-half RoPE | 一次 kernel 完成前后半维配对旋转 | 已接入；不能声称单算子普遍胜过 compile | `_rotary`；`INFERENCE_EVIDENCE.md` |
| 3 | Triton SwiGLU activation | 融合 SiLU 与乘 up，消除激活中间张量 | 已接入；某些 shape 与 Inductor 相当 | `_activation`；`artifacts/inference_evidence.json` |
| 4 | HF 舍入语义对齐 | 保留 RMSNorm、RoPE、SiLU 的低精度 cast 边界 | 数值基础工作，不独立包装为性能收益；BF16 仍有完整模型反例 | `HF_INFERENCE.md` |
| 5 | Q/K 布局去复制 | 用 stride 读取 projection view，不先转连续 BHSD 再转回 | 独立 Q 分支 BF16 示例 55.4 → 29.1 μs，peak delta 12 → 4 MiB | `artifacts/qk_layout_ablation.json` |
| 6 | Q/K norm + RoPE 单次 launch | 合并四次前处理；Q/K head 数可不同 | 与 GQA 去复制一起形成已验证端到端收益；不要把组合收益全归给此项 | `fuse_qk=True`；`E2E_ATTENTION_OPTIMIZATION.md` |
| 7 | 原生 compact GQA | `enable_gqa=True`，不 materialize 扩展 K/V | 前处理组合让旧 eager decode 再下降约 25–32%；保留原生 Flash | `compact_gqa=True`；同上 |
| 8 | Decode gate/up GEMV + SwiGLU | 针对小 M 融合两次投影与激活 | 当前 B≤2 decode 基础路径；Graph 下重新对比 cuBLAS，仍保留 | `decode_mlp=True`；`decode_gate_up.py` |
| 9 | Fused GEMM + SwiGLU prefill | Tensor Core tile、warp/stage 调参，HF rounding | 接入真实 prefill；S=512 有收益，短 prompt 不稳定 | `prefill_mlp=True`；`STATIC_DECODE_OPTIMIZATION.md` |
| 10 | GEMM 分组 program 顺序 | 尝试 GROUP_M=1/4/8 改变局部性 | 指定扫描无收益，不默认启用 | `KERNEL_TUNING_RESULTS.md` |
| 11 | 多行 RMSNorm / RoPE tile 扫描 | 每 program 处理多行，减少小行调度 | Q/K norm 算子有提升，完整 HF 路径没有稳定收益 | `multirow_norm=True`；同上 |
| 12 | Packed QKV | 三个 Linear 合为一个，split 保留 view | Graph 基础上额外约 2–4% pipeline 收益；多占 448 MiB 快照 | `packed_qkv=True`；`STATIC_DECODE_OPTIMIZATION.md` |
| 13 | Static contiguous KV cache | 预分配 BSHD，用 GPU 有效长度避免 cat 和容量 mask | 单独使用未加速；是当前 Graph 路径的前提 | `StaticDecodeCache` |
| 14 | 完整 decode CUDA Graph | 固定地址捕获模型 forward + LM head；动态更新 token/position | 主收益来源；配合 packing/K 写入，两轮 pipeline 比旧优化 eager 快约 2.4–2.8× | `StaticDecodeRunner.capture/decode`；同上 |
| 15 | RoPE 后 K 直接写 cache | K 分支输出直达 cache，不再单独复制 | 每 step 少 28 个复制 kernel；Graph 基础上约 0.7–1% pipeline 收益 | `direct_k_cache=True` |
| 16 | V 写入并入 QK kernel | K program 同时搬运对应 V，V 不做 norm/RoPE | 再少 28 个复制 kernel；新增四配置实验约 0.7–1.4% pipeline 收益 | `direct_v_cache=True`；`DECODE_REFINEMENTS.md` |
| 17 | Residual add + RMSNorm | 合并 attention residual 加法与后置归一化 | 少 28 个 kernel；单项 wall-time 很小、部分 CI 跨 1，组合单独测 | `fuse_residual=True`；同上 |
| 18 | Decode MLP W8A16 | 只量化 gate/up 权重，激活仍 FP16；G=64/32/16 | 近似算法，不是免费数值等价；G64/G32 均发现超过原 2% 门槛的输入，不推荐 | `decode_int8_group`；同上 |
| 19 | Grouped scale 复用 | 显式加载每组 scale，广播后使用二维 reduction | 解决朴素 W8 第一版的严重退化；B1 算子例约 43 μs，FP16 对照约 75 μs | `decode_int8_impl='grouped'`；`weight_only_tuning.json` |
| 20 | W8A16 Tensor Core 尝试 | 反量化 tile 后用 tl.dot，扫描 tile/warp/stage | 有配置超过 A6000 shared-memory 上限，明确拒绝；可运行配置未胜过 grouped GEMV | `ops/triton/weight_only_tuning.py` |
| 21 | 按 batch 分流精度实现 | B1 可选 G16；B≥2 回到 FP16 gate/up | 避免把 B1 的算子结果推广到 B2；默认仍不量化 | `decode_int8_max_batch=1`；同上 |
| 22 | 自写 tiled prefill attention | BLOCK_M×BLOCK_N、online softmax、autotune、紧凑 GQA | 独立学习/测试路径；实际 HF 推理继续使用更有竞争力的 Flash | `ops/triton/attention.py` |
| 23 | Split-KV decode + GQA 共享 | 多 CTA 并行读长 cache，用 LSE 合并 partial | 改善旧自写 decode，长 KV 最佳约 4.9×；仍未超过原生 Flash，不接入 HF 默认 | `KERNEL_TUNING_RESULTS.md` |
| 24 | Attention backward 合并 dK/dV、片上累计 dQ | 复用概率等中间量，避免 dQ partial workspace | 独立训练实验；不算作本报告推理端到端收益 | `attention_backward_triton_fused_kv`、相关测试 |
| 25 | 原权重 Triton LM head GEMV | 小 M / 大词表投影，扫描输出 tile 与 warps | 连续输入与原生 Linear 基本持平；不称为显著 kernel 加速 | `linear_decode.py`；`lm_head_tuning_shared.json` |
| 26 | LM head W8A16 | 分组 INT8 权重、FP32 scale/累加，保留 tied 原权重 | 第一轮相对既有 Graph pipeline 约降低 4.3–6.7%；属于近似量化，额外快照 315.3 MiB | `lm_head_impl='int8'`；`LM_HEAD_COMPILE.md` |
| 27 | LM head 跨 batch 复用权重 | 同一 program 内反量化一次，处理多个 batch 行 | G64 B2 算子 937→512 μs，B4 1866→732 μs；避免朴素 W8 多次读权重退化 | `share_batch=True`；两份 `lm_head_tuning*.json` |
| 28 | Last-token LM head 连续化 | 对 `[B,1,H]` 的非连续 view 做小复制，使原生投影从 bmm 转为 mm | 不量化；B4 算子含复制约 3861→893 μs，pipeline 约降 2.5%；B1 无收益 | `lm_head_impl='contiguous'`；`lm_head_layout.json`、`static_decode_e2e_lm_head_v2.json` |
| 29 | 明确 custom-op mutation 契约 | 绕开 pinned Torch 的 Triton launch 参数误判，注册 fake 和准确 cache 写入 schema | fullgraph 无回退；保留舍入与 no-FMA；本身不是加速保证 | `ops/compile_ops.py`；`tests/test_compile_ops.py` |
| 30 | 整模型 compile 强基线 | 相同静态 cache/Flash/Graph，编译 prefill 和 decode，记录重编译次数 | B1/S128 与现有 Triton pipeline 基本持平；S512 Triton 更快；自写算子再 compile 无收益 | `compile_pipeline_custom_ops.json`；`LM_HEAD_COMPILE.md` |
| 31 | 分阶段混合执行 | compiled PyTorch prefill + Triton Graph decode，原参数与 KV cache 共享 | 不量化；B1/S128 完整流程再降约 3.8%，S512 反而变慢约 3.2%，保持可选 | `HybridDecodeRunner`；`compile_pipeline_hybrid.json` |
| 32 | FASA 风格频率选择 + indexed attention | validation 校准 split-half FC，低维打分/top-k，直接索引 KV；独立 test split 128 步验证 | 8K/16K 变慢；32K F16/K512 pipeline 快 7.45% 但 top-1 仅 75.78%、最大 logits L2 142.28%，拒绝采用；不计作有效加速 | `FrequencySparseRunner`；`frequency_sparse_pilot.json`、`frequency_sparse_long.json` |
| 33 | BLASST 风格 running-max 块跳过 | 每个 split 的局部阈值跳过 V/softmax/PV，K/QK 仍计算；有零阈值 dense 对照 | 小阈值通过局部数值初筛；8K pipeline 仅快 0.71%，32K 慢 0.85%，未达采用门槛，默认保持 Flash | `BlockSkipRunner`；`block_skip_screen.json`、`block_skip_e2e_confirm.json` |
| 34 | Triton AdamW rounded 与完整训练验证 | 融合 optimizer 更新，检查低精度舍入；保持原生 HF 前后向 | A6000/B1/S512/BF16 WikiText 持续预训练，两组平台停止；同质量耗时下降约18%，各自完整作业下降约7.7%；单顺序作业对，不外推生产 | `training/triton_adamw.py`；`TRAINING_CONVERGENCE_RESULTS.md`；`convergence_comparison_v1/verified_result.json` |

上述开关均为可选实验，不会默认替换用户的训练路径或覆盖 checkpoint。
“已接入”不等于“所有形状都更快”，更不等于“全面超越 torch.compile / vLLM”。

## 证据应该怎样读

1. **算子级**：`inference_evidence.json`、`kernel_tuning_*.json`、`weight_only_tuning.json`。
   固定张量的 Graph GPU 时间，不能直接代替完整模型 wall-time。
2. **早期真实 HF 路径**：`hf_attention_e2e_float16*.json`。
   前处理/GQA 消融，单 token decode，不是 16 步 pipeline。
3. **Graph pipeline**：`static_decode_e2e*.json`。
   完整 prefill + 16 次连续 decode forward，包含 LM head、argmax、cache 更新和同步，
   排除初始化、packing、JIT、capture，成本另报。
4. **本轮进一步优化**：`static_decode_e2e_refine.json`、`static_decode_e2e_refine_v2.json`。
   包括性能负结果与数值门槛失败，不应只筛出最快行。

所有真实模型的数值门槛保持 relative L2 < 2%，并另报 argmax 和自然文本 smoke。
随机 logits 检查与单条自然文本不能证明任务质量无损，量化尤其需要额外质量评估。
固定 batch、等长无 padding、非并发、固定版本的原型；不宣称生产服务吞吐或普适最优。

## 上一轮端到端结果

最终分流实验 seed=20261004，16 个交错 round；包含完整 prefill + 16 步 decode，单位 ms。
“此前 Graph”是已包含上一阶段优化的基线，不是未优化 HF。

| Batch / prompt | 此前 Graph | 新组合，不量化 | 新组合，B1 可选 G16 | 量化是否实际执行 |
|---|---:|---:|---:|---|
| 1 / 128 | 110.783 | 109.088 | 98.376 | 是 |
| 2 / 128 | 118.628 | 116.296 | 116.417 | 否，FP16 |
| 1 / 512 | 116.173 | 114.256 | 103.755 | 是 |
| 1 / 2048 | 171.828 | 169.996 | 161.331 | 是 |
| 4 / 128 | 125.292 | 123.107 | 123.099 | 否，FP16 |

新无量化融合每步少 56 个 kernel，实测 logits 与此前 Graph 路径逐位一致。
可选 G16 分流为近似路径，最后一轮最大 logits L2 约 0.496%，此前 G16 输入中出现过约 1.375%。
G64/G32 存在超 2% 的失败输入，因此未用于最终分流。
该阶段完整测试 **150 passed**；自然文本 smoke 一致，不代替广泛质量评测。

阶段证据：`artifacts/static_decode_e2e_dispatch.json`，源码指纹对应当时实现；当前已新增后续实验。
调用示例、显存成本和复现命令见 [进一步优化说明](DECODE_REFINEMENTS.md)。

## 本轮新增结果与边界

LM head、输入布局和完整 compile 的实测及复现见 [LM head / compile 报告](LM_HEAD_COMPILE.md)。
第一轮相对既有 Graph，LM head G64 在五组配置的完整流程下降约 4.3–6.7%；
B1 叠加旧 MLP G16 后最大约 16.5%，但自然文本续写出现分歧，不能写成无损优化。
无量化方案优先保留 `lm_head_impl='contiguous'` 的布局消融，以及 compile 强基线的对照。
原 FP16 自写 LM head 与连续输入 cuBLAS 基本持平的负结果也保留。
第二个随机种子五组复测：LM head G64 仍有约 4.0–6.6% 完整流程收益；连续化在
B2/B4 有收益，所测 logits 相对当前 Graph 基线最大 L2 约 0.00365%，argmax/smoke 一致。
量化组合的自然文本分歧复现。
混合路径在 B1/S128：108.911→104.808 ms，相对 compiled PyTorch 的 109.103 ms 也更快。
S512：113.976→117.597 ms，负结果保留。因此没有把混合路径默认用于所有输入长度。

## 下一步的可选研究方向

更大 batch、自然文本集的量化 NLL/任务质量验证、融合后的 CUDA Graph 生成循环
还有研究价值。它们不能写成现有成果。LM head 带宽和整模型 compile 已有本轮实测，
后续应以这些强基线继续衡量，而不是反复对最早 eager 基线报告巨大倍率。

2026-11-01 的论文算法筛选已完成，见 [研究实验第 6–8 节](RESEARCH_OPTIMIZATIONS.md#6-论文算法可行性实验2026-11-01fasa)。
这次采用本地 WikiText-103 的 validation/test 分离、8K/16K/32K prompt 和 128 步 decode；
两个候选都未默认启用，保留可复现负结果。不是 LongBench 或通用任务质量结论。
