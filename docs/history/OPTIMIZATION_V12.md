# Qwen3-8B：词表分片输出层与交叉熵

本轮目标是寻找需要改计算路径、不能直接由 HF/FSDP 配置开关得到的优化。
V11 的整步收益主要来自共有的参数保留策略，本轮继续保留该策略，改造输出层。
**双顺序完整对照及独立审查已完成：词表分片 head 与融合版通过原门槛，
推荐融合版；辅助 HF 定制 head 的梯度检查失败，拒绝采用。**
原始全队列报告保留失败状态，逐方法审查同时要求该方法和同顺序两个基线
通过原全部检查，未调整任何容差。原入口与默认值不变。

## 完整训练结果

官方 Qwen3-8B，4×RTX A6000，每卡 batch=1、sequence=512；以下是完整
optimizer step 的中位数，两轮按正反顺序执行，编译和验证不计入。

| 实现 | 顺序一，秒/步 | 逆序，秒/步 | allocated 峰值，GiB/卡 |
| --- | ---: | ---: | ---: |
| 原生 HF＋Flash，共有 V11 训练策略 | 5.285977 | 5.249324 | 40.248 |
| V11 完整 Triton 控制 | 5.212518 | 5.208927 | 40.177 |
| Triton＋词表分片 head，未融合 dW | 4.909584 | 4.899512 | 37.769 |
| Triton＋词表分片 head，融合 dW | **4.902588** | **4.887280** | **37.468** |
| HF decoder＋定制 head | 梯度检查失败，拒绝 | 梯度检查失败，拒绝 | 不作性能结论 |

推荐版相对同轮原生 HF 耗时减少 **7.25% / 6.90%**，相对 V11 Triton
减少 **5.95% / 6.17%**。共有的保留参数、FSDP、重计算与 AdamW 策略相同，
这些数字来自新输出层路径及其融合的增量，不能再次计入 V11 的收益。
推荐版 reserved 峰值 41.807 GiB/卡；allocated 比同轮 HF 少约 2.780 GiB/卡。
这里均为计时训练区间的 PyTorch allocator 峰值，不代表 NVML 总显存。

融合相对未融合整步仅减少 **0.14% / 0.25%**，allocated 再少 **0.301 GiB/卡**。
两轮所有 27 次 global loss、初末验证值及记录的首步梯度比较指标与未融合版
完全一致；没有保存候选稠密梯度，不声称独立重放了全部候选梯度的逐位一致性。
单模块 56%–57% 的加速没有变成相同幅度的整步收益。

每组 27 次实际更新：16 次质量、3 次 warmup、8 次计时；计时取每步四 rank
的最大 wall time。每 rank 的全部 399 个 FP32 梯度在首步比较全部坐标，所有
master、梯度、AdamW moments 均为有限 FP32。原生 HF 控制首步梯度误差为零、
前向逐位一致。推荐版全局梯度相对 L2 为约 0.00691–0.00772，各参数通过
原 6%×参考范数＋1e-6 门槛，不宣称与 HF 逐位一致。

推荐版 16 次质量更新相对同轮 HF 的平均/最大 loss 漂移为
0.001626 / 0.003836，初末完整验证漂移约 0.000096 / 0.001934。
原门槛分别为 0.002 / 0.01 / 0.002；末次验证接近门槛，短程通过不证明长期收敛。
两次验证均覆盖全部 512 块、261632 个 shifted tokens。

实际 CUDA trace 验证全部模型 kernel 数量和 launch geometry，没有模型
BLAS/Flash 回退；FSDP/NCCL、分配及 native fused AdamW 仍为 PyTorch。
trace 是 27 次更新后额外一次 rank0 前向/反向，没有第 28 次 optimizer step，
不能用它代替完整训练计时或 optimizer profiling。双顺序不构成统计显著性证明。

证据：[逐方法审查](../../results/qwen3_triton_v12_method_review_v1.json)、
[原始全队列报告](../../results/qwen3_triton_v12_cohort_review_v1.json)、
[资源接纳审查](../../results/qwen3_triton_v12_resource_review_v1.json)、
[显式策略](../../results/qwen3_triton_v12_strategy_v1.json)、
[证据索引](../../results/qwen3_triton_v12_evidence_index_v1.json)。

## 实现范围

新增 [词表分片输出层](../../qwen3_triton/vocab_parallel.py) 与
[局部 CE kernel](../../qwen3_triton/kernels/vocab_ce.py)。每卡保留本地词表行的 FP32
master 与 AdamW 状态；交换各 data rank 的 BF16 hidden states 和标签，在本地
执行投影，使用 FP32 softmax 统计量规约得到完整词表的 loss。没有词表采样或量化。

反向只交换 FP32 hidden 梯度，不再聚合完整 LM head 权重或规约完整权重梯度。
每个 data rank 的 dW GEMM 仍输出 BF16，再在 FP32 中求全局平均；decoder 的
梯度保持原来的 FSDP 平均策略。dX 的词表分块累加会改变归约顺序，最终转为
BF16，必须通过原有全参数梯度与短程质量门槛，不能声称逐位一致。

模型计算使用 Triton；分配、布局、autograd、NCCL、decoder FSDP 与 fused AdamW
使用 PyTorch。新 head 已预先完成分片和梯度平均，必须排除在 root FSDP 之外。
训练标签、形状、调用顺序必须在各 rank 配合。这需要自定义 autograd、参数所有权
和模型 loss 路径。HF decoder 也可通过同样的定制接入，不宣称该算法只适用于 Triton。

原每卡投影形状是 `512×151936`，新路径是 `2048×37984`；每卡 head 的总乘加量
相同，完整词表与 global batch 保留。新路径改变数据流和通信对象，仍保存本地
BF16 logits；没有实现消除全部 logits 缓冲的分块/重计算 loss 算法。

假设 ring 算法，head 每卡单方向 tensor payload 的分析值从约 2.800 GB 降至
0.0378 GB；它不包含协议、staging、拷贝、诊断 logits 或 decoder 通信开销，
不是实测链路字节或带宽。实际 trace 的新路径为 42 次 AllGather、39 次
ReduceScatter、2 次 AllReduce；控制为 40/39。次数增加与消息体积减小同时发生。

算法参考：

- [Megatron-LM vocabulary-parallel CE](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/cross_entropy.py)。本轮 kernel 自行实现，没有复制其代码。
- [Liger fused linear CE](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py) 提供跨投影/loss 边界优化的相关思路；本轮不是复现其分块算法。

## 已完成的模块实验

官方 Qwen3-8B 的 `[151936,4096]` LM head 权重，4×RTX A6000，
每卡 `[1,512,4096]` hidden，BF16 计算、FP32 master 与梯度。参考为 native
FSDP2 Linear＋FP32 CE，保留前向权重，和 V11 相同。两模块同时驻留 GPU，
模块计时不含 optimizer；因此不能将其峰值或耗时当作完整训练数字。

三次 warmup、八次测量，正反顺序：参考约 0.380–0.382 秒，新 head 约
0.049–0.050 秒，耗时减少约 87%。四 rank 的 logits 逐位一致；loss 绝对误差
不超过 9.54e-7；dX 相对 L2 约 0.00125–0.00128；dW 约 1.03e-6–2.03e-6。
不同 rank 使用不同非单位上游梯度，也通过同样的 dX/dW 检查。全部坐标参与比较。

原始记录：`artifacts/triton_qwen3_v12/module_001`。固定源码、连续 120 秒空闲
admission 及全过程进程观察均通过。模块结果尚未代表真实 decoder hidden 分布、
整模型梯度、长期收敛或整步速度。

## 完整模型接入与前期筛选

[训练入口](../../scripts/train_qwen3_vocab_v12.py) 使用官方 8B 全部 36 层、399 参数，
保留相同语料、BF16/FP32 策略和原 41.875 GiB allocator 预算。
[接入辅助](../../scripts/vocab_model_v12.py) 将 head 排除在 FSDP 外，并显式替换 labeled
loss 路径。默认不返回完整 logits；仅首步梯度诊断可显式交换完整 logits，计时不含该步。

第一次完整接入在首次 optimizer step 因 mixed Tensor/DTensor 参数组失败，
没有完成更新或计时。现将所有对照都分成 decoder、head 两组，native fused AdamW
及 lr、betas、eps、weight decay 不变。失败与修复记录保留在
`artifacts/triton_qwen3_v12/integration_fix_001.json`，不把该次计入性能。

正式对照前运行九次更新的可行性短测，包含完整 Triton 新 head、V11 策略控制以及
HF decoder＋定制 head；三组均已完成。新 head 为 4.909835 秒/步，Triton 控制为
5.228398 秒/步，短测耗时减少 6.09%；allocated 峰值为 37.769 / 40.178 GiB。
新 head 的九步损失相对控制 mean/max 漂移约 0.001262/0.003910，低于原门槛。
HF decoder＋定制 head 也完成短测，但后续完整梯度检查失败，不采用其速度。
短测没有逐参数参考检查，不能替代正式对照；该机制可以定制移植至 HF，
不宣称是 HF 无法实现的算法。

额外测试减少重计算：同轮逐层重计算 4.909808 秒/步、隔层重计算 4.810395
秒/步、不重计算 4.937026 秒/步，三组九步损失均一致。隔层虽快约 2.02%，
没有达到事先固定的 3% 筛选门槛；不重计算略慢。因此仍选逐层重计算，保留
负结果与未达门槛的结果，不把“减少 FLOPs”直接当作整步加速。

[融合 dW kernel](../../qwen3_triton/kernels/vocab_dw.py) 另行实验：在同一 kernel 内
计算各 data rank 的 dW，逐份保留 BF16 舍入边界，再按顺序求 FP32 平均。
目标是省去四份 BF16 权重梯度的中间写回与独立求平均 kernel。单 GPU 模块实验中，
完整 `[37984,4096]` 本地权重梯度从约 19.4–19.5 ms 降至 8.4–8.5 ms，减少
56%–57%；四个候选配置的全部输出坐标均与原路径一致，选择 `(64,128,32,8,8,3)`。
此处含分配及原来的 deterministic fill，不把模块收益直接当作整步收益。
融合后的四卡 head 模块又独立通过官方权重、全坐标及不同上游梯度检查；
相对 native FSDP head 的前向＋反向为约 0.038 秒对 0.382 秒。

融合版已完成额外九步整模型短测，实际 trace 确认 `_vocab_dw_mean`、完整模型
kernel 数量及 launch geometry；没有模型 BLAS/Flash 回退。融合/未融合新 head 的
九步损失完全一致；融合版 4.900033 秒/步、未融合 4.908673 秒/步，同轮原 Triton
路径 5.270423 秒/步。allocated 峰值分别 37.466 / 37.769 / 40.178 GiB。
这些是四个计时样本的筛选结果，最终整步贡献以上述完整对照为准。
投影与 CE 本身仍使用独立 kernel，不宣称单一 mega-kernel。

正式验收已完成各 rank 全部 399 参数梯度比较、16 次质量更新、初末 512 块完整
验证、实际 CUDA 模型/collective 覆盖以及正反顺序完整 optimizer-step 计时。
原入口与已发布 V9/V10/V11 结果均不修改。

## 资源排队与记录

`formal_001` 在 GPU admission 期间因审查脚本短暂编辑被冻结校验拦截，未启动
训练；精确恢复原提交后保留该记录。`formal_002/native_r1` 完成了 27 次更新，
全参数首步数值检查通过，但监控观察到外部 GPU 进程，按原规则排除其性能。

`formal_003` 保持同一模型、数据、数值门槛和固定正反顺序，逐组接纳第一个
完整且全过程观察干净的尝试；资源干扰时只终止自己的 torchrun，并重新等待
连续 120 秒四卡空闲。所有尝试的日志、监控和哈希均保留，成功干净的尝试
不得因为较慢而跳过。干净的技术错误直接停止，不按数值或速度筛选重复结果。
CPU 正负检查已验证拒绝接纳受干扰尝试，也拒绝跳过干净尝试。

独立审查已核对完整数值记录、原生参考档案、全量验证、实际 trace 及接纳顺序。
十个角色共十二次尝试，其中两次受干扰的尝试被排除，首个完整且观察干净的
尝试才被接纳；没有按计时挑选重复结果。CPU 复核验证参考档案及已记录的误差
运算，候选稠密梯度未另存档，不能声称重新计算了候选梯度。

HF decoder＋定制 head 的两轮完整运行均有一项未过门槛的梯度：rank1 的
`model.layers.8.self_attn.k_norm.weight` 相对误差约 7.93%，超过原 6%＋绝对项门槛。
该配置拒绝采用，不用其速度支持性能结论。保留原始全队列报告的失败状态，
另外逐方法审查推荐的 Triton 配置及同顺序 native HF/V11 控制；各方法均沿用
原全部 399 参数、短程质量、显存、CUDA 及双顺序至少 3% 收益门槛，容差不变。

## 复现与显式启用

先完成 [环境、官方权重、语料和原生梯度参考准备](../../QWEN3_TRITON_REPRODUCE.md)。
使用 [训练入口](../../scripts/train_qwen3_vocab_v12.py)，保留原输入、输出、全量验证及
梯度参考参数，并追加：

```text
--variant triton_optimized
--vocab-ce
--fused-vocab-dw
--keep-full-layers 36
--checkpoint-every 1
--skip-embedding-backward-gather
--gemm-plan results/qwen3_triton_v10_gemm_plan_001.json
```

完整协议另需 `--quality-steps 16 --warmup-steps 3 --measure-steps 8
--full-validation --trace --reference-raw-dir REFERENCE_DIR`。
新 head 不使用 `--retain-lm-head`，它已自行分片、平均梯度并排除在 FSDP 外。
native HF/V11 控制省略两个 vocab 标志，使用 `--retain-lm-head`；HF 同时省略
GEMM 计划并选择 `--variant hf`。策略 JSON 是描述文件，不是 CLI 加载器。

固定数学源码见队列 manifest；资源队列与 CPU 审查源码单独列哈希。
[完整队列](../../scripts/run_vocab_resource_queue_v12.py)、
[数值审查](../../scripts/review_vocab_full_v12.py)、
[逐方法审查](../../scripts/review_qwen3_vocab_methods.py) 和
[小型证据收集](../../scripts/bundle_qwen3_vocab_evidence.py) 均已提交。
原始本机 artifacts、权重、语料、稠密梯度和 profiler trace 不随 Git 分发。

## 下一轮优先方向：按瓶颈选择

以下尚未实现或验收，不计入本轮收益。分析日期为 2026-11-15；本轮软件环境仍
固定 Torch 2.10.0、Triton 3.6.0、Transformers 4.51.0，不能代表最新版 HF 生态。

已有 [rank0 前向/反向 trace 汇总](../../results/qwen3_triton_v12_forward_backward_profile_v1.json)
显示 FP32 ReduceScatter 39 次累计 2.810 秒，AllGather 42 次累计 1.482 秒，
通用 GEMM 866 次累计 0.272 秒。通信 kernel 时间可能包含其他 rank 的等待，
各类别可以重叠，不是整步占比或可消除时间。机器拓扑为 0/1、2/3 各经 PCIe
switch，跨组为 SYS，没有观察到 NVLink。先优化通信与数据搬运比继续穷举 GEMM
tile 更值得验证；仍需四 rank 含 optimizer 的完整 trace 才能归因关键路径。

1. **训练通信和计算重叠。** 先记录真实 transport、消息大小、stream 等待及跨
   rank 不均衡，再调整预取/桶粒度和通信调度，保留 FP32 梯度规约。
   [PyTorch 2.13](https://pytorch.org/blog/pytorch-2-13-release-blog/) 新增独立
   ReduceScatter process group 以重叠 AG/RS，值得隔离环境测试；它也是 HF
   可用的公共优化，两方须同时启用后再比较。不能仅凭 trace 名称为 RING_LL
   就断定协议选错；[NCCL 官方说明](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
   不建议长期强设协议，任何诊断实验都需检查数据和实际端到端效果。
2. **融合布局、类型转换和梯度写入。** 当前有 38 次 chunk/cast、39 次 split
   拷贝；可探索 Triton 直接写入连续的 FP32 通信桶，减少中间 BF16 梯度、转换
   和再次读取，并保持原 BF16 舍入及 FP32 平均语义。还可合并 Q/K/V 的小 GEMM，
   但需解决 stride、反向以及保持官方 399 参数/权重映射，不能只测拼接后的 GEMM。
3. **消除已证明冗余的输出初始化。** trace 中 BF16/FP32 Fill kernel 累计约
   0.188 秒；当前 deterministic fill=True。对每个输出须先证明所有后续读取
   的元素都已写入，才能探索安全复用缓冲。不能直接关闭保护并声称数值等价；
   公共分配策略优化也须给 HF 同样条件。
4. **分块 projection＋CE。** 当前仍写本地 logits，下一版可边计算边规约、反向
   重计算分块以降低显存，并尝试用节约的显存增加 microbatch。需比较固定有效
   batch 的 tokens/s，防止仅增大 batch 造成表面提速。
   [Liger 实现](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py)
   和上述 PyTorch 2.13 的 LinearCrossEntropyLoss 是相关基线；不能声称 HF
   生态没有 fused loss。已有减少整层重计算的短测未达 3% 门槛，暂不采用。
5. **推理单独建立基准。** 先补正确的增量 KV cache、最后 token 投影、decode
   专用 GQA attention，再做 CUDA Graph/静态缓冲、分页 KV 和连续批处理。
   A6000 上再比较 BF16 与 W4A16/AWQ/GPTQ 的真实吞吐和精度；量化不算本轮
   BF16 训练的同精度优化。低并发长生成再测 draft/EAGLE 投机解码，报告接受率、
   TTFT、TPOT 和高并发吞吐，而非直接移用训练收益。
   [vLLM 官方文档](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
   明确效果取决于负载与硬件；原版 Qwen3-8B 不应假定带原生 MTP head。

FlashAttention-4 是 2026 年值得关注的实现，但
[官方仓库](https://github.com/Dao-AILab/flash-attention) 说明针对 Hopper/Blackwell，
不能把其性能数字套到本机 SM86 A6000。FP8/FP4 同样不能假设有本机原生硬件优势。
本轮没有实现推理/KV cache，没有推理提速或长期收敛结论。
