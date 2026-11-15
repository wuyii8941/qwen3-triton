# Qwen3-8B：词表分片输出层与交叉熵

本轮目标是寻找需要改计算路径、不能直接由 HF/FSDP 配置开关得到的优化。
V11 的整步收益主要来自共有的参数保留策略，本轮继续保留该策略，改造输出层。
**当前模块实验及九步整模型筛选通过；正式完整 8B 对照尚未验收，不作正式整步收益声明。**

## 实现范围

新增 [词表分片输出层](qwen3_triton/vocab_parallel.py) 与
[局部 CE kernel](qwen3_triton/kernels/vocab_ce.py)。每卡保留本地词表行的 FP32
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

## 完整模型接入与验收计划

[训练入口](scripts/train_qwen3_vocab_v12.py) 使用官方 8B 全部 36 层、399 参数，
保留相同语料、BF16/FP32 策略和原 41.875 GiB allocator 预算。
[接入辅助](scripts/vocab_model_v12.py) 将 head 排除在 FSDP 外，并显式替换 labeled
loss 路径。默认不返回完整 logits；仅首步梯度诊断可显式交换完整 logits，计时不含该步。

第一次完整接入在首次 optimizer step 因 mixed Tensor/DTensor 参数组失败，
没有完成更新或计时。现将所有对照都分成 decoder、head 两组，native fused AdamW
及 lr、betas、eps、weight decay 不变。失败与修复记录保留在
`artifacts/triton_qwen3_v12/integration_fix_001.json`，不把该次计入性能。

当前先运行九次更新的可行性短测，包含完整 Triton 新 head、V11 策略控制以及
HF decoder＋定制 head；三组均已完成。新 head 为 4.909835 秒/步，Triton 控制为
5.228398 秒/步，短测耗时减少 6.09%；allocated 峰值为 37.769 / 40.178 GiB。
新 head 的九步损失相对控制 mean/max 漂移约 0.001262/0.003910，低于原门槛。
HF decoder＋定制 head 为 4.960445 秒/步，说明该机制可移植至 HF，不能宣称是
HF 无法实现的算法。短测不替代原生 HF 的正式对照。

额外测试减少重计算：同轮逐层重计算 4.909808 秒/步、隔层重计算 4.810395
秒/步、不重计算 4.937026 秒/步，三组九步损失均一致。隔层虽快约 2.02%，
没有达到事先固定的 3% 筛选门槛；不重计算略慢。因此仍选逐层重计算，保留
负结果与未达门槛的结果，不把“减少 FLOPs”直接当作整步加速。

[融合 dW kernel](qwen3_triton/kernels/vocab_dw.py) 另行实验：在同一 kernel 内
计算各 data rank 的 dW，逐份保留 BF16 舍入边界，再按顺序求 FP32 平均。
目标是省去四份 BF16 权重梯度的中间写回与独立求平均 kernel；当前正在模块筛选，
尚未接入上述整模型数字。投影与 CE 本身仍使用独立 kernel，不宣称单一 mega-kernel。

正式验收需各 rank 比较全部 399 参数梯度、16 次质量更新、初末 512 块完整验证、
实际 CUDA 模型/collective 覆盖以及正反顺序完整 optimizer-step 计时。
原入口与已发布 V9/V10/V11 结果均不修改。
