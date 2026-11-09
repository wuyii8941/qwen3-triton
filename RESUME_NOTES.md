# 简历表述与证据

以下表述需结合自己实际完成和能解释的部分使用。不要把官方 FSDP、Flash SDPA 或 Torch 本地归约归为自研 Triton。

## 核心项目表述

**Qwen / Triton GPU Kernel 优化实践**

- 实现 Qwen RMSNorm、split-half RoPE、SwiGLU 的 Triton forward/backward，覆盖 FP16/BF16、多组 batch/sequence 和随机上游梯度，并建立 PyTorch 数学 reference 与真实 HF 接入的分层数值验证。
- 建立 eager、torch.compile、Triton 算子和模型 benchmark，区分编译/预热、steady-state、GPU 时间和主机可见延迟，分析 launch、布局复制与中间张量开销。
- 完成 scoped HF 推理替换、QK-Norm/RoPE 融合、GQA、静态 KV/CUDA Graph 等原型；保留未超过成熟实现或精度门槛未通过的结果，不将形状特例写成通用加速。

## 多卡训练扩展

- 基于官方 FSDP2 搭建四卡 A6000 上真实 DeepSeek-R1-0528-Qwen3-8B 全参数续训、验证平台停止、最佳恢复、全量 test 及断点恢复流程，使用 BF16 计算与 FP32 master/AdamW/归约。
- 同实际 41.875 GiB/卡预算下，native K36/every1 对实验性 Torch decoder exchange K32/every1，完整 WikiText 任务耗时 8408.48→7705.41 秒，观测减少 8.36%；记录 step 中位耗时减少 13.92%，allocated 峰值少约 1.44 GiB/卡。
- 另一个同 40 GiB、K34 的 bounded BF16 exchange / Triton 打包归约实验，完整任务耗时观测减少 6.40%，step 中位耗时减少 8.44%。这是独立配置，不能与上一项相加。

第一项完整任务 test loss 为 2.10615081→2.10707202，候选高 0.00092121；第二项为 2.10640750→2.10626447。均为单种子、共享硬件、停止步数不同的观察，不写质量无损、统计等价或孤立 kernel 因果收益。通信扩展依赖固定 PyTorch 私有接口，不是自研 FSDP 机制。

## 可独立描述的训练优化

Qwen3-1.7B、compiled HF + FlashOptim、BF16/no-master、B1/S512 下，共享 embedding/head 梯度生命周期优化两轮完整 step 耗时少约 5.5–6.6%，allocated 约 11.14→9.85 GiB。它不是上述 FP32 master 多卡协议，也不是完整训练作业时间的结论。

## 证据与不可扩大之处

- [完整结果](FSDP_8B_RESULTS.md)、[推理证据](INFERENCE_EVIDENCE.md)、[复现边界](FSDP_REPRODUCIBILITY.md)。
- results/fsdp_exchange_expandable_full_quality_final_v1.json。
- results/fsdp_scratch_bounded_quality_final_v1.json。
- results/fsdp_exchange_profiles_final_v1.json。
- results/fsdp_8b_resume_summary_v1.json。

已有恢复证明是同 world/拓扑/版本的两步重放，不是弹性恢复或 reader 加速。长程任务最佳权重未导出；早期部署导出是另一个已验证流程。最新 CPU 回归 714 passed / 311 GPU skipped，核心 GPU 契约 52 passed，不写全仓 GPU 验证全部通过。
