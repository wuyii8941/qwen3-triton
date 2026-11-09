# 优化结论与取舍

本页只列当前决定。详细数值使用测量时的环境和工作负载，不跨阶段相加。

## 推理与算子

| 方向 | 当前结论 |
|---|---|
| RMSNorm / RoPE / SwiGLU | 已实现 reference 与 Triton，完成 FP16/BF16 前后向契约；真实 HF 路径另保留低精度 cast 边界 |
| QK-Norm + RoPE | 融合减少 launch 和布局复制；纯数学 kernel 与 HF-rounded kernel 不能混为逐位等价 |
| GQA | 实验路径不 materialize 重复 K/V；真实模型精度门禁仍必须逐 dtype 验证 |
| 自写 attention | 已做 tiling、online softmax、GQA 与 backward；未普遍超过 Flash SDPA，成熟实现保留 |
| GEMM + SwiGLU | 已做 forward/backward 与 tile 实验；融合不保证训练 step 变快 |
| 静态 KV / CUDA Graph | 有固定 batch、连续无 padding decode 的原型；私有 API 与固定版本约束保留 |
| W8A16 / 稀疏算法 | 近似路径有自然文本误差反例或形状依赖，不默认启用 |
| kernel autotune | 最佳配置仅对已测 shape 有证据，不称泛化最优 |

算子 eager/compile/Triton 的测量口径与真实 HF 接口分别见 [推理证据](INFERENCE_EVIDENCE.md)、[HF 接入](HF_INFERENCE.md)。

## 训练

1.7B compiled HF + FlashOptim 的共享 embedding/head 梯度生命周期优化，两轮完整 step 耗时减少约 5.5–6.6%，allocated 约 11.14→9.85 GiB。该结果是 BF16/no-master、B1/S512，不是 FP32 master FSDP 或新增完整收敛作业。Graph、optimizer launch 合并和 loss/GEMM fusion 没有因此全部叠加默认。

多卡主要结果见 [8B 证据](FSDP_8B_RESULTS.md)：

| 配置 | 完整任务结果 |
|---|---|
| native K36 对 Torch decoder K32，同实际 cap 41.875 GiB | 总耗时少 8.36%，step 中位耗时少 13.92%；test loss 高 0.00092121 |
| 同 40 GiB、K34 的 bounded/Triton decoder | 总耗时少 6.40%，step 中位耗时少 8.44%；模型 allocated 峰值相同 |

两者均完成冻结任务平台期、最佳参数恢复及全量 test；单种子共享硬件，不是统计质量等价、单 kernel 因果加速或原始预训练收敛。

## 最后两项增量实验：不采用为速度默认

- K36→K24：native 两轮慢 6.35% / 11.53%。decoder K32→K24 首轮快 1.10%，反向慢 3.03%。重试/同步更少，但没有稳定 step 收益。
- 固定 K32，仅将本地归约从 Torch 换为 Triton：首轮快 1.72%，反向慢 3.05%；allocated 峰值及每步重试/同步不变。
- K24 的 allocated 虽少约 2.88 GiB/卡，reserved 没有下降，不能宣称释放等量物理显存或已经验证更低预算。

证据：results/fsdp_retention_pressure_final_v1.json、results/fsdp_decoder_backend_final_v1.json。实验通过数值/记录审计不等于通过速度采纳门槛；这些配置没有追加长程质量训练。

## 推荐

默认使用官方路径。完整质量支持的通信扩展保留为固定版本、固定协议下的研究选项。不要把所有 Triton 开关同时打开，也不要把不同时段/预算/环境的 speedup 相加。

本轮优化和增量反证已完成，没有活跃的本轮长训练队列。结论是已测配置中的工程选择，不是跨硬件全局最优证明。
