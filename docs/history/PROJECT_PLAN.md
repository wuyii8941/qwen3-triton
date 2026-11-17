# 架构与完成状态

目录已于 2026-11-10 拆分。本文的原始实现路径以 `experiments/legacy/` 为根；当前主训练与算子入口、新旧命名对照以 [目录导航](REPOSITORY_LAYOUT.md) 为准。

当前版本已完成从独立算子到真实模型、再到完整训练作业的验证闭环。这里是现状，不是待执行计划。

## 代码分层

| 目录 | 职责 |
|---|---|
| ops/reference/ | PyTorch 数学 reference、显式 backward |
| ops/triton/ | RMSNorm、RoPE、SwiGLU、attention、GEMM、cache 等实验 kernel |
| model/ | Tiny-Qwen、HF scoped 替换、静态 decode 原型 |
| training/ | 训练、官方 FSDP2、自研分片原型、loss、优化器、恢复 |
| benchmarks/ | 算子、完整 step、真实模型与数值门禁 |
| scripts/ | 串行实验监督、资源守卫、输入冻结和独立审计 |
| tests/ | 数值、契约、模型集成与实验报告反例测试 |
| results/ | 小型可分享证据；不是权重或原始 profiler trace |

## 已完成的验证层级

1. 独立数学 reference 和随机上游梯度检查。
2. HF-compatible Tiny-Qwen 权重映射与层级 parity。
3. 核心 Triton FP16/BF16、多 shape forward/backward。
4. 真实 HF Qwen3-1.7B 推理接入与误差、prefill/decode benchmark。
5. 官方 FSDP2 / 自研分片原型的多卡梯度、更新、导出与恢复。
6. 真实 8B 的冻结平台期训练、最佳恢复、全量 test。
7. 最终配置 profiler、正反顺序对照及未获益候选的反证。

## 默认与实验边界

成熟 GEMM、Flash SDPA、HF cache 和官方 FSDP 是可靠参考；手写 attention 或 GEMM 并不因此自动进入默认路径。HF scoped 推理替换不用于训练或并发服务。静态 decode、量化和通信扩展都有额外约束，必须显式启用。

Tiny-Qwen 中为学习保留的 extension/TODO 不等于默认模型不能运行；它们不是发布完成状态的阻塞项。数值、协议、版本和资源约束见 [复现指南](FSDP_REPRODUCIBILITY.md)，当前结论见 [优化总表](OPTIMIZATION_SUMMARY.md)。

不支持或未证明的能力包括：生产并发服务、通用任务质量等价、跨节点/弹性恢复、所有形状全局最优、全新环境的一键安装。不要把这些写成已有成果。
