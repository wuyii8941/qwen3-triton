# Qwen3 / Triton GPU Kernel Lab

基于真实 Qwen3 架构模型的算子、推理和多卡训练优化实验。当前文档对应 2026-11-09 的已验证状态，不包含阶段性排队日志。

本项目实现 PyTorch reference、Triton forward/backward、Tiny-Qwen、真实 HF 推理接入，以及官方 FSDP2 和自研分片原型。不是所有 kernel 都比成熟库快；默认保持可验证的原生路径，实验选项显式开启。

## 已完成的工作

- RMSNorm、Qwen split-half RoPE、SwiGLU 和 QK-Norm+RoPE 融合；支持低精度数值验证与随机上游梯度检查。
- attention 分块、GQA、prefill/decode、GEMM fusion、静态 KV cache 和 CUDA Graph 实验；保留未加速或未通过精度门槛的结果。
- Qwen3-1.7B 的 HF 推理替换和训练优化；对比 eager、torch.compile 与 Triton，区分单算子和完整模型。
- 真实 DeepSeek-R1-0528-Qwen3-8B 的四卡全参数续训、最佳参数恢复、全量 test、HF 导出/重新加载，以及同 world size 的训练状态恢复。
- profiler、正反顺序性能对照、资源守卫与独立原始记录审计。

## 主要结果

| 工作负载 | 已观察结果 | 限制 |
|---|---|---|
| 8B，四卡 A6000，S512/B1，实际 cap 41.875 GiB/卡 | 官方 FSDP2 native K36/every1 对 Torch decoder exchange K32/every1：完整任务耗时 8408.48→7705.41 秒，少 8.36%；记录 step 中位耗时少 13.92% | test loss 2.10615081→2.10707202；单种子共享硬件，不能称质量无损或单 kernel 因果加速 |
| 8B，40 GiB/卡，K34/every1 | bounded decoder / Triton 通信扩展完整任务耗时少 6.40%，step 中位耗时少 8.44% | 独立配置，allocated 峰值相同；不能与上一行叠加 |
| 1.7B，compiled HF + FlashOptim，BF16/no-master | 共享 embedding/head 梯度生命周期优化，两轮完整 step 少约 5.5–6.6%，allocated 少约 1.29 GiB | 不是 FP32 master FSDP 协议，也不是收敛时间结论 |
| 最终配置后的两项增量尝试 | 降低驻留到 K24、本地归约换 Triton 均未证明稳定额外 step 收益 | 不切换默认，不隐藏负结果 |

K 表示 forward 后保留完整参数的最后 K 个 decoder 单元；every1 表示每层 activation checkpoint。这里的“完整任务”指冻结 WikiText 续训协议达到验证平台期，不是原始预训练收敛。

8B 的准确 checkpoint 名称是 DeepSeek-R1-0528-Qwen3-8B，不是原版 Qwen3-8B 或 7B。候选使用官方 FSDP2 的实验性通信扩展，不称自研 FSDP 胜过官方。

## 使用与文档

- [架构与完成状态](PROJECT_PLAN.md)
- [优化结论与取舍](OPTIMIZATION_SUMMARY.md)
- [真实 HF 推理接口](HF_INFERENCE.md)
- [算子和推理性能口径](INFERENCE_EVIDENCE.md)
- [分片原型、导出与部署](MINIMAL_FSDP.md)
- [8B 完整结果与证据](FSDP_8B_RESULTS.md)
- [环境、资源限制和复现入口](FSDP_REPRODUCIBILITY.md)
- [训练复现流程](TRAINING_REPLICATION_PLAN.md)
- [checkpoint 保存与恢复](CHECKPOINT_RESUME.md)
- [可用于简历的表述](RESUME_NOTES.md)

## 环境与测试

当前多卡研究环境：RTX A6000、Python 3.11.16、PyTorch 2.10.0+cu128、Triton 3.6.0、Transformers 4.51.0。早期推理/私有 Flash varlen 实验使用 Torch 2.5.1 / Triton 3.1.0，结果不能跨环境直接混算。pyproject.toml 不是完整 GPU 栈锁文件。

本机研究入口使用已配置的 .venv-research，不安装依赖或修改系统 Python；缓存和临时文件放到项目 .runtime-research。克隆仓库需要自行准备独立环境、模型和数据，不能直接使用本机路径。

~~~bash
# CPU；Gloo 测试需要可用的本地回环网络
CUDA_VISIBLE_DEVICES='' GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  bash scripts/research_python.sh -m pytest -q tests

# GPU 空闲时单独验证核心 kernel
CUDA_VISIBLE_DEVICES=0 bash scripts/research_python.sh -m pytest -q \
  tests/test_operator_contract_matrix.py tests/test_rms_norm.py \
  tests/test_rope.py tests/test_silu_and_mul_triton.py
~~~

最终 CPU 回归：714 passed / 311 GPU skipped；独立核心 GPU 验证：52 passed，使用单卡 4 GiB allocator cap。这不是全仓 GPU 测试全部通过的声明。

## 发布范围

发布源码、测试、配置及 results/ 中的小型最终证据。submission/、artifacts/、模型、语料、checkpoint、环境、缓存、密钥不发布。文档里的 artifacts 路径指本机原始记录，克隆仓库后并不存在；只有保留这些记录才能重新执行独立审计。

保留的历史实现用于基线、反例和审计依赖，不代表推荐启用。过期阶段文档与中间结果已从发布树移除，当前说明以上述指南为准。
