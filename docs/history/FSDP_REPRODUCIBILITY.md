# 环境、资源与复现边界

本文描述历史结果的冻结环境与门禁。旧源码/配置相对路径以 `experiments/legacy/` 为根；命令在仓库根目录通过 `scripts/legacy_python.sh` 启动。当前维护入口见 [目录导航](REPOSITORY_LAYOUT.md)。

## 环境

当前FSDP研究：Python3.11.16、PyTorch2.10.0+cu128、Triton3.6.0、Transformers4.51.0、RTX A6000。版本观测见 results/fsdp_runtime_packages_v1.json。早期推理实验为Torch2.5.1/Triton3.1.0，不能直接混用私有API或性能数字。

本机 .venv-research 的 .pth 引用了现有环境的只读site-packages，没有修改母环境或系统Python；这不是已独立锁定的全新安装环境。pyproject.toml及版本观测JSON也不是完整transitive lock。

scripts/research_python.sh 使用项目解释器，并把HF、Triton、Inductor、CUDA和扩展缓存、临时文件放到 .runtime-research。它不安装依赖。新机器需自行配置独立Conda/venv和匹配GPU软件栈。

## 模型、数据与协议

- 实际8B是DeepSeek-R1-0528-Qwen3-8B；1.7B实验使用Qwen/Qwen3-1.7B。
- 权重、语料、环境、checkpoint与原始artifacts不发布。
- benchmark的模型路径可通过其CLI调整；长程任务从 training/fsdp_convergence_protocol_v1.json 读取本机模型/数据路径。移植需创建显式配置副本，不改冻结原件后仍声称哈希不变。
- results/frozen_fsdp_reference_v1.json记录已有本机输入内容身份；不是自动下载器。原协议的 preregistered_not_executed 是冻结时历史标签，执行状态以原始作业和独立审计为准。

## 放行顺序

先核对版本、源码和输入，再做D128 tied/untied梯度/更新门禁、真实post-update全量验证预检、匹配性能对照，最后才跑完整任务质量。短测、成功导出、或checkpoint提交不等于完整作业质量/恢复证明。

多卡监督器逐组串行，启动前检查GPU余量与连续低负载采样、RAM和磁盘。41.875 GiB实验保守要求四卡各空闲至少44.5 GiB、三次低负载样本、RAM至少160 GiB、data盘至少16 GiB。它不是GPU独占或外部作业不增长的保证。

不把权重/checkpoint/缓存写入home，不因等待或观察超时重启仍存活的作业，不终止其他人的任务。输出目录必须是新路径，避免覆盖证据。

## 本机原始记录的只读复核

以下命令只读本机已保留artifacts，输出小型审计结果；克隆仓库后没有原始记录就不能执行这些复核。输出路径必须不存在。

~~~bash
CUDA_VISIBLE_DEVICES='' bash scripts/legacy_python.sh -m scripts.audit_fsdp_exchange_expandable \
  --root artifacts/fsdp_exchange_expandable_full_quality_v1 --output results/recheck_quality_new.json

CUDA_VISIBLE_DEVICES='' bash scripts/legacy_python.sh -m scripts.audit_fsdp_profiles \
  --root artifacts/fsdp_exchange_profiles_v1 --output results/recheck_profiles_new.json

CUDA_VISIBLE_DEVICES='' bash scripts/legacy_python.sh -m scripts.audit_fsdp_retention_pressure \
  --root artifacts/fsdp_retention_pressure_v1 --output results/recheck_pressure_new.json

CUDA_VISIBLE_DEVICES='' bash scripts/legacy_python.sh -m scripts.audit_fsdp_decoder_backend \
  --root artifacts/fsdp_decoder_backend_v1 --output results/recheck_backend_new.json
~~~

真实恢复可用 scripts.audit_fsdp_resume / scripts.audit_fsdp_1p7b_states 等专用入口；参见各入口 --help，不用短测审计器替代完整状态审计。

## 测试

~~~bash
CUDA_VISIBLE_DEVICES='' GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  bash scripts/legacy_python.sh -m pytest -q tests
~~~

Gloo需要本地回环通信；受限沙盒可能禁止它。最终已在沙盒外使用相同CPU设置通过714项，311项GPU测试跳过；另用单卡4 GiB cap完成52项核心GPU测试。跳过项不是GPU通过证据。CPU/GPU日志不随仓库发布。
