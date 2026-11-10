# 从哪里开始读代码

2026-11-10 整理后，主目录与历史实验不再共用 Python 模块搜索路径。主运行时不导入 `experiments`、`benchmarks` 或 `scripts`。

~~~text
training/                 当前真实模型训练运行时
  train.py                唯一多卡训练 CLI：加载 → 分片 → 训练 → 验证 → 最佳参数恢复
  data.py                 uint32 语料分块，不下载数据
  policy.py               后端、驻留层数、通信适用条件
  evaluation.py           全量验证与 token 加权 loss
  stopping.py             验证平台期规则
  best_state.py           CPU 最佳参数快照，不是 optimizer checkpoint
  provenance.py           当前源码身份，不冒充历史 hash
  fsdp/
    official.py           官方 FSDP2 包装，默认后端
    custom.py             自研 block 分片后端，显式选择
    primitives.py         自研分片所需 Triton copy/scale
    activation_checkpoint.py  当前 full activation checkpoint 策略
    communication.py      可选 BF16 exchange + Torch FP32 local reduction
configs/fsdp_8b.json       当前 8B 任务配置，非通用硬件配置
ops/reference/            当前训练算子的 PyTorch 数学参照
ops/triton/               当前可训练算子实现，不收纳推理专用优化与历史基线
model/                    Tiny-Qwen 单元测试模型，不是 8B 主训练模型
examples/train_tiny.py    小模型练习入口
benchmarks/               四个当前算子性能入口
tests/                    当前运行时与算子测试
experiments/legacy/       重构前的完整源码树、原始模块名及全部历史测试
results/                  已发布的历史小型证据，内容不因目录整理而重算
~~~

## 先读这条路径

1. `training/train.py`：了解训练生命周期。
2. `training/fsdp/official.py`：了解参数分片与保留完整参数的层数。
3. `training/fsdp/communication.py`：了解最后采用的 Torch decoder exchange 候选。
4. `model/qwen_tiny.py`：在小模型中逐个切换并调试 Triton 算子。
5. `ops/triton/attention.py` → `attention_backward.py`：当前 attention forward/autograd/backward。

**算子库的“当前实现”不等于 8B 默认路径全部替换成 Triton。** 真实 8B 入口保留 HF 模型、Flash SDPA 和 fused AdamW；主要历史收益来自驻留/重计算/通信组合。Tiny 模型提供自定义算子接入。没有把未胜过 Flash 的 attention 或未获益的 GEMM 强行接入主训练。

## 名字为什么改

| 旧名字（归档中保留） | 当前名字 | 职责 |
|---|---|---|
| `fsdp_convergence_job.py` | `training/train.py` | 训练入口，不再以某轮实验命名 |
| `flashoptim_convergence_job.Blocks` | `training/data.py` | 数据组件不依赖另一个训练脚本 |
| `fsdp_validation.py` | `training/evaluation.py` | 验证与分片实现分离 |
| `fsdp_best_parameters.py` | `training/best_state.py` | 明确只是最佳参数 |
| `convergence_rules.py` | `training/stopping.py` | 明确是停止规则，不承诺预训练收敛 |
| `official_fsdp.py` / `block_fsdp.py` | `training/fsdp/official.py` / `custom.py` | 同类后端放在同一层 |
| `official_keep8` / `official_keep24` | `--backend official --keep-full-layers N` | 不把某个参数值编码进后端名 |
| `benchmarks/fsdp_bf16_exchange_adapter.py` | `training/fsdp/communication.py` | 真正的通信组件不放在 benchmark 中 |

旧路径没有兼容别名回流到主目录，避免“看着干净，实际仍偷偷调用旧实现”。

## 当前入口

在项目根目录，准备本地 checkpoint 和语料后，用**新的输出目录**：

~~~bash
# 官方原生通信；默认 K36，每层 activation checkpoint，cap 41.875 GiB/卡
PYTORCH_ALLOC_CONF=backend:native,expandable_segments:True \
  bash scripts/research_python.sh -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m training.train --backend official --output artifacts/current_native_new

# Torch decoder exchange；默认 K32，其余任务配置相同
PYTORCH_ALLOC_CONF=backend:native,expandable_segments:True \
  bash scripts/research_python.sh -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m training.train --backend official --exchange torch_decoder \
  --output artifacts/current_exchange_new

# 小模型算子集成；不占用真实 8B 权重
bash scripts/research_python.sh examples/train_tiny.py --help
~~~

这是当前可维护运行入口，不是资源监督器。启动真实训练前自行确认 GPU/RAM/磁盘余量，准备协议中所需的本地模型与语料；不要并发运行两个大作业。`--model`、`--corpus` 可覆盖本地路径；驻留层数不能超过实际层数。默认 8B 的四卡预算不适用于所有机器。

Exchange 默认检查原始贡献是否可精确表示成 BF16；只允许官方 FSDP、BF16 计算、FP32 master/reduce、accumulation=1。历史 debug=False 性能路径需要显式 `--unchecked-exchange`，仅在独立门禁通过后使用；检查开销与历史吞吐不能混算。

当前入口没有 optimizer checkpoint 恢复或 HF 导出；相关已验证部署原型保留在归档，不伪装成当前 CLI 功能。`custom` 后端明确使用平均梯度，不保留含糊的 `custom_avg` 名字。

## 历史结果如何复核

~~~bash
# 验证 428 份归档源文件与重构前 SHA256 一致
bash scripts/research_python.sh scripts/verify_legacy.py

# 启动的是归档根目录，不是当前 training/ops
bash scripts/legacy_python.sh -m training.fsdp_convergence_job --help
bash scripts/legacy_python.sh -m pytest -q tests
~~~

归档的 `results` 是指向根目录证据的相对符号链接。本机 `artifacts` 链接仅用于原始记录复核，已忽略，不上传。克隆仓库后如需审计本机原始记录，需自己提供该目录；普通单元测试不需要它。

历史文档中的模块名与相对路径以 `experiments/legacy/` 为根；统一使用 `scripts/legacy_python.sh` 执行。8.36% 等数字属于原始源码与协议，**不是重新组织目录后的性能重测结果**。新入口有独立源码身份，禁止覆盖历史 manifests 或把新源码 hash 填回旧结果。

原监督器还可能依赖冻结清单中的绝对路径，因此归档不承诺所有旧长程监督器可在新目录原样重启。已完成记录的只读审计与重新训练是两回事；新训练用当前入口和新记录。

## 本次整理的验证（2026-11-10）

- 当前测试集：GPU 可用时 **106 passed**；关闭 CUDA 时 **45 passed / 61 skipped**。GPU 测试使用 GPU0、4 GiB allocator cap，包含随机上游梯度、attention standalone backward 与 Tiny 模型集成。
- 两卡 Tiny：official/native、official/Torch decoder exchange、custom 三个入口各完成两次参数更新、三次全量小验证和最佳参数恢复；每卡 cap 4 GiB，记录的两步 loss 一致。Exchange 两个 decoder 回调均执行两次。这不是完整 8B 训练重测或全参数哈希等价证明。
- 历史测试通过独立进程执行：**714 passed / 311 skipped**。
- 428 份归档文件逐字节保持原始 SHA256；历史训练源码总哈希仍为 `910b351ffbad62ed0eb3e1ba0b6186f2de84277f686026e7fe04b3b4d2015c2e`。
- 迁移后重新执行历史 8B expandable 完整结果只读审计：两个任务均通过；未更改结果、模型、数据或旧 manifest。
- 发布检查与 Markdown 链接检查通过；未上传 artifacts、数据、权重、环境或缓存。

小型摘要见 [目录整理验证记录](results/repository_layout_validation_v1.json)。本机原始输出位于 `artifacts/layout_refactor_smoke_v1/`，不发布。未重新执行真实 8B 长训练，也未把原来的收益宣称为新入口实测。
