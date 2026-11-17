# Qwen3-8B：减少参数聚合的训练策略实验

本轮推进完整 Triton Qwen3-8B 的训练整步性能。V9/V10 实现、证据和工作区均保留。
两轮短测、六组 GPU 完整对照及独立 CPU 审查均已完成。新策略在两种顺序下
均通过数值、短程质量、显存、CUDA 覆盖和至少 3% 整步收益的验收。
以下分别记录短测与完整对照，默认训练入口保持原策略。

## 改动范围

模型全部前向与反向计算继续使用已有 Triton 实现；FSDP2、NCCL 和 fused AdamW
继续使用共享 PyTorch 基础设施。BF16 计算、FP32 master/梯度/AdamW 状态、
禁用 TF32 与 BF16 reduced-precision reduction 的策略不变。

新增独立入口 [train_qwen3_training_v11.py](../../scripts/train_qwen3_training_v11.py)，允许显式选择：

- decoder 前向后保留的完整参数层数，减少反向前的参数 AllGather。
- 保留 LM head 参数，减少它的反向 AllGather。
- 跳过 embedding 反向不需要的权重 AllGather。
- 整层重计算的间隔；`0` 表示不重计算，`1` 表示逐层重计算。
- 已有 V10 GEMM 调优计划，保持显式启用。

更新、计时、全量验证及首步全参数梯度比较的 AST 与原入口一致。
原 [V9/V10 训练入口](../../scripts/train_qwen3_triton_8b.py) 未修改，默认策略未切换。
这属于训练策略与已有算子的组合优化，不宣称新发明了 FSDP 或新的 GEMM 算法。

## 实验条件和选优规则

官方 `Qwen/Qwen3-8B`，36 层、399 参数张量、8,190,735,360 个参数；
4×RTX A6000，每卡 batch=1、sequence=512。每卡 PyTorch allocator 的
allocated/reserved 峰值上限均为 41.875 GiB，其他 GPU 进程进入时拒绝该组。
本地文件与缓存只写入 `/data1/tzh`，不生成新的稠密梯度档案。

第一轮筛选 8 组；根据观察到的显存余量，第二轮复测控制组、当前最佳方案，
并扩展到 24/36 层保留，共 9 组。每组 7 次更新，含 1 次初始更新、2 次 warmup、
4 次计时。必须资源干净、FP32 状态有限、损失 mean/max 漂移低于 .002/.01，
且相对同轮控制组至少快 3%，才进入候选排序。短测没有执行全参数参考梯度比较。

候选按短测中位数排序，先检查 HF 能否在相同策略及显存预算下完成短测。
选择首个 HF 可运行的候选后，固定执行以下完整队列，再逆序运行：
原 Triton 策略、HF 新策略、Triton 新策略，共 6 组。

完整对照每组 16 次质量更新、3 次 warmup、8 次计时；初末均验证全部
512 个完整序列块（261632 个 shifted tokens）。首步四个 rank 各比较全部
399 参数的梯度，复用原生 HF 的原始档案。HF 控制使用原有更严格门槛及
前向逐位一致检查，Triton 梯度门槛保持不变。

最终必须通过数值、短程质量、显存及 CUDA 审查，并在两种顺序下均比原
Triton 策略快至少 3%，才认定训练策略改进成立。与 HF 的比较使用相同新策略，
训练策略收益与模型实现差异分别报告。两种顺序不构成统计显著性证明。

## 第二轮短测结果：用于选优

| 策略 | 秒/步 | 较同轮控制组减少 | allocated 峰值 GiB |
| --- | ---: | ---: | ---: |
| 原策略：不保留、逐层重计算 | 7.547975 | — | 34.139 |
| 复测首轮最佳：保留16层及head、隔层重计算 | 6.558181 | 13.11% | 34.171 |
| 保留24层、逐层重计算、GEMM计划 | 6.360517 | 15.73% | 35.865 |
| 保留36层、逐层重计算 | 5.665784 | 24.94% | 40.178 |
| 保留36层、逐层重计算、GEMM计划 | 5.653831 | 25.09% | 40.178 |
| 保留36层及head、逐层重计算、GEMM计划 | 5.479843 | 27.40% | 40.178 |
| 保留36层、隔层重计算、GEMM计划 | 5.639191 | 25.29% | 41.358 |
| 保留24层、不重计算、GEMM计划 | 5.887937 | 21.99% | 38.225 |
| 保留36层、不重计算、GEMM计划 | OOM | 不采用 | — |

除原控制组外，表内策略均跳过 embedding 反向权重聚合。成功组的 7 次
global loss 与同轮控制组完全一致。OOM 记录保留，不放宽预算，也不采用其计时。
部分 36 层方案计时存在波动，完整对照另行记录全部 8 个计时样本。

## 六组完整对照记录

| 实现与策略 | 顺序一，秒/步 | 逆序，秒/步 |
| --- | ---: | ---: |
| Triton 原策略 | 7.141106 | 7.195410 |
| HF 新策略 | 5.246453 | 5.251104 |
| Triton 新策略 | 5.248731 | 5.213568 |

Triton 新策略相对同轮原策略耗时减少 **26.50% / 27.54%**。与相同新策略的
HF 相比，首轮慢约 0.04%，逆序快约 0.71%，总体接近持平；不据此声称稳定胜过 HF。
候选两轮的全部 27 次 global loss、初末验证及记录的首步梯度比较指标均与原
Triton 策略一致。HF 的首步梯度误差范数为 0，前向逐位一致，16 次质量更新及
两次验证损失也与原生参考一致。候选仍遵循已有 BF16 数值容差，不声称与 HF
全部梯度逐位相同。

最终选择保留全部 36 层参数及 LM head，并跳过 embedding 反向权重聚合；
**逐层重计算保持不变**，同时显式使用已有 GEMM 计划。该组合的收益主要来自
减少参数聚合。筛选中 GEMM 计划的单独变化很小，未在完整队列中额外分离其贡献，
因此上述百分比是组合策略的整步收益。

两轮实际 trace 均确认：新策略 AllGather **40 次**、ReduceScatter **39 次**；
原策略为 **78 / 39 次**。Triton 模型 kernel 数量与 launch geometry 检查通过，
没有模型 GEMM/Flash 库回退。PyTorch 的张量基础设施、FSDP/NCCL 和 AdamW
继续按前述范围使用。

Triton allocated 峰值从约 **34.140 GiB** 增至 **40.177 GiB**；新策略的
reserved 峰值为 **41.840 GiB**。同策略 HF 分别约为 **40.248 / 41.865 GiB**，
全部符合原 41.875 GiB 预算。这里是计时训练区间的 PyTorch allocator 峰值，
不代表 NVML 统计的进程总显存。这是用显存换取更少通信的取舍，适用范围仍为
本次 batch、sequence、world size 和硬件，不是其他形状的通用最优配置。

## 实际执行的审查

[qwen3_training_v11_contract.py](../../scripts/qwen3_training_v11_contract.py) 按重计算层数
核对全部模型 kernel 数量、GEMM/gate-up 的 grid 和 warp 数，以及实际 NCCL
AllGather/ReduceScatter 次数。原控制组应为 78/39；保留36层及head并跳过
embedding 反向聚合时应为 40/39。该规则已用三份真实 V10 trace 校验，并拒绝
错误的重计算和参数保留标签；六组正式运行的实际 trace 均通过对应规则。

[独立验收报告](../../results/qwen3_triton_v11_training_review_v1.json) 记录六组数值、
短程质量、计时样本及 CUDA 审查；[证据索引](../../results/qwen3_triton_v11_evidence_index_v1.json)
给出小型副本、原始记录与源码的哈希。所有 399 参数的稠密梯度坐标均在运行时
比较；候选稠密梯度没有另存档，CPU 复核验证原生参考档案及记录的误差运算，
不声称独立重算了未存档的候选梯度。

原始记录位于 `artifacts/triton_qwen3_v11`：`strategy_screen_002`、
`retention_screen_001`、`formal_001`。`communication_model_001.json` 是根据官方
参数数量建立的 ring payload 分析，不是实测链路流量、带宽或延迟；其内存估计
是排除梯度、激活、workspace 和 allocator 碎片的下界。

首个 `strategy_screen_001` 在 GPU admission 前因复制脚本格式清理而终止，
没有模型训练或计时；保留对应记录。本分支提供通过验收的可选配置，原入口
默认值不变。不作完整收敛或推理性能声明。

## 显式使用策略

完成 [环境、权重、语料与参考档案准备](../../QWEN3_TRITON_REPRODUCE.md) 后，使用新的
训练入口，并保留原来的输入、验证、梯度参考及输出参数。Triton 候选追加：

```text
--variant triton_optimized
--keep-full-layers 36
--checkpoint-every 1
--skip-embedding-backward-gather
--retain-lm-head
--gemm-plan results/qwen3_triton_v10_gemm_plan_001.json
```

HF 对照使用相同训练策略，将 variant 改为 `hf` 并省略 `--gemm-plan`。
没有改变原入口的默认值。该显存配置需要重新验证才能用于更长序列或其他 batch。

本分支已包含经过统一审查的小型证据和
[显式策略描述](../../results/qwen3_triton_v11_strategy_v1.json)。在保留完整原始记录的环境中可用
`scripts/bundle_qwen3_training_v11.py --artifacts ARTIFACT_ROOT --results results`
生成小型证据副本及显式策略参数描述。描述文件不是新的 CLI 配置加载器。
