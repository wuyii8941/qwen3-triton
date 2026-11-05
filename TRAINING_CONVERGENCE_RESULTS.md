# Qwen3-1.7B 完整训练作业比较

日期：2026-11-02。状态：两组实际作业均成功退出，按预先冻结的验证集平台规则停止；不是固定 1000 步 benchmark，也不是触发安全时限。

## 结论

在本机 RTX A6000、B=1、S=512、BF16 参数和动量、WikiText-103 持续预训练任务上：

- 首次达到同等验证质量的观察耗时：7208.42 → 5912.05 秒，**1.219×，耗时减少 17.98%**。
- 各自达到验证平台并完成最终评估、退出的进程耗时：8320.38 → 7682.28 秒，**1.083×，耗时减少 7.67%**。
- 选定 checkpoint 的测试 loss：2.39936 → 2.39764，没有观察到测试 loss 恶化；这不是跨任务、跨随机种子的无损保证。

采用的优化只有 **Triton AdamW rounded**。两组保留同一个原生 HF Qwen3 forward/backward 和 Flash SDPA，不能把这些训练收益归给自研 attention、RoPE 或 SwiGLU。这里没有测试完整 `torch.compile` 训练作业，因此不宣称胜过 compiled HF。

## 实际结果

| 指标 | HF + native fused AdamW | HF + Triton AdamW rounded |
|---|---:|---:|
| 停止原因 | validation plateau | validation plateau |
| 停止步数 | 15750 | 17500 |
| 训练输入 token 数（每步 512） | 8,064,000 | 8,960,000 |
| 验证次数，不含初始验证 | 63 | 70 |
| 最后一次验证 loss | 2.44219487 | 2.44106895 |
| 选定 checkpoint 步数 | 13750 | 15500 |
| 选定 checkpoint 测试 loss | 2.39935566 | 2.39763860 |
| 最后 checkpoint 测试 loss | 2.39738511 | 2.39594264 |
| 主函数内完整作业秒数 | 8315.47 | 7677.34 |
| 外部观察的完整进程秒数 | 8320.38 | 7682.28 |
| checkpoint 写入累计秒数 | 2726.83 | 2875.29 |

`best.pt` 是“最近一次超过 min_delta 的有效改善 checkpoint”，**不是所有验证记录的精确最小 loss checkpoint**。所以最后 checkpoint 的测试 loss 可以更低。测试集不参与选择 checkpoint 或停止训练。

Triton 多训练了 1750 步（约 11.1%），源于微小数值差异影响平台计数；不能将不同训练步数的终点比值解释为相同工作量的 kernel 加速。

## 同等质量比较

规则在两组正式训练前冻结：目标为原生 HF 全曲线最低验证 loss + 冻结的 min_delta。数值由基线结束后确定，不是事后为 Triton 挑选阈值。

本次目标：**2.4441948650851844**。

| 指标 | Native | Triton |
|---|---:|---:|
| 首次达到目标的验证步数 | 13750 | 13750 |
| 此时验证 loss | 2.44389465 | 2.44384584 |
| 主函数计时秒数 | 7204.73 | 5908.20 |
| 外部事件观察秒数 | 7208.42 | 5912.05 |

这是截至目标验证完成的累计耗时，包含此前的加载、数据传输、训练、验证和落盘，不含该目标验证之后的 checkpoint 写入。只每 250 步评估一次，不推算未观察到的更早 crossing。

## 冻结协议与工作负载

- 本地完整 28 层 Qwen3-1.7B 预训练权重初始化；不是从头预训练，也不是复现原始 Qwen 全量预训练收敛。
- 同一张 GPU0 顺序运行 Native → Triton；同种子 20261102、相同数据顺序、batch=1、seq=512、lr=1e-5；无梯度累积。
- 原生基线为 `torch.optim.AdamW(..., fused=True)`，并非故意选用逐 tensor Python optimizer。
- 两组 betas=(0.9,0.999)、eps=1e-8、weight_decay=0.01；BF16 参数、梯度与动量，不使用 FP32 master weights，不启用 AMP GradScaler。
- 原生 HF loss/backward、强制 Flash SDPA、use_cache=False，固定完整词表 logits；未启用 activation checkpointing。
- DataLoader 两个 worker、pin_memory=True、非阻塞 CUDA 传输；训练 token 文件映射为确定性 block 排列，支持从 step 恢复顺序。
- 环境：PyTorch 2.5.1+cu121、Triton 3.1、Transformers 4.51；使用项目既有虚拟环境，没有修改宿主 Python。
- 每 250 步遍历固定验证集；至少 10 次检查，连续 8 次没有相对上次有效最佳 loss 改善超过 0.002 才停止。验证噪声校准三次均为 3.1232176097109914，范围为 0。
- 每组 6 小时仅为安全上限。本次两组均由平台条件停止，没有将超时包装成收敛。

数据来源是固定 revision 的 WikiText-103，tokenizer 为本地 Qwen3 tokenizer。token 文件实际 SHA256 已逐个复核：

| Split | token 文件总数 | 完整 512-token block | 未使用尾部 token |
|---|---:|---:|---:|
| Train | 124,253,184 | 242682 | 0 |
| Validation | 262,228 | 512 | 84 |
| Test | 298,958 | 583 | 462 |

验证和测试覆盖全部完整 block；训练在平台条件满足时结束，**没有声称遍历完 124M-token 训练集的一整轮**。每块 labels=input_ids，由 HF 内部做 next-token shift；指标是这些固定非重叠块上的平均 loss，不是官方滑窗 perplexity。

## 计时及局限

主函数计时包含模型/数据加载、传输、训练、验证、durable checkpoint 写入、最终两份模型测试；不包含 Python imports 和退出清理。外部进程观察器补足该范围，两种口径分开报告。

外部事件时间是 observer 读取日志时的时间，存在 0.25 秒轮询及调度误差。Native 在启动后约 86 秒接入，Triton 约 399 秒接入。首次交接遇到目录创建竞争，修复后只重新接入计时监控，**训练未重启**；未为历史事件补造时间。两个同质量 crossing 均在接入后实际观察到。进程退出观察也有轮询误差，不能解读到毫秒精度。

共享原始文本导出和 tokenization 未计时，不包含在每组总耗时中；预检/校准也不包含在单组计时中。因此没有“从下载数据到训练结束”的总流程加速结论。整套 orchestrator 从预检到两组退出耗时 16317.46 秒，但此前的 Native 初始预检及共享数据准备不在其中。

checkpoint 写入约占 Native/Triton 主函数总耗时的 **32.8%/37.5%**。这些真实 I/O 开销稀释单步加速，且随系统 I/O 状态波动。验证成本在总耗时内，但未单独打点，不能给出精确验证耗时占比。

只有一对顺序作业、一个随机种子，不能给置信区间，也不能排除缓存/温度/I/O 顺序效应。BF16 optimizer rounding 与原生超参数表达式并非逐位相同；不外推到 FP32 master、分布式、多参数组或生产训练。

## 证据与复现入口

- `artifacts/convergence_comparison_v1/frozen_protocol.json`：两组相同协议；SHA256 `2efab832618ee3ce7a4fe48b0356a44fc900376e5481f37b1d62f2383b84fb45`。
- 同目录 `native/`、`triton/`：逐步日志、全部验证点、最后/选定 checkpoint、终态 summary。
- `comparison.json`：原始 orchestrator 比较；`external_process_clocks.json`：进程计时。
- `verified_result.json`：数据文件哈希、逐次平台规则重放、终态及四份 checkpoint 元数据审计，加上两种计时的比较。
- `artifacts/convergence_final_regression_v1.xml`：**270 passed，2 warnings**，实际 GPU 回归。
- 保存/恢复预检已实际执行 Native 和 Triton 各 2 → 4 步；不声称与未中断四步运行逐位比较过。
- 历史 v1 的 Native 初始保存预检在根目录之外；收尾时已让新 tag 的 runner 自行创建 Native save/resume 预检，不再依赖或改写该历史目录。这个新目录组织没有再启动一对长作业，不与上述已完成 v1 的测量混用。

```bash
cd /data1/tzh/proj/qwen_operator_lab
PYTHONPATH=. /data1/tzh/proj/systems/.venv/bin/python \
  benchmarks/report_convergence_comparison.py --root artifacts/convergence_comparison_v1

# 验证已有结果；输出必须使用新的文件名，拒绝覆盖历史审计。
PYTHONPATH=. /data1/tzh/proj/systems/.venv/bin/python \
  benchmarks/verify_convergence_result.py --root artifacts/convergence_comparison_v1 \
  --output /tmp/qwen-convergence-audit-new.json
```

checkpoint 是 trusted local pickle，仅可加载可信本地文件。最终审计验证元数据结构，不等同于对每个 tensor 做数值有限性或逐位等价检查；真实模型评估、optimizer 单元测试和长程曲线是不同层级的证据。

本次可用项目表述：**在 Qwen3-1.7B/WikiText-103 单卡持续预训练实验中，实现并验证 Triton AdamW，建立含数据加载、验证和 checkpoint I/O 的完整训练对比，同等验证质量耗时下降约 18%；两组均按冻结验证平台规则完成。**
