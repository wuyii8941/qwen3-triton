# 汇总结论

下表区分清理后的正式验收与清理前的历史对照。正式 hybrid 数据来自当前分支、同一固定输入、4×RTX A6000、B1/S512。

| 实验 | 条件 | Triton | HF 对照 | 结论 |
| --- | --- | --- | --- | --- |
| 正式 hybrid 训练 | 4×RTX A6000，每卡 B1/S512，8 个计时步 | 中位 4.6535 秒/步，均值 4.6110 | 历史 HF native 中位 5.2758 秒/步 | hybrid 快约 11.8%；相对清理后 full 对照中位 5.970 秒/步快约 22.1%。 |
| 全 Triton collective 对照 | 4×RTX A6000，每卡 B1/S512，两次反向顺序短筛 | 5.971 / 5.970 秒/步 | 历史 HF native 5.2760 / 5.2756 秒/步 | full 模式仍无优势，只保留为显式对照。 |
| 固定 Graph 推理 | 单 RTX A6000，B1/S65，生成 8 token | 202.146 / 202.154 ms | 220.338 / 220.151 ms | 完整请求约快 8.2%；Eager 仍慢于 HF。 |

当前 hybrid 正式验收覆盖全部 399 个参数梯度、16 步短程质量、训练前后各 512 块验证和四段 GPU trace。4/4 rank 的 numerical gate 均通过，logits 相对误差为 0，loss 误差不超过 2.39e-7；验证 loss 从 2.605893 降到 2.227712。有效 reserved 峰值为 41.8652/41.875 GiB，因此显存余量很窄。推理历史验收曾覆盖 24 项数值检查、两条 16-token 生成、512 块 S512 验证和五段 GPU trace；另有 23 个逐步文本 logits 诊断项超过阈值。

原始 JSON、梯度、权重、语料和 profiler trace 已从当前树移除，可从 Git 历史追溯。当前可复现入口是 `scripts/prepare_corpus.py`、`scripts/configure_inputs.py`、`scripts/export_training_reference.py`、`scripts/train.py`、`scripts/generate.py` 和 `scripts/verify_inference.py`；固定配置是 `configs/input_recipe.json` 与 `configs/gemm_plan.json`。

因此当前结论是：推理 Graph 路径在所测形状有优势；训练应默认采用 selective hybrid collective，不能再称为“全 Triton collective runtime”。下一阶段继续在同一正确性门槛下量化 overlap 与 TP/PP/DP；没有完整步收益的候选不纳入稳定代码。

## Collective、overlap 与并行策略

默认 hybrid 只把至少 64 MiB 的 FP32 FSDP reduce-scatter 路由到 Triton peer transport；all-gather、小 reduce-scatter、all-reduce、all-to-all 和 barrier 仍由 NCCL 执行。正式 28-update 运行中，rank 0 记录到 1,036 次 Triton reduce-scatter、1,506 次 NCCL all-gather、56 次 NCCL 小 reduce-scatter、242 次 NCCL all-reduce、1 次 all-to-all 和 28 次 barrier。外部显存增量为 500 MiB；320 MiB peer staging 是在完整模型内通过 512 MiB 外部预算的最大已验收配置。

去掉 profile 更新中重复的 399 参数有限性检查后，干净 full-step trace 显示：peer reduce-scatter 占 2.247 s，其中 1.665 s 与计算重叠，重叠率约 74.1%；NCCL 通信占 1.492 s，只有 0.105 s 与计算重叠。主要 overlap 缺口是 forward all-gather，而不是 peer reduce-scatter。更激进的两层 FSDP 预取需要额外容纳一层数百 MiB BF16 参数，但正式有效 reserved 峰值距离 41.875 GiB 上限仅约 10 MiB；双 peer buffer 也不能证明完整步收益。因此当前不增加 overlap 模式。

四卡拓扑是 `(0,1)`、`(2,3)` 为 PIX，其余为 SYS。实测 4 MiB NCCL all-reduce 在 PIX 为 0.403–0.407 ms、SYS 为 1.05–1.06 ms；128 MiB 输出的 all-gather 在 PIX 为 6.64–6.71 ms、SYS 为 17.24–17.43 ms；256 MiB FP32 输入的 reduce-scatter 在 PIX 为 14.05–14.25 ms、SYS 为 24.37–24.67 ms。由此只值得继续研究 `TP=2 × DP=2`：TP 固定在 PIX 对，DP 跨 SYS。TP=4 会让每层通信跨 SYS，拒绝；当前 FSDP 本身就是 DP/sharded baseline；microbatch=1 时 PP2 的 1F1B 利用率只有 1/2，拒绝。TP2×DP2 仍需重构 attention head、row/column parallel linear、vocab CE、两 rank DP group 与 runtime，尚无完整训练步证据，所以不加入公开入口。

## 推理和 all-gather 复验

以下推理数据均为单张 A6000、B1、固定 CUDA Graph、生成 8 token。对 HF 的检查关闭 TF32 和 BF16 降精度归约。此前短 prompt 对照遗漏了后者，HF 在相同输入上的最大 logits 偏差为 0.375–0.75。关闭后 HF 的有缓存和无缓存 prefill logits 逐元素一致。

- 65-token **prefill tiling** 搭配参考累加时，Graph 延迟约为 207 → 201.3 ms。三个样本的 prefill logits 与 HF 逐元素一致，21 个 teacher-forced decode 步和贪心 token 均通过。该组合在独占 A6000 上通过正式验收。普通累加约为 205.1 → 194.6 ms，开启和关闭 tiling 的对照均只通过 14/21 个 decode 数值检查。
- **长上下文 decode** 在 SM86 单批、单 token、可见 KV ≥512 时采用 16 行 query tile。512-token 请求从 286.7 降到 279.4 ms，2048-token 请求从 684.7 降到 654.0 ms。两种长度各 21 个 teacher-forced 步的 logits 和生成 token 均与 64 行路径逐元素一致。19 个 KV 长度和三种输入尺度共 57 个独立 attention 检查均逐元素一致，包含 NaN 缓存尾部。兼容性审计通过 24 项数值、512 块语料、两条生成和五段 trace。
- 原 **split-KV** 改变了累加和 BF16 softmax 权重的舍入顺序，2048-token 的一个近并列样本从 token 197 变为 220。16 行路径保留 32-key 反向遍历及原舍入，解决该样本的分叉。Eager 的实测波动盖过收益，16 行路径只在固定请求 Graph 的预热和捕获时启用。
- 免打包 **QKV 合并** 用单次 launch 写出三块独立连续输出。65-token Graph 从约 194.75 降到 193.97 ms，三个样本的 logits 和 token 逐元素一致。当前收益约 0.4%，保留实验结论。
- 持久复用 **all-gather 缓冲区** 的两轮五步均值为 4.610、4.614 s，早期匹配基线为 4.650 s，有效 allocated 增加约 368 MiB。跳过会被完整覆盖的确定性填充后，显存回到基线。更紧的计时对照为 4.6148 → 4.6103 s，四个 rank 的首步 399 参数梯度字节摘要一致。外部 GPU 任务打断了基线的最终状态摘要。训练默认采用既有 hybrid。
