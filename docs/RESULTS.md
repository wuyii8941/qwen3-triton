# Qwen3-8B 验收结果

## 训练

四张 RTX A6000，每卡 B1/S512，完整模型 36 层。训练默认用 Triton 处理至少 64 MiB 的 FP32 FSDP reduce-scatter，其余 collective 使用 NCCL。

| 对照 | 条件 | 中位数 |
| --- | --- | ---: |
| Triton 混合通信 | 正式验收，8 个计时步 | 4.6535 s/步 |
| HF native | 同模型与输入的参考测量 | 5.2758 s/步 |
| 全 Triton 通信 | 相反执行顺序的完整步短测 | 5.970 s/步 |
| 同一 Triton 模型的混合 reduce-scatter | 与全 NCCL reduce-scatter 配对短测 | 4.750 s/步 |
| 同一 Triton 模型的全 NCCL reduce-scatter | 与混合 reduce-scatter 配对短测 | 5.161 s/步 |

正式训练四个 rank 的 399 个参数梯度均通过数值检查。16 步短程训练后，512 块语料上的验证 loss 从 2.605893 降到 2.227712。有效 reserved 峰值为 41.8652 GiB，上限为 41.875 GiB。

完整步 trace 中，Triton peer reduce-scatter 约 74.1% 的时间与计算重叠，主要未覆盖的通信是 forward all-gather。all-gather 跳过缓冲区填充的配对短测为 4.6148 → 4.6103 s/步，未确认稳定收益。训练支持既有 FSDP2 数据并行，验收范围为四卡。

## 推理

单张 RTX A6000，BF16，固定 CUDA Graph，单请求生成 8 token。

| 输入长度与优化 | 原路径 | 当前路径 | 延迟降幅 |
| --- | ---: | ---: | ---: |
| 65 token，参考累加 + prefill tiling | 207.1 ms | 201.3 ms | 2.8% |
| 512 token，16 行 decode tile | 286.7 ms | 279.4 ms | 2.6% |
| 2048 token，16 行 decode tile | 684.7 ms | 654.0 ms | 4.5% |

65-token 验收使用 `--prefill-tiling --reference-accumulation`，通过 24 项数值检查、512 块语料、两条贪心生成和五段 GPU trace。两条文本另有 23/32 个逐步 logits 诊断超阈值，贪心 token 一致。参考比较关闭 TF32 和 BF16 降精度归约。512 和 2048 token 的 Graph 优化各验证了 21 个 decode 步的 logits，均与原路径逐元素一致，生成 token 也一致。57 次独立 attention 数值检查覆盖了 KV 长度、输入尺度和未写入的缓存尾部。

这些结果适用于已测的设备、输入形状和运行方式。短程训练检查不覆盖长期收敛，Graph 延迟不代表 Eager 性能。
