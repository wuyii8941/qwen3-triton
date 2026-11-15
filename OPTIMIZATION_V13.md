# Qwen3-8B：继续优化通信和数据搬运

状态：进行中，尚无新的完整训练性能采用结论。V12 的双顺序结果保持原样，
本轮以它作为已有 Triton 控制，继续寻找真正降低完整 optimizer-step 成本的方法。
全部文件、缓存及原始记录位于 `/data1/tzh`，不写 `/home/tzh`。

## 已完成的通信模块实验

四卡 RTX A6000，Torch 2.10.0+cu128、NCCL 2.27.5，FP32 ReduceScatter。
真实 decoder 层包含 192946432 个参数，规约输入为 771785728 字节。
隔离模块计时包含新路径的重排和临时分配，不包含模型及 optimizer。

| 方法 | 顺序一，毫秒 | 逆序，毫秒 |
| --- | ---: | ---: |
| 原生四卡 RS | 70.746 | 70.747 |
| 独立四卡 RS 通信组 | 70.778 | 70.764 |
| 两级分层 RS | 103.397 | 103.251 |

分层方法先按 `[0,2,1,3]` 重排梯度块，在 `(0,1)/(2,3)` 内规约，再在
`(0,2)/(1,3)` 内规约。SUM/AVG、全部坐标的检查均通过，最大相对 L2
约 5.28e-8，但耗时更高，拒绝采用。拓扑标签本身不能证明分层更快。
独立组的隔离延迟无改善；它是否影响模型中的重叠仍需整模型短测。

随后独立测试自动协议、Simple、跨 SYS 的 P2P、SYS+Simple，并在队列首末
重新运行自动协议。随机 FP32 输入通过 AllToAll 将每个来源的目标块送到对应
rank，以显式 FP64 求和/均值作为独立参考，检查四 rank 的全部 AVG/SUM 输出。
所有配置通过模块检查，但没有候选在两个顺序下均比首末控制快至少 3%。
Simple 的大消息仍约 70.7 ms；SYS 约 83.7–84.0 ms，更慢。保留默认自动选择，
环境变量仅作用于各实验子进程，没有修改系统配置。

小型记录：[分层/独立组实验](results/qwen3_triton_v13_communication_micro_001.json)、
[协议筛选](results/qwen3_triton_v13_protocol_screen_001.json)。
模块实验不能支持训练提速结论。

## 当前候选：只省掉可证明冗余的初始化

当前基准启用确定性算法及 `fill_uninitialized_memory=True`。已证明下一生产者
完整覆写的 GEMM、precision gate/up/SwiGLU 输出，可以通过标准 PyTorch
UntypedStorage 创建张量，省去初始化；全局确定性保护保持开启。
不用于部分写入、累加器、参数或 optimizer state。原路径默认不变。

GEMM 的每个有效 `(row,column)` 坐标有唯一写入者，尾块带掩码；gate/up 同时
完整写入 G/U/Y，SwiGLU backward 对全部有效元素写 DG/DU。检查程序还将输出
预先填入 NaN，覆盖奇数形状、尾块及实际 decoder 形状，比较完整输出和梯度。
该 GPU 检查及整模型筛选尚未完成，不将源码推理当作实测证明。

另一个独立开关使用 FSDP 的 custom Comm 接口，为 AG/RS 的完整覆写缓冲省去
初始化。限定 Torch 2.10、四 rank、dim0 可被四整除的 Qwen 参数；不修改
PyTorch 内部函数。它是 HF 同样可用的基础设施优化，最终对照须给 HF 相同条件。

这条优化依赖当前确定性初始化协议。在通常关闭该保护的生产运行中，收益可能
消失；它不能代替新的计算或通信算法，也不能声称是普适的生产提速。

[新入口](scripts/train_qwen3_communication_v13.py) 仅在原 V12 分片完成后、
optimizer 创建前应用新配置。梯度比较、完整验证与完整更新/计时循环的 AST
与 V12 保持相同。短测采用全 36 层官方 8B、9 次更新，比较独立通信组、
计算输出、通信缓冲及两者组合，队列首末均运行已有路径。
短测不包含原生梯度档案和全量验证，不能用于最终采用。

只有通过原全部 399 参数梯度门槛、16 次质量更新、初末 512 块验证、
FP32 master/gradient/moments、显存及实际 CUDA 检查，并在两个顺序下均达到
至少 3% 的完整整步增量，才认定新方案有效。没有放宽原容差。

## 2026 年方向：Symmetric Memory 与 Triton 归约

[PyTorch 2026 年开发日志](https://docs.pytorch.org/devlogs/distributed/2026-05-14-python-first-comms/)
介绍了 Symmetric Memory＋Triton 的自定义通信原型；它不是已调优的性能保证。
本机安装的 [Torch 2.10 接口](https://docs.pytorch.org/docs/2.10/symmetric_memory.html)
可以查询 peer buffers 和同步信号，但本机 SM86/PCIe 是否能完成所需路径仍须实测。

已准备小型四卡探测：从四份对称 BF16 缓冲读取值，在 Triton 内 FP32
累加并输出本 rank 的块，以 AllToAll＋FP64 参考检查全部坐标。
该探测只验证可用性和正确性，没有模型接入或速度结论。即使成功，仍须证明
训练输入可无损使用该格式、缓冲同步正确、完整显存预算成立，再讨论压缩通信。
不会把 BF16 传输自动等同于 FP32 梯度规约，也不会将量化误差隐藏在容差中。

原始记录在 `artifacts/triton_qwen3_v13`。早期模块源码修改前已保存独立快照
和逐文件哈希，历史记录不会按当前源码重新解释。推理仍需要单独实现 KV-cache
decode 及端到端基准；本轮不宣称推理性能提升。
