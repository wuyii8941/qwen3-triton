# Qwen3-8B V13：无损梯度传输与 Triton FP32 规约

状态：完整八组双顺序实验及独立 CPU 审查通过，推荐显式启用 peer 通信方案。
在同一训练协议下，相对 V12 整步耗时减少 **7.95% / 7.85%**，
相对同轮原生 HF 减少 **14.75% / 13.80%**，相对使用同一通信算法的 HF
减少 **9.24% / 8.57%**。原默认配置不变；全部文件、缓存及原始记录
位于 `/data1/tzh`，不写 `/home/tzh`。

## 完整模型结果与公平对照

| 同轮完整 8B 训练 | 顺序一，秒/步 | 逆序，秒/步 |
| --- | ---: | ---: |
| 原生 HF＋Flash，共有 V11 策略 | 5.296109 | 5.240922 |
| V12 Triton，词表分片 head＋融合 dW | 4.904509 | 4.902587 |
| HF＋Flash＋同一 peer 通信 | 4.974458 | 4.940806 |
| 完整 Triton＋peer 通信 | 4.514766 | 4.517596 |

官方 Qwen/Qwen3-8B，36 层、399 个参数张量，4×RTX A6000，
每卡 B1/S512；BF16 计算，FP32 master/梯度/AdamW moments。
V11 公共参数保留/逐层重计算策略、native fused AdamW 和确定性填充均保持
相同。V12 和候选使用同一 GEMM 计划、词表分片 head 和融合 dW。
每组实际更新 27 次，前 16 次用于质量，随后 3 次预热、8 次计时；整步计时
包含 optimizer，取四 rank 最大 wall time 的中位数，编译、参考复核及验证不计入。
固定顺序为 HF＋peer、V12、原生 HF、候选，然后逆序；没有按速度选择尝试。

推荐方案的有效 allocated/reserved 峰值最高 **37.966 / 41.871 GiB/卡**，
包含 0.5 GiB 外部对称缓冲预留，仍在原 41.875 GiB 上限内。
HF 可以使用相同 peer 算子，本轮同时测量了该控制，不能将通信收益称为独占。
这些是本地固定 Torch 2.10.0 / Transformers 4.51.0 的 HF＋Flash 对照，
不代表最新 HF/compile 或其他部署系统的最好成绩。

所有八组通过原全部 399 参数首步全坐标梯度门槛、16 次质量更新、初末全量
512 块验证（261632 个 shifted tokens）、有限 FP32 状态、显存及实际 CUDA
契约。HF 控制仍使用更严格的梯度门槛和逐位前向要求；容差未放宽。
质量区间平均损失偏差最大 0.001626000、单步偏差最大 0.003835917；初末验证损失偏差最大 0.001933967。
候选两轮全部 27 次记录损失、初末验证和首步梯度误差统计均与 V12 一致。
这只描述已记录的量；CPU 审查重新验证原生参考档案与候选误差记录，
候选稠密梯度没有另存档，不能声称独立重放或逐位比较了整个候选训练轨迹。

两轮控制不是统计显著性证明，16 次质量更新不代表长期收敛；没有推理提速
结论。模块约 23.6% 的 RS 收益与完整 optimizer-step 收益分别披露。
分层通信、协议筛选和省初始化的负结果、初次 API 调用错误，以及旧完整队列
验证 OOM 均保留，未用失败或受干扰尝试支持性能结论。

[独立完整审查](../../results/qwen3_triton_v13_training_review_v1.json)、
[显式策略](../../results/qwen3_triton_v13_strategy_v1.json)、
[完整尝试历史](../../results/qwen3_triton_v13_full_attempt_history_v1.json)、
[环境](../../results/qwen3_triton_v13_environment_v1.json) 和
[证据索引](../../results/qwen3_triton_v13_evidence_index_v1.json) 给出数值、资源、
源码及原始记录哈希；模型、语料、环境、原始 trace 和梯度档案不随 Git 分发。

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
独立组的隔离延迟无改善；整模型短测也更慢，不采用。

随后独立测试自动协议、Simple、跨 SYS 的 P2P、SYS+Simple，并在队列首末
重新运行自动协议。随机 FP32 输入通过 AllToAll 将每个来源的目标块送到对应
rank，以显式 FP64 求和/均值作为独立参考，检查四 rank 的全部 AVG/SUM 输出。
所有配置通过模块检查，但没有候选在两个顺序下均比首末控制快至少 3%。
Simple 的大消息仍约 70.7 ms；SYS 约 83.7–84.0 ms，更慢。保留默认自动选择，
环境变量仅作用于各实验子进程，没有修改系统配置。

小型记录：[分层/独立组实验](../../results/qwen3_triton_v13_communication_micro_001.json)、
[协议筛选](../../results/qwen3_triton_v13_protocol_screen_001.json)。
模块实验不能支持训练提速结论。

## 未采用的整模型筛选：省去冗余初始化

当前基准启用确定性算法及 `fill_uninitialized_memory=True`。已证明下一生产者
完整覆写的 GEMM、precision gate/up/SwiGLU 输出，可以通过标准 PyTorch
UntypedStorage 创建张量，省去初始化；全局确定性保护保持开启。
不用于部分写入、累加器、参数或 optimizer state。原路径默认不变。

GEMM 的每个有效 `(row,column)` 坐标有唯一写入者，尾块带掩码；gate/up 同时
完整写入 G/U/Y，SwiGLU backward 对全部有效元素写 DG/DU。检查程序还将输出
预先填入 NaN，覆盖奇数形状、尾块及实际 decoder 形状，比较完整输出和梯度。
六组 GPU NaN 污染检查已通过，全部前向与梯度坐标逐位一致，全局确定性填充
仍开启。九步整模型筛选和独立 CPU 审查也已完成：

| 短测配置 | 秒/步 |
| --- | ---: |
| 队首控制 | 4.901463 |
| 独立 RS 通信组 | 5.135879 |
| 仅计算输出省去初始化 | 4.882910 |
| 仅通信缓冲省去初始化 | 4.840555 |
| 两者组合 | 4.816007 |
| 队尾控制 | 4.897864 |

六组全部九步损失一致，实际 CUDA 契约、有限 FP32 状态及原显存门槛通过。
组合版相对首尾控制只减少约 1.74% / 1.67%，均未达到原 3% 门槛，**不采用**。
这些运行没有原生梯度档案比较或全量验证，不称为完整训练验收。
[独立短测审查](../../results/qwen3_triton_v13_short_screen_review_v1.json) 保留全部结果。

另一个独立开关使用 FSDP 的 custom Comm 接口，为 AG/RS 的完整覆写缓冲省去
初始化。限定 Torch 2.10、四 rank、dim0 可被四整除的 Qwen 参数；不修改
PyTorch 内部函数。它是 HF 同样可用的基础设施优化，最终对照须给 HF 相同条件。

这条优化依赖当前确定性初始化协议。在通常关闭该保护的生产运行中，收益可能
消失；它不能代替新的计算或通信算法，也不能声称是普适的生产提速。

[新入口](../../scripts/train_qwen3_communication_v13.py) 仅在原 V12 分片完成后、
optimizer 创建前应用新配置。梯度比较、验证数值核心与完整更新/计时循环的 AST
与 V12 保持相同；后续增加的验证外层仅管理 peer 缓冲生命周期。
短测采用全 36 层官方 8B、9 次更新，比较独立通信组、
计算输出、通信缓冲及两者组合，队列首末均运行已有路径。
短测不包含原生梯度档案和全量验证，不能用于最终采用。

只有通过原全部 399 参数梯度门槛、16 次质量更新、初末 512 块验证、
FP32 master/gradient/moments、显存及实际 CUDA 检查，并在两个顺序下均达到
至少 3% 的完整整步增量，才认定新方案有效。没有放宽原容差。

## 2026 年方向：Symmetric Memory 与 Triton 归约

[PyTorch 2026 年开发日志](https://docs.pytorch.org/devlogs/distributed/2026-05-14-python-first-comms/)
介绍了 Symmetric Memory＋Triton 的自定义通信原型；它不是已调优的性能保证。
本机安装的 [Torch 2.10 接口](https://docs.pytorch.org/docs/2.10/symmetric_memory.html)
提供 peer buffers 和同步信号接口。本机 SM86/PCIe 的实际可用性检查现已通过。

首次探测错误地在 CUDA 默认 allocator 已注册时重复调用 `set_backend('CUDA')`，
报 allocation backend 错误；这是探测 API 调用错误，不是硬件不支持的证据。
保留原错误、原始源码快照及分类。检查默认 backend 后再分配，四 rank 探测
均成功，全部坐标对 AllToAll＋FP64 参考的最大绝对误差为 1.49e-8。

随后使用真实层大小，比较原生 RS、FP32 peer 和 BF16 peer；后两者都使用 FP32
累加和输出，计时包含输入拷贝/检查、前后两个 GPU barrier，不含一次性分配和
rendezvous。源值先按 BF16 舍入后升为 FP32，模拟“已有 BF16 梯度的 FP32 桶”，
不能由该模块输入设置推断真实训练也满足同样条件。

| 层大小 RS 模块 | 顺序一，毫秒 | 逆序，毫秒 |
| --- | ---: | ---: |
| 原生 FP32 RS | 70.854 | 70.771 |
| FP32 peer | 107.107 | 107.111 |
| 无损 BF16 peer，FP32 累加/输出 | 54.108 | 54.121 |

BF16 peer 的模块耗时减少约 23.6%，FP32 peer 更慢；全部四 rank 的 AVG/SUM
完整坐标、不可表示 FP32 值的负例，以及反复更换数据的读取检查通过。
[模块记录](../../results/qwen3_triton_v13_symmetric_rs_micro_v1.json) 保存原实现来源。
模块和实际训练接入分开验收，不把模块收益直接称为整步收益。

## 从试运行到完整验收：保留失败并修复生命周期

[接入](../../scripts/fsdp_peer_rs_v13.py) 保持 FSDP 的 FP32 输入/输出及 AVG/SUM
接口，只替换 36 层 decoder 的大梯度规约，其他尺寸仍用 native FP32 NCCL。
每次传输、每个坐标都会检查转换为 BF16 再还原后的 FP32 **位模式**；任何
差异都会记录为失败并拒绝结果，不能用原梯度容差掩盖额外量化。
现有训练的本地梯度已按 BF16 舍入，FP32 桶中的这些值通过该检查后才可称为
无损传输。规约自身仍按 FP32 逐项求和，求和顺序变化须接受原梯度/质量检查。

训练时每卡复用约 0.359 GiB 的对称缓冲。为避免绕过 allocator 预算，额外保守预留
0.5 GiB，并将 Torch allocator 上限降至 41.375 GiB；有效峰值为 Torch 峰值
加这 0.5 GiB，仍须低于原 41.875 GiB。setup 同时核对 NVML 进程内存增量，
实际约 370 MiB，不等同于完整进程总显存预算。验证时物理释放该缓冲，恢复
原 41.875 GiB Torch 限制；恢复训练前重新建立缓冲并恢复 41.375 GiB Torch
限制及 0.5 GiB 外部分配预算，始终不超过原总预算。

官方全 36 层、399 参数的九步 Triton 试运行已完成，包含首步原生参考梯度
全坐标比较：四 rank 通过原门槛，所有大消息的无损检查通过，九步损失与
V12 对应步骤一致。整步短测为 **4.523120 秒**，相对最近首尾控制减少
**7.72% / 7.65%**。有效 allocated/reserved 峰值约 **37.966 / 41.871 GiB**。
这仍没有 16 次质量更新和完整验证，不能称为正式采用结果。

HF/Flash 接入相同通信路径的九步试运行也通过：更严格的原生控制梯度门槛、
前向逐位一致、无损传输和有效显存均合格。梯度全局相对 L2 不超过 8.65e-11，
九步损失与原 HF 一致；有效峰值约 40.748 / 41.852 GiB。说明该 Triton
通信算子可以移植到 HF，需要自定义 Comm 和数值/显存契约，不是独占算法。

首次完整队列 `formal_001` 在 HF＋peer 的第 16 次更新后全量验证中 OOM。
训练九步试运行通过没有覆盖这一情况；验证不使用 peer 梯度缓冲，但它仍占用
保留预算，限制了 HF 原生 CE 的大临时张量。该队列停止并保留日志、监控和
98 项来源快照，未将失败组用于性能结论，也没有复用前两组作为新队列控制。

修复只管理缓冲生命周期：所有 GPU 流同步后释放 peer 存储，核对 NVML 观察到
释放约 370 MiB，再执行原 batch=4 的全 512 块验证；验证数值核心没有改变。
训练恢复前重建缓冲，跨重建保留同一个错误标记及累计覆盖计数。
[生命周期检查](../../scripts/check_peer_lifecycle_v13.py) 在四 rank 上通过两次释放/
重建、真实层大小输出检查及错误标记负例：即使在缓冲暂停时置入错误，也会拒绝
恢复，不能靠新建缓冲清零错误。没有减少验证批量、样本或放宽显存/数值门槛。

修复后的 `formal_002` [完整队列](../../scripts/run_qwen3_peer_full_v13.py) 已完成。
为先覆盖此前的失败点，事先固定顺序为 HF＋peer、V12 Triton、原生 HF、
Triton＋peer，再逆序执行，共八组；每组 27 次实际更新、全部 399
参数首步梯度、16 次质量更新、初末全部 512 块验证及实际 CUDA 契约检查。
全部八组通过原门槛，候选两轮均比三个合格控制快至少 3%，并通过
[独立 CPU 审查](../../scripts/review_qwen3_peer_full_v13.py)，据此推荐显式启用。
本队列共有 15 次尝试；其中 7 次观察到未归属本作业的 GPU 进程，
按预设规则剔除并保留日志，等待连续 120 秒四卡空闲后重试。接纳的是每个角色
第一个完整且监控干净的尝试；没有因速度或数值筛选重复。
原入口默认值和此前已发布的 V12 结论均未改变。

原始记录在 `artifacts/triton_qwen3_v13`。早期模块源码修改前已保存独立快照
和逐文件哈希，历史记录不会按当前源码重新解释。推理仍需要单独实现 KV-cache
decode 及端到端基准；本轮不宣称推理性能提升。

## 实现范围与显式复现

新增的两个 Triton kernel 是检查并写入传输缓冲的 `_checked_wire_copy`，以及
读取四个 peer 分片、以 FP32 累加并写出结果的 `_peer_reduce`。每次反向替换
36 次 decoder 大梯度 ReduceScatter，仍执行 36 次逻辑规约和 72 次真实 GPU
barrier。没有删除规约语义，也没有将 trace 中的自定义 kernel 改名成 NCCL。
一个正式角色包含 27 次 optimizer 更新和额外一次不更新参数的 trace 反向，
故每 rank 期望 1008 次 peer 调用，检查 `1008 × 192946432` 个输入坐标。

这不是把 FSDP 的 `reduce_dtype` 改为 BF16：其输入、累加及输出仍为 FP32，
只传输已能由 BF16 精确表示的本地输入。FP32 加法次序可能变化，不能由传输
无损推导整个梯度或训练轨迹与 HF 逐位相同。梯度累积、多机、其他消息尺寸、
模型或软件版本需要重新验证，当前入口固定 Torch 2.10.0、四 rank 的 CUDA
Symmetric Memory backend。相关接口仍具有实验性；本机证明可用不代表各平台
都支持或更快。HF 接入相同通信算法是正式控制，优化收益不归为 Triton 独占。

先按 [原复现流程](../../QWEN3_TRITON_REPRODUCE.md) 准备固定环境、官方 Qwen3-8B
权重、语料、重新绑定的 input manifest 和原生首步梯度参考。通过
`scripts/research_python.sh -m torch.distributed.run --standalone --nproc_per_node=4`
调用 [peer 训练入口](../../scripts/train_qwen3_peer_rs_v13.py)，保留输入/输出参数并追加：

```text
--variant triton_optimized
--vocab-ce --fused-vocab-dw
--keep-full-layers 36 --checkpoint-every 1
--skip-embedding-backward-gather
--gemm-plan results/qwen3_triton_v10_gemm_plan_001.json
--quality-steps 16 --warmup-steps 3 --measure-steps 8
--full-validation --trace --reference-raw-dir REFERENCE_DIR
```

输入/输出参数为 `--protocol PROTOCOL --input-manifest INPUT_MANIFEST
--input-root INPUT_ROOT --output NEW_OUTPUT`。使用 GPU 映射 `0,1,2,3`、
`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8` 和
`PYTORCH_ALLOC_CONF=backend:native,expandable_segments:False`；不强制
`NCCL_PROTO` 或 `NCCL_P2P_LEVEL`。运行前应取得四卡独占租约；新结果目录不得
覆盖已有证据。缓存和临时目录由 wrapper 放在项目 `.runtime-research` 下。

HF＋peer 使用同一 peer 入口，选择 `--variant hf --retain-lm-head`，省略两个
vocab 标志和 GEMM 计划。V12/native HF 控制使用
[原生通信入口](../../scripts/train_qwen3_communication_v13.py) 与
`--communication native`，其余参数分别与 Triton/HF arm 对应。
[完整研究队列](../../scripts/run_qwen3_peer_full_v13.py) 和
[CPU 审查](../../scripts/review_qwen3_peer_full_v13.py) 固定本机历史参考、试运行及
输入路径，迁移时须准备相同证据并调整编排路径；它们不是通用的便携训练框架。
本分支的 peer 路径显式启用，原默认入口和 V9–V12 历史结果保持不变。
