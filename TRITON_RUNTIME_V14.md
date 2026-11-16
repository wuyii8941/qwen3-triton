# Qwen3-8B：尽可能完整的 Triton 训练数据通路

状态：完整8B功能验收通过；**性能未采用**。基于已验收的 V13 `95a27d0`，
独立分支 `research/triton-runtime-v14`；历史分支和默认性能推荐不改。正式四组
正逆序对照及独立CPU审核通过：完整 GPU 数据通路的数值、质量、显存和内核覆盖
符合预先冻结门槛。实测完整 V14 为 **5.927 秒/步**，HF＋Flash 为 **5.276 秒/步**，
V13 为 **4.521 秒/步**；完整实现分别慢约 **12.3%** 和 **31.1%**。V14 作为可复现的
全 Triton GPU 路径发布，性能推荐仍是 V13。实验范围仅限固定的 4×A6000、B1/S512
和冻结的软件环境，不代表收敛、推理或跨硬件结论。

所有输出、缓存、临时文件位于 `/data1/tzh`，不写 `/home/tzh`。

## 正式完整8B验收

预先冻结的实验包含 HF native、V13、V13＋Triton AdamW、full runtime，按正序和
逆序各跑一轮。每个角色完成16次质量更新、3次预热、8个计时样本和1次真实
profile更新；在更新0和16后都运行全部512个验证块。每角色28次实际更新。
独立审核确认全部399参数梯度、短程质量、FP32状态、完整验证、有效显存和
全部训练 GPU 事件门槛通过，没有修改原阈值。

| 实现 | 正序中位数（秒/步） | 逆序中位数（秒/步） | 与同轮 HF 耗时差 |
| --- | ---: | ---: | ---: |
| HF＋Flash | 5.2760 | 5.2756 | 基线 |
| V13 | 4.5188 | 4.5222 | 快约 14.3% |
| V13＋Triton AdamW | 4.6249 | 4.6287 | 快约 12.3% |
| V14 full runtime | 5.9268 | 5.9269 | 慢约 12.3% |

完整运行时在两种顺序下都比 V13 慢约31.1%，所以它**通过功能资格，但没有
通过性能采用门槛**。Triton AdamW 单独加入 V13 后也慢约2.35%；这些结果不能
支持把新增 runtime/optimizer 当作提速。逐角色结果、原始尝试历史、冻结协议、
资源审核、512块验证记录及历史源版本清单见[正式证据索引](results/qwen3_triton_runtime_v14_evidence_index_v1.json)
及其关联 JSON。密集梯度、模型权重、检查点和 profiler 原始 trace 不随分支发布。

主机仍负责 PyTorch FSDP2/autograd 状态机、CUDA Driver 内存分配和 IPC 映射、
streams/events、CPU 数据准备及 host-device DMA。full runtime 的训练 GPU 数据路径
（模型、FSDP 张量打包/拆分与累加、AdamW、通信、屏障和验证统计）均经 Triton；
这是受支持训练路径的限定，不声称所有外围服务也是 Triton。

### 留在主机侧的工作及原因

这里的目标是把 GPU 张量数据面交给 Triton；Triton kernel 由 CPU 启动并操作
GPU 指针，本身不运行 Python、创建 CUDA runtime 对象或管理多进程生命周期。
因此 FSDP2 的主机状态机和 CUDA 资源管理保留在 PyTorch/CUDA Driver，GPU 数据
操作则由 Triton 执行。

| 主机侧工作 | 为什么不是 Triton kernel | 实际边界 |
| --- | --- | --- |
| FSDP2/autograd hook、模块递归、参数状态和分片元数据 | 它们运行在 Python/C++ 主机控制流上，注册回调、管理对象生命周期并决定何时启动 GPU 工作；GPU kernel 不能代替这些框架回调或改变主机对象图。 | 保留 PyTorch FSDP2 调度；AG/RS、chunk-cat、拆分拷贝、cast、fill、梯度累加均进入 Triton。 |
| 多进程启动、CPU rendezvous 和 IPC 句柄交换 | 进程创建、socket/Gloo 消息与不透明句柄序列化属于 OS/主机通信；kernel 只能在已有进程与映射内执行。 | Gloo 仅交换 rank/IPC 元数据并协调生命周期；张量 payload 和 GPU 屏障使用 Triton。 |
| CUDA allocation、IPC map/unmap、stream/event 创建和销毁 | 这些是必须在 kernel 启动前后由 CUDA Driver/runtime 调用的资源管理 API，不是对 GPU 张量逐元素计算。 | `runtime/peer_memory.py` 在 CPU 调用 Driver API；映射后的地址传给 Triton kernel 做读写与 system-scope 同步。 |
| CPU 文件读取、token/block 准备及 host-device DMA | 文件系统和数据加载在 CPU；DMA 由 CUDA copy engine/runtime 安排，Triton kernel 不负责提交该传输。 | 输入仍在 CPU 准备并通过 host-device copy 上传；设备内转换/拷贝走 Triton。 |
| CUDA stream/event 查询、同步和显存/NVML 记账 | 这些操作查询或改变 host runtime 的调度/资源状态；kernel 无法创建或等待一个尚未由主机建立的 stream/event。 | 由 PyTorch/CUDA Driver/NVML 管理；其间发生的 GPU 张量计算仍受严格 dispatch 和 trace 审核。 |

full 模式通过 `TritonRuntimeMode` 将未实现的 CUDA 张量操作直接报错，并以原始
kernel 与 GPU 内存事件再审核隐藏路径。正式运行的四个 rank 都启用了严格
dispatch，未实现的 CUDA 张量操作会报错；原始 profiler trace 由 rank 0 采集。
两轮 rank 0 trace 在 setup、首次更新、完整更新、完整验证中分别记录808、
13579、12294、51459 个 GPU kernel 事件；这些事件均归入 Triton，未分类/原生
计算为空，原生 GPU memset/设备间拷贝事件为零。审查范围和哈希见
[`training_review`](results/qwen3_triton_runtime_v14_training_review_v1.json)；
V14 资格限于审核中的 4×A6000、软件版本和消息形状。

下面的组件和失败条目是按当时进度保留的审计历史；它们的“尚待完成”状态已由
上面的正式整体验收结果覆盖，不应解读为当前状态。

用户要求覆盖整个训练流程，不能再把“模型前后向都是 Triton”称为全流程完成。
以下项目必须逐项落实，并用包含 optimizer、统计和通信的完整 step trace 核对。
CPU 数据准备、主机控制、驱动服务与 GPU 数据计算分开说明，不隐藏原生计算回退。

| 项目 | 现状与本轮要求 | 验收证据 |
| --- | --- | --- |
| 模型全部前后向 | 已有完整 Triton；保留官方8B参数映射 | 全399参数梯度、真实内核与形状 |
| AdamW | 替换 native fused AdamW；FP32参数/梯度/m/v及逐参数步数 | 连续更新、非零状态、跳过梯度、恢复状态及8B轨迹 |
| FSDP GPU打包/拆分 | 替换 chunk_cat、split copy、cast、foreach copy | 分片布局独立参考与完整 step trace |
| 参数/梯度通信 | Triton AllGather、ReduceScatter，所有大小均覆盖 | 四rank全部坐标、尾块、重复换数据、stream顺序 |
| head与训练统计通信 | Triton AllReduce、AllToAll及scalar归约 | SUM/AVG/MAX等实际使用类型，FP32/FP64/整数 |
| GPU同步 | 尝试Triton系统作用域release/acquire屏障 | 长重复、换数据、超时失败；不冒充零拷贝排序 |
| allocator初始化/布局拷贝 | Triton fill/zero/clone/copy/contiguous | NaN初始化、任意实际stride、无未初始化读 |
| autograd梯度累加/辅助算术 | Triton add与所有实际出现的GPU运算 | 严格dispatch白名单＋实际完整step内核审查 |
| 质量与整体性能 | 正式8B、固定BF16计算/FP32累加，公共协议对照 | 全梯度、16步质量、512块验证、显存和完整step计时 |

验收不能只看Python入口使用了哪个类。FSDP2中的参数布局、autograd hook、
stream/event生命周期属于主机控制；本轮优先替换其全部GPU数据通路。保留哪些
主机组件及原因必须明确列出；如果公开接口/dispatch无法接管某个GPU计算，
继续实现相应分片路径或给出实证限制，不能直接算作已完成。

Triton GPU kernel不能代替进程启动、CPU文件系统、CUDA虚拟内存映射、主机
pageable内存到GPU的DMA、Python图/对象/事件管理及编译启动所需驱动API。
这些保留CPU/驱动实现，不将其中的GPU数学计算一并豁免。通信bootstrap可用
CPU rendezvous，正式GPU张量交换必须检查实际路径。

研究依据：
[PyTorch2.10 Symmetric Memory](https://docs.pytorch.org/docs/2.10/symmetric_memory.html)、
[Triton inline PTX接口](https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html)、
[NVIDIA PTX内存一致性](https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-consistency-model)。
本机Torch2.10的FSDP与fused AdamW源码为数值/布局参考；新实现独立编写。
Alpha接口及跨GPU系统作用域可用性必须在本机SM86四卡实测，不能从文档推出。

## 当前完成的组件检查（不代替完整8B验收）

`adamw_check_002`：四rank各完成128次连续更新，包含非零m/v、不同参数组、
跳过梯度、64步后的CPU状态恢复及原地版本计数。1/17/64/128步的全部检查点
参数和m/v与native fused AdamW逐位一致；另覆盖官方8B的norm、Q、MLP及
`[37984,4096]`词表分片形状。独立CPU oracle通过，候选专用trace只出现
`_adamw_prepare`和`_adamw_update`。这不是8B训练已采用或optimizer提速结论。
首版检查因FMA次序带来细小m差异而失败；保留失败，不放宽逐坐标门槛。

`collectives_check_002`：四rank各112项dtype/操作/尾块检查及32次更换数据；
跨AG/RS stream的不同主机入队次序、粘性无损错误及缺失rank的超时负例通过。
真实192946432元素decoder桶的每rank全部48236608个FP32输出，与独立CPU
FP64 oracle逐位一致。bootstrap为CPU Gloo；候选trace只有4次stage、8次
Triton屏障和4次peer collective，没有NCCL或CUDA barrier kernel。
首轮因Triton不支持Python `break`的编译失败已保留，改为有界循环条件后通过；
不能把该错误解释成硬件不支持。

`fsdp_check_002`：FSDP＋严格GPU操作替换＋Triton AdamW的小模型集成检查通过。
四rank各覆盖forward后保留/重新分片两种策略、每种4次更新。逐坐标比较输出、
输入梯度、独立CPU FP64平均的参数梯度，以及native AdamW更新后的参数与m/v。
候选完整forward/backward/optimizer/训练统计trace只包含Triton kernel，包含
打包、拆分、转换、初始化、累加和通信；没有native CUDA计算/NCCL/barrier。
首次集成因Triton不支持dtype的Python `in`表达式而编译失败，改为显式constexpr
分支后通过；失败保留。保留的是FSDP主机状态/事件调度，不是它的GPU计算实现。

上述组件检查阶段，完整36层8B接入、全399梯度、512块验证、完整step trace
和整体/部分消融均待完成；后续真实8B进展见本文末尾。
组件记录在原始 `artifacts/triton_qwen3_runtime_v14`，小型摘要见
[组件证据](results/qwen3_triton_runtime_v14_components_v1.json)。

后续 `adamw_check_003` 已通过相同四rank128步及官方8B分片检查；新增最大
`[37984,4096]` 分片的CPU检查点恢复，确认m/v/step使用原GPU缓冲区，恢复后
继续更新仍与native fused AdamW逐位一致。该结果覆盖有界恢复实现，不代表
整模型检查点系统已实现，也不代表完整8B训练已验收。

完整入口为 `scripts/train_qwen3_runtime_v14.py`，默认 `--runtime full`。
严格模式现在覆盖FSDP初始化；第一次8B试跑暴露的 `new_zeros` 已补齐，原失败
保留。新的试跑仍需通过实际全399梯度和初始化/首次更新/整步trace检查。
验证使用GPU FP64 NLL累加，显存峰值按初始化、验证、训练及更新后检查分别
记账。完整验证预检查另外发现FP64除法接口仅支持FP32，已改用libdevice的
FP64正确舍入除法；FP32/FP64及tensor/scalar右操作数四种SM86预编译通过，
尚待完整验证GPU实测，不将预编译当作数值验收。

已提供512块验证可行性运行器、固定四组正逆序消融运行器及CPU审核脚本。
这些脚本仅通过语法/导入预检查；仍须运行完整8B并审核真实结果，不能从脚本
存在推出验收通过。现阶段不启动未通过前置可行性检查的正式性能采用流程。

`full_pilot_002` 的两次GPU尝试均被资源监控确认为外部进程污染，没有接纳
任何完整训练结果。其中第二次已导出的真实8B初始化trace单独检查通过：
804次kernel调用全部为Triton（399次strided copy、398次typed fill、7次fill）。
该证据只覆盖初始化；不足以证明全训练完成。等待下一次空闲的自有控制器
已在确认无worker后停止，以纳入上述已知修复及无损gzip trace存储，原因和
原始尝试记录保留。未按计时挑选或丢弃任何干净的完整尝试。

新trace使用profiler原生gzip导出，保留全部原始事件；审核同时记录压缩文件
SHA和解压后原始字节SHA，不重写或过滤trace。已用真实8B初始化trace验证
无损解压和相同kernel审核结果；验证阶段还将核对B4/rank对应的实际GEMM
launch geometry及32次GPU FP64统计累加。

## 最近完整8B检查暴露的未完成项

`statistics_check_001`：四rank各32组不同loss/token权重，与独立CPU FP64累加
及最终除法逐位一致；零有效token的NaN不参与累计，FP64工厂填充值和确定性
非连续empty初始化均通过。实际trace没有原生GPU memset或设备间拷贝。

`full_pilot_003`：第一个资源干净的尝试完成5次真实更新，FP32状态和有效显存
门槛通过（各rank全阶段最大reserved含外部预算41.87109375GiB）。rank0/1/2
全399梯度通过；rank3的 `model.layers.6.self_attn.k_norm.weight` 相对误差
0.061313974258671126，绝对误差0.0003133364662062377，大于原门槛
0.00030762158504128457。因此完整数值验收失败，保留第一次干净结果，不
自动重跑或放宽阈值。正在按真实词表CE及dX形状诊断FP32归约顺序。

更完整的trace事件审核还发现原Symmetric Memory allocator内部的8次
`gpu_memset`，它们不是kernel事件，旧的kernel白名单检查未能覆盖。因此
前述“804次初始化kernel均为Triton”**仅证明kernel种类**，不足以证明初始化
全数据路径完成；历史组件/进度记录不能据此推导整个运行时已验收。
[Torch2.10源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.cu#L400)
在allocator里直接调用 `cudaMemset`，没有在这里将其归类为主机分配豁免。

新增 `runtime/peer_memory.py` 仅用CUDA Driver API做CPU侧allocation/IPC mapping，
通过Gloo交换CPU句柄；payload和mailbox初始化均显式执行Triton。没有全局
CUDA拦截或PyTorch修改。主机负责生命周期：所有GPU使用结束后，每rank先
关闭import，CPU协调确认后owner才能free。`collectives_check_003` 已完成该路径
四GPU通信检查：每rank112项、32次换数据、不同stream入队、无损粘性错误和
超时负例，以及真实大桶全部48236608坐标的独立CPU FP64 oracle均通过。
初始化trace只有9次Triton fill，没有任何原生GPU memset或memcpy；显式IPC
teardown完成。该证据仍不代表完整8B数值/验证已通过，详见
[IPC组件证据](results/qwen3_triton_runtime_v14_peer_memory_v1.json)。
完整trace审核已扩展为拒绝所有原生GPU memset和设备间
memcpy事件，保留CPU↔GPU DMA。组件和失败证据摘要见
[最新进度](results/qwen3_triton_runtime_v14_progress_v3.json)。

`reduction_order_probe_001` 对真实head dX大小8388608个FP32输入及词表CE
4088/16352元素做原生NCCL与Triton比较。dX在rank0/1/2有112/100/117个
BF16舍入结果不同；rank3相同。独立CPU枚举加法树确认原生RS按每rank的
循环顺序相加，全部2097152输出坐标均匹配该顺序。现已只调整任意FP32
输入的SUM ReduceScatter顺序；BF16无损梯度线和AVG路径保留原算法。
四rank FP32/BF16分支预编译通过；`reduction_order_probe_002` 已实测确认每rank
全部2097152个FP32输出与原生NCCL逐位一致，之前的BF16舍入差异全部消除，
详见[归约顺序证据](results/qwen3_triton_runtime_v14_reduction_order_v1.json)。
`full_pilot_004` 正在按新源码冻结重做8B检查；仍须全399梯度及完整质量验证
证明整个模型的修复有效，不能从该组件逐位一致推出完整验收。
该阶段的FP32 AllReduce仍有加法顺序差异；后续诊断和修复如下，门槛始终不变。

`full_pilot_004` 完成5次真实更新后，初始化、首次更新、完整step的扩展GPU
审核全部通过，包括没有原生GPU memset/设备间拷贝；各rank显存门槛也通过。
但rank1第8层K Norm梯度相对误差0.0635948656630924，仍未通过原门槛，
其他rank全399梯度通过。首次干净失败和全部原始证据保留，详见
[失败记录](results/qwen3_triton_runtime_v14_full_screen_failure_004.json)。

`reduction_order_probe_003` 捕获实际NCCL `RING_LL` 内核，并用独立CPU加法树
枚举定位到AllReduce各段的顺序：训练CE4088元素使用1channel、每rank1024
元素（尾段1016）；验证CE16352元素使用2channel、总长度8192/8160，分别
按2048/2040元素分段。候选现在对这两个经过确认的4×A6000实际形状保留该
顺序，仍只执行Triton；通用dtype/形状路径保留原语义。当前正在用3组不同
输入复核全部坐标，尚待新8B梯度及完整质量验证。此为固定硬件/协议的数值
边界对齐，不声称任意GPU/消息大小与NCCL逐位一致，不修改原有数值门槛。
[原始诊断摘要](results/qwen3_triton_runtime_v14_AllReduce_order_diagnosis_v1.json)，
[对应NCCL源码](https://github.com/NVIDIA/nccl/blob/v2.27.5-1/src/device/all_reduce.h)。

`reduction_order_probe_004` 已确认：四rank的两种CE AllReduce形状各3组不同
输入、以及8388608元素dX ReduceScatter全部坐标与原生NCCL逐位一致，见
[通信顺序证明](results/qwen3_triton_runtime_v14_collective_order_proof_v1.json)。

`full_pilot_005` 完成首次干净的5次真实8B更新：四rank全部399参数梯度通过，
完整首步梯度比较记录与历史已验收V13相同。FP32参数/梯度/m/v、实际更新
计数、有效显存、初始化/首次更新/完整step的所有GPU事件检查均通过；初始化
808次kernel均为Triton，且没有原生GPU memset/设备间拷贝。前5个全局loss与
V13最多相差一个FP32舍入单位，不能据此声称完整状态逐位相同。
[真实8B试跑证据](results/qwen3_triton_runtime_v14_full_pilot_v1.json)。
正在运行 `full_validation_screen_001`：仍为5次更新，但在0/1更新后分别覆盖
全部512块、261632 shifted tokens、每卡验证B4，并检查完整验证GPU路径。
正式16质量更新/28实际更新、四组两轮消融和独立审查尚未完成。

`full_validation_screen_001` 的首次干净GPU结果已完成：0/1更新后各验证全部
512块、261632 shifted tokens、每卡B4；有optimizer状态时的有效allocated/
reserved验证峰值为38.662/40.877GiB（含512MiB外部预算），没有缩小batch或
数据覆盖。全399梯度、FP32状态、全部阶段显存与GPU事件/形状检查均通过，
见[完整验证可行性证据](results/qwen3_triton_runtime_v14_validation_feasibility_proof_v1.json)。

原CPU审核曾因AG计数预测少两次而拒绝结果；原始GPU运行、原审核状态和
attempt history均保留。Torch2.10在no-grad forward中仍记录post-forward顺序，
直到backward结束才清理；因此每次验证后的首次训练backward额外误预取一次
final norm。实际trace的模块注解也明确显示这一次额外prefetch，故两个验证
阶段总共应计202次FSDP AG（200次验证forward＋2次训练时预取），不是200。
只修正CPU审核并重新审核同一GPU结果，没有GPU重跑、没有放宽数值/显存阈值。
原生HF和V13控制的首次训练trace也按这个原生FSDP行为计入额外一次AG。

`formal_001` 已冻结源码并开始四组正逆序完整对照：HF native、V13、
V13＋Triton AdamW、full runtime，再逆序。每角色16质量＋3预热＋8计时＋
1真实profile更新＝28次实际更新，验证点0/16各512块。全部8组及CPU独立
审查尚未完成，因此目前不能声明完整V14数值验收或整体性能采用。

## 完整8B验收约定（在本轮8B计时之前声明）

沿用官方权重/语料身份、4×A6000、每卡B1/S512、BF16计算、FP32 master/梯度/
moments及41.875GiB有效预算。全399参数门槛、16步质量阈值与初末512块全量
验证（每卡验证B4）不降低。比较原生HF、V13、仅替换AdamW、完整GPU运行时，
固定顺序再逆序，各角色接纳第一个完整且资源干净的尝试，保留所有失败。

每角色16次质量更新＋3次预热＋8次计时，另做**1次真实完整profile更新**，
共28次实际optimizer更新。新增profile必须包含optimizer和统计；它不进入8个
计时样本。不能沿用V13“trace不更新参数、仅前后向”的27次更新口径。
同时检查首次更新（含optimizer状态初始化）及完整运行时的验证GPU路径，
避免只审查已有state的稳态训练。所有实际GPU计算kernel必须逐项识别为Triton；
不得把Fill/Copy/gradient accumulation一并列为“framework”免检。

完整Triton数据通路的正确性/覆盖验收与性能采用分开：实现即使较慢也要如实
报告，不能退回原生GPU计算而继续称完整。性能推荐仍需双方相同协议的两轮
整步计时支持；不从组件检查推断8B速度。新V14训练入口将默认完整运行时，
历史入口/历史发布分支保留便于比较；当前文档首页仍待完整8B验收后更新。
