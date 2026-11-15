# Qwen3-8B：尽可能完整的 Triton 训练数据通路

状态：实现中，尚未完成完整8B验收，不作为性能推荐。基于已验收的 V13
`95a27d0`，独立分支 `research/triton-runtime-v14`；历史分支和默认入口不改。
所有输出、缓存、临时文件位于 `/data1/tzh`，不写 `/home/tzh`。

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

完整36层8B接入、全399梯度、512块验证、完整step trace和整体/部分消融均待完成。
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
