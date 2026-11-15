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
