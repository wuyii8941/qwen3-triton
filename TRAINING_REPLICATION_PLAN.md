# 补强完整训练证据：三方对照与多数据顺序种子

状态（2026-11-02）：`training_replication_v1` 在 compiled 评估预检失败后停止，未启动正式训练。
修正后的队列使用 `artifacts/training_replication_v2/`，失败记录保留。
两种长度、三种后端的实际数值/保存恢复预检均已通过，首个正式 compiled HF 作业已启动。
完整 GPU 回归 **278 passed，2 warnings**，证据为 `artifacts/training_replication_regression_v2.xml`。
这里不是已完成的性能结论。
历史两组结果保留在 `convergence_comparison_v1/`，不混入本轮重复统计。

## 预先固定的实验矩阵

| 长度 | 种子 | 后端 | 正式作业数 |
|---|---|---|---:|
| 512 | 20261102、20261103、20261104 | Native / Compile / Triton | 9 |
| 2048 | 20261102 | Native / Compile / Triton | 3 |

全部在 GPU0 顺序执行。S512 采用三次轮换顺序：Compile→Native→Triton、
Native→Triton→Compile、Triton→Compile→Native。每种后端各占一次首/中/尾位置，
降低固定先后顺序的影响；这不能消除所有温度、系统 I/O 或缓存变化。
S2048 是单种子的长度扩展，不冒充三种子复现。

种子真正改变训练 token block 的排列，验证/测试顺序不变；模型始终加载同一预训练权重。
旧 v1 的 block 排列写死 20261102，本轮已修正，并测试不同种子的顺序确实不同、恢复顺序不变。
默认种子保持原有行为，不改历史文件。

## 三种实现到底比较什么

- Native：HF Qwen3 forward/backward + `AdamW(fused=True)`。
- Compile：`torch.compile(HF loss, fullgraph=True, dynamic=False)`，包含 AOT backward；
  optimizer 仍是相同的 native fused AdamW。并未宣称 optimizer/Python 数据循环整体单图。
- Triton：原生 HF forward/backward + Triton AdamW rounded。

相同 BF16 参数/梯度/动量、无 FP32 master、B=1、lr=1e-5、Flash SDPA、use_cache=False。
不修改 optimizer 超参数、loss 定义、数据或精度去帮助某一种后端。
Compile 出现 graph break 必须失败，不静默回退；最终 summary 保存 compile counters。
三方验证/测试统一使用原生 HF eager。当前 Transformers 4.51 在 compiled eval 时
构造显式 mask，与强制 Flash SDPA 不兼容；v1 预检已记录该失败。
没有将 `suppress_errors` 打开，也没有在 compiled 训练中静默回退。
首次训练编译、反向编译和重编译都计入完整作业；不声称验证/测试也被编译。
每个正式进程使用独立 Inductor/Triton cache 目录，不沿用预检的编译 cache。
这覆盖冷编译成本，但不是事先清空 OS/驱动所有缓存的“绝对冷启动”。

## 预检门槛

两种长度、三种后端各实际保存第2步 checkpoint，再恢复到第4步。
以 Native 相同步数的实际 loss 为参考：初始验证差≤0.002、训练 loss 差≤0.03、
恢复后的验证 loss 差≤0.01。门槛已在 `plan.json` 冻结。
检查必须覆盖相同事件/步数，任何失败会停止整个队列，保留日志，不自动调宽门槛。
这是启动门槛，不是长程质量等价证明；长程曲线和独立 test 仍需单独报告。

## 收敛与计时

- 继续采用验证 loss 平台：min_delta=0.002、patience=8、至少10次验证。
- S512 每250步验证；S2048 每64步验证，使相邻验证的输入 token 数约为128K/131K。
  不跨 sequence length 直接比较 loss 或把两种工作负载的 speedup 混成一个数字。
- 每组6小时安全上限。触及时只记 `not_converged`，不当作收敛。
- parent 在 subprocess spawn 前开始计时，直到进程退出；包括 imports、加载、
  数据传输、训练、验证、durable checkpoint、最终两份 checkpoint 测试及退出清理。
- parent 同步观察日志，记录实际 crossing 观察时间，0.25秒轮询及系统调度带来误差。
  observer 不创建训练输出目录，避免旧版交接竞争；resume 不重新标记历史事件时间。
- 共享语料准备仍然排除且未计时。训练/编译/数据缓存和 checkpoint 均在独立实验目录。

每个种子/长度的目标按同一冻结规则确定：Native 最低验证 loss + 0.002，
对三方取第一次达到目标的实际验证点，目标未达到则保留缺失，不更换阈值。
报告同时给 Native 对 Compile、Native 对 Triton、**Compile 对 Triton** 的配对耗时。

三种子完成后报告各 seed、几何均值、中位数、最小/最大值。范围不叫置信区间，
只有三种子仍不足以作广泛生产保证；不丢弃不利 seed、不选择性隐藏失败/未收敛作业。

## 文件和查看命令

- `benchmarks/run_training_replication.py`：预检、独立 cache、冻结矩阵、顺序调度、父进程计时。
- `benchmarks/summarize_training_replication.py`：部分/完整结果审计及三方配对统计。
- `artifacts/training_replication_v2/plan.json`：矩阵、容差、源码 SHA256。
- 每个阶段 `<name>.log`、`<name>.clock.jsonl`；正式阶段另有协议、checkpoint、事件及终态 summary。
- 队列根 `summary.json` 随正式作业结束更新，不代表所有作业完成；检查 `all_jobs_terminated`
  和 `all_jobs_converged`，以及 aggregate 的实际 seed 数。

```bash
cd /data1/tzh/proj/qwen_operator_lab
# 只读查看阶段；文件存在不等于对应进程还活着。
ls artifacts/training_replication_v2
tail -n 5 artifacts/training_replication_v2/preflight_s512_compile.log

# 汇总已产生的结果，不会启动/重启训练。
PYTHONPATH=. /data1/tzh/proj/systems/.venv/bin/python \
  -m benchmarks.summarize_training_replication --root artifacts/training_replication_v2
```

完整12组预计需要约一天，实际取决于平台停止步数、编译和 I/O。
少于50GiB空闲时不再启动下一组，不删除历史结果来腾空间。
现有默认 HF 接口不变，环境未安装新包，未修改宿主 Python。

## 运行期资源监控与 checkpoint 保留

用户授权在空间不足时适当清理 checkpoint 后，已启动独立 watchdog，每60秒检查
训练队列、GPU显存、主机可用内存和磁盘空间，快照为本轮目录下 `watchdog.json`，
历史为 `watchdog_history.jsonl`。监控器不改冻结的训练源码/协议，不重启训练，不自动调 batch。

空闲磁盘低于100GiB时，仅允许删除本轮六个明确列出的预检目录中的 `last.pt`/`best.pt`，
前提是该长度的数值门槛已通过、恢复到第4步的预检已正常退出。不会扫描全工程或递归删除。
预检日志、协议、gate证据保留；正式作业的last/best全部保留。删除意图和逐文件结果写入
`checkpoint_pruning.jsonl`；删除的预检权重无法直接恢复，须重新运行预检生成。
若仍不足，不扩大删除范围：原队列已有50GiB的下一组启动保护。

checkpoint占磁盘，不是显存。显存/主机内存异常仅记录，不通过删除checkpoint冒充解决，
也不擅自改变训练工作负载。自动清理如发生，时间记录可用于检查其对测量的潜在I/O影响。

## 独立 FA2 对照筛查（2026-11-03，已执行）

用户要求另测官方独立 FA2 与当前 PyTorch Flash SDPA。新建 `.venv-fa2`，
通过本环境的 `.pth` 只读复用原 PyTorch 2.5.1+cu121/Transformers 4.51依赖，
额外包只安装在新环境，不升级原训练环境、不改宿主Python、不改本轮冻结协议。
用户提供的 `127.0.0.1:27890/27891` 两个代理均可访问官方源，下载问题已解决。
默认环境代理不是这两个端口，之前的失败记录保留。官方2.8.3 wheel完整下载且SHA256匹配
release摘要，但要求glibc2.32，宿主为2.31，导入失败；没有改系统库或使用符号shim。
随后在新环境安装官方 **2.7.4.post1** wheel，Python3.12/Torch2.5/CUDA12/ABI=False，
导入和GPU运行成功。原环境再次确认没有 `flash_attn`。不将本次结果标为最新版FA2。

安装包位于`artifacts/fa2_install_v1/`。2.8.3的官方SHA256为
`51f6422861ed951b968428adc9fa7406027f73f2145be5e163810df6f459abea`；
2.7.4.post1的本地SHA256为`a496f383c843ed0cc6e01302556a0947f56828f1ec2b54e5297c4c7eaae39357`，
对应官方release的187814472字节asset（该旧asset API未提供digest，不能声称对过官方哈希）。

`benchmarks/benchmark_fa2_comparison.py` 已实现两个独立模式：

- operator：Qwen head_dim=128、Hq=16/Hkv=8，FP16/BF16，输出和随机upstream的dQ/dK/dV
  对比FP32 math oracle；三轮打乱顺序比较forward和forward+backward及峰值增量显存。
  SDPA 包含 HF 的 repeat_kv，FA2 使用紧凑 GQA，因此该收益不能全归为纯kernel差异。
- model：真实28层Qwen、BF16/B1，同一权重先比较loss和全部参数梯度，再以相同原生fused AdamW
  做三轮交错顺序的训练step测试，每种后端每轮恢复原始权重；不保存新大checkpoint。
  这是固定GPU输入step测试，排除warmup，不等同于完整训练作业/收敛收益。

### 实际结果与停止原因

两项CPU辅助测试通过；实际GPU结果保存在`artifacts/fa2_screen_v2/`。
v1仅记录2.8.3导入失败。v2六组算子（B1/S512、B1/S2048、B2/S512，各FP16/BF16）
输出与随机upstream的dQ/dK/dV均通过FP32 oracle门槛。三轮forward+backward的中位数：

| B/S/dtype | HF Flash SDPA ms | FA2 ms | SDPA/FA2 |
|---|---:|---:|---:|
| 1/512/BF16 | 1.0957 | 1.1984 | 0.914× |
| 1/512/FP16 | 1.4800 | 1.9881 | 0.744× |
| 1/2048/BF16 | 1.2897 | 1.7389 | 0.742× |
| 1/2048/FP16 | 1.2119 | 1.6179 | 0.749× |
| 2/512/BF16 | 0.9364 | 1.0291 | 0.910× |
| 2/512/FP16 | 1.5754 | 1.9193 | 0.821× |

B1/S512/BF16的forward-only为0.2007→0.1751ms，但forward+backward未获益。
这些是共享GPU3上的CUDA Events测量，包含dispatch/launch间隙与HF repeat_kv成本，
不是隔离GPU的纯kernel时间。其他进程负载在阶段前后变化，记录于每个stage JSON；
不能据此断言FA2普遍更慢，也不把forward-only优势外推为完整训练加速。

真实模型BF16/B1的梯度门槛为全参数相对L2≤2%，loss绝对差≤0.01。
S512初测loss差0.000145、全参数梯度差19.63%，门槛失败，**训练step耗时测试立即停止**，
没有放宽门槛或产出完整模型加速结论。追加不更新参数的三次反向诊断：

| 长度 | SDPA重复反向相对L2 | FA2对SDPA相对L2 | loss绝对差（FA2） |
|---|---:|---:|---:|
| 512 | 1.253% | 19.630% | 0.000145 |
| 2048 | 0.911% | 11.667% | 0.000496 |

S512诊断在GPU3，S2048诊断在当时更空闲的GPU2，都是A6000；各自的重复控制与FA2比较
在同一模型/同一设备上进行，不混合做跨GPU性能比较。诊断文件为`model_parity_s*_v1.json`，
包含310个参数张量的全局比较和误差最大的参数名。SDPA重复控制也并非逐位确定，
但本次FA2差异显著更大，不只靠相同基线的一次重复波动即可解释。

结论是**没有通过当前BF16严格等价筛查，因此不接入现有长程训练**；
不等于证明FA2算法错误或一定无法正常收敛。若继续，需要定位真实激活下的逐层梯度/
舍入敏感性，以及独立控制负载的计时，而不是跳过门槛直接宣传收益。

```bash
# 独立环境已经安装FA2。新结果使用新文件名，不覆盖历史证据。
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. .venv-fa2/bin/python \
  -m benchmarks.benchmark_fa2_comparison --mode operator --seq 512 \
  --dtype bf16 --iterations 30 --output artifacts/fa2_operator_s512_bf16_v1.json
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. .venv-fa2/bin/python \
  -m benchmarks.benchmark_fa2_comparison --mode model --seq 512 \
  --iterations 12 --output artifacts/fa2_model_s512_v1.json
# S2048、FP16以及B>1算子测试使用新输出文件名，禁止覆盖历史结果。
# 当前model模式预计会被严格梯度门槛阻止，不会生成step性能数字。
```

## 新算法独立筛查：FlashOptim、CCE/Liger、Muon/NorMuon（2026-11-03）

用户授权后，新建 `.venv-research`，只读复用本机现有 Torch2.10.0+cu128/Triton3.6.0，
在新环境固定 Transformers4.51.0、tokenizers0.21.4、HF Hub0.36.2。
新安装 flashoptim0.1.4、cut-cross-entropy25.1.1、liger-kernel0.7.0。
原 Torch2.5.1 长程训练环境与冻结源码/协议不变，不将框架升级收益归给候选算法。

源码：`benchmarks/research_candidates.py`、`screen_research_candidates.py`、
`run_research_screen.py`、`summarize_research_screen.py`。
结果与冻结设置：`artifacts/research_screen_v1/plan.json`、`status.json`、`initial_report.json`。
两项CPU辅助测试验证 tied embedding/head 不重复分配、只对hidden矩阵用Muon，以及状态storage去重。

### 固定协议

- GPU1、BF16参数与梯度、无FP32 master、B1/S512、同一预训练Qwen3-1.7B。
- FlashOptim显式`master_weight_bits=None`，不启用gradient release，不调用会改变norm/head精度的cast_model。
- CCE显式`filter_eps=None`，关闭梯度过滤；Liger使用默认FLCE。不能把它们当逐位等价。
- Muon/NorMuon使用作者的SingleDevice版本，hidden二维矩阵用正交化更新；
  tied embedding/head和norm等用同一个native fused AdamW，aux lr固定1e-5。
  NorMuon官方repo commit为`c6989a8354730695d9f5a9faa6c55eeb24865209`；
  Muon因git握手失败改为下载官方raw文件，其精确SHA256保存在plan中。来源见下方链接。
- 原生AdamW lr网格3e-6/1e-5/3e-5；Muon和NorMuon各1e-4/3e-4/1e-3；FlashOptim固定1e-5。
  这不是联合搜索aux lr的最优性证明。所有轨迹64步，相同训练块顺序、seed20261106。
- 固定验证集前16个完整block，每8步检查；早期目标是初始loss减0.15（本次2.64165）。
  没有独立test成绩，不是完整收敛实验；未用test调参。
- 每轮初始loss相同，为2.7916498333。多数候选首次第8步检查已达早期目标，分辨率不足，
  不据此宣传训练步数减少。完整曲线、首次到达和主进程计时均保留。
- 没有新增大模型checkpoint；优化器保存恢复仅在约12.6M参数合成模型上用内存BytesIO测试。
  共享GPU有其他进程，微小耗时差异不作生产性能结论。

### 已完成的初筛

真实词表V151936、H2048、N512/2048的loss/随机梯度均通过算子门槛。
但完整模型BF16/S512全局参数梯度检查：CCE约6.23%，未过2%门槛，所以不执行其训练性能测试；
Liger约1.37%，原生重复控制约1.30%，按本轮全局门槛通过。
该门槛是全局L2，不是所有单个参数都小于2%，不冒充全面数值保证。

| 候选与lr | 稳态step中位数ms | peak allocated GiB | 第64步验证loss |
|---|---:|---:|---:|
| Native 1e-5 | 383.62 | 14.00 | 2.35360 |
| FlashOptim 1e-5 | 299.34 | 10.99 | 2.35369 |
| Liger FLCE 1e-5 | 1289.64 | 13.58 | 2.35622 |
| Native网格最佳 3e-5 | 377.96 | 14.00 | 2.21698 |
| Muon网格最佳 1e-3 | 1670.06 | 11.37 | 2.25242 |
| NorMuon网格最佳 1e-3 | 1702.69 | 11.37 | 2.23937 |

主函数计入加载/JIT/验证的64步耗时：Native1e-5为38.65秒，FlashOptim为52.57秒。
因此初次冷启动FlashOptim并没有总耗时优势，不能只报告稳态数字。
在合成优化器测试中，首次FlashOptim step约8.36秒（含JIT），native约54ms；
稳态约0.957/1.418ms。状态张量表示约26.7/50.3MB，含模型的序列化快照约51.9/75.5MB。
保存恢复后下一步参数完全一致，但FlashOptim更新与native更新的相对差约4.1%，
压缩并非精确重写，短期loss接近不等于长期无损。

本轮网格训练进程总成本：Native约124秒，Muon约384秒，NorMuon约389秒，FlashOptim约58秒。
不能只挑最佳学习率那一次耗时而忽略搜索成本；以上仍不包含环境准备和其他预检。

初筛决定：CCE保留数值反例；Liger在本配置未显示吞吐优势；Muon/NorMuon在这个小batch持续
预训练任务与冻结网格下未胜出，不外推成这些优化器普遍无效。
FlashOptim值得下一阶段验证，但需单独处理JIT摊销、更多种子及完整验证集收敛。

### FlashOptim重复确认

`benchmarks/confirm_flashoptim_screen.py` 在同环境/GPU1做三轮native↔FlashOptim交错进程测量，
相同lr=1e-5、每轮重新加载权重、64步固定数据、验证曲线最大abs差≤0.01。
证据在`artifacts/flashoptim_confirmation_v1/`。明确复用初筛生成的磁盘JIT cache，
不是再次冷启动：首次冷结果另列在上面，不能混为一组。
这仍是短程确认，不是“完整训练到收敛加速”的证据。

三轮已经实际完成，验证曲线最大loss绝对差分别0.00444/0.00192/0.00073，均通过0.01门槛：

| 进程轮次 | Native稳态step ms | FlashOptim稳态step ms | Native/FlashOptim | 完整64步进程耗时比 |
|---|---:|---:|---:|---:|
| 0 | 362.34 | 205.93 | 1.760× | 0.893× |
| 1 | 326.63 | 189.67 | 1.722× | 0.952× |
| 2 | 303.14 | 189.56 | 1.599× | 0.919× |

配对step speedup中位数1.722×，约对应稳态耗时减少41.9%；各轮方向一致，但共享卡波动明显，
不能只取最快一轮。peak allocated稳定为14.00→10.99GiB，状态张量表示6.41→3.41GiB。
**包含整个进程启动/加载/验证的64步耗时仍慢约5–12%**，中位数约慢8.8%。
即使复用磁盘cache，每个新进程的首次FlashOptim step仍约9.9–14.1秒，native约0.41–0.48秒。
这同时包含状态初始化、模块加载/JIT等成本，不能全归因于编译，也不能从总耗时中删掉。

因此采用决定仍为：保留为独立候选，不改变在跑的长期训练。下一步如继续，应先在相同
Torch2.10环境直接对比**我们现有Triton AdamW与compiled HF**，再做含完整验证、真实checkpoint
和独立test的长程收敛比较。这里尚未证明FlashOptim优于我们的现有最优组合。

来源：[FlashOptim](https://github.com/databricks/flashoptim)、
[CCE](https://github.com/apple/ml-cross-entropy)、[Liger](https://github.com/linkedin/Liger-Kernel)、
[Muon](https://github.com/KellerJordan/Muon)、[NorMuon作者实现](https://github.com/zichongli5/NorMuon)。

### 梯度搬运与多张量更新初筛（2026-11-05）

脚本：`benchmarks/gradient_transport_screen.py`；原始记录：
`artifacts/gradient_transport_screen_v1/{embedding,dx,optimizer}.json`。
研究环境 Torch 2.10 / Triton 3.6，GPU0，BF16。每项预热3次、测15次，
下表为同步后的 host wall-clock 中位数，包含 Python 调度；不是纯 CUDA kernel 时间。

| 独立微实验 | baseline ms | candidate ms | 数值检查 | 决定 |
|---|---:|---:|---|---|
| tied embedding 梯度合并，V=151936,D=2048,N=512 | 3.896 | 2.027 | 全量梯度 relative L2 0.218% | 保留候选，未接入模型 |
| MLP dX，两个 GEMM+add → GEMM+addmm | 0.308 | 0.313 | relative L2 0.232% | 无加速，不采用 |
| pointer-table 多张量 BF16 AdamW | 3.139 | 0.812 | 4步参数及一阶矩检查误差0 | 保留原型，未接入模型 |

embedding 候选包含 unique 分组、FP32 行归约及 dense clone 成本；增量峰值分配
1190→598.3 MiB，减少约591.7 MiB。仅为合成 tied-gradient 合并实验，
ID 来自256个词的重复采样；尚未覆盖真实 token 分布、完整 autograd/compile 集成。
全量 L2 会被已有 dense LM-head 梯度稀释，后续必须单独检查命中行及新增梯度，
不能凭此宣布训练梯度对齐。dX 的 BF16 rounding 边界也有变化。

多张量原型比较的是旧 `TritonAdamW`，72个参数张量、约3370万元素，
包含每步梯度指针表上传，不拼接梯度。**不是 FlashOptim 量化状态实现**，
不能用3.86×局部结果声称优于当前 compiled HF + FlashOptim。
目前只验证4步参数和一阶矩，二阶矩及长程训练仍需补验。

三项都没有修改默认训练路径，没有启动长训练或新增 checkpoint。
下一步优先验证 embedding 命中行误差及真实模型梯度；多张量更新需先保留
FlashOptim 原有逐参数量化边界，再与当前最优训练组合比较。

### 共享 embedding/head 梯度合并：真实模型验证

实现与入口：`benchmarks/check_tied_embedding.py`、
`benchmarks/benchmark_tied_embedding_step.py`；独立测试
`tests/test_tied_embedding_transport.py`（9/9通过）。
自定义 autograd Function 同时返回 embedding 输出及共享权重的 view，
反向接收 embedding upstream 与 LM-head dense gradient；稳定排序输入token，
每个命中词表行由一个 Triton program 做 FP32 累加，再 cast/add 到 dense clone。
避免独立的整张词表 embedding 梯度及之后的全量加法；无原子更新，保留参数共享。
当前仅支持普通 dense embedding、无 padding_idx/max_norm/频率缩放，以及无 bias LM head。

首次非确定性梯度门槛未过（最大参数5.61%，原版重复3.70%）；保留
`artifacts/tied_embedding_model_v1/result.json`，不隐去失败结果。
使用确定性算法与 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 复查，不放宽门槛：
原版重复误差0，候选 loss 完全一致，全量梯度relative L2 0.0531%，
共享权重0.2413%、命中行0.2507%，其余参数梯度完全一致。
见 `artifacts/tied_embedding_deterministic_v1/result.json`。
独立测试另行检查仅 embedding 新增贡献，避免 dense 背景稀释误差；
覆盖FP16/BF16/FP32、重复token、seq=1/15/129、dim=64/128/130。

最终性能以 **每组独立进程** 的
`artifacts/tied_embedding_step_v2/result.json` 为准；v1同进程切换模型存在显存残留，
不作为绝对显存依据。RTX A6000，Qwen3-1.7B，BF16，B=1,S=512，
compiled HF + 原有 FlashOptim，无 master weight、不改变optimizer；性能模式为非确定性。
每组重新加载相同预训练权重、20步固定同一batch，前4步不计入稳态中位数。

| 交错轮次 | baseline step ms | 合并后 step ms | 耗时减少 | peak allocated GiB（前→后） |
|---|---:|---:|---:|---|
| 0：baseline→candidate | 104.406 | 100.713 | 3.54% | 11.140→10.429 |
| 1：candidate→baseline | 104.582 | 101.088 | 3.34% | 11.140→10.429 |

两轮均减少728.10 MiB峰值分配；20步loss轨迹最大绝对差分别0.00338/0.00235。
这是完整 forward/backward/optimizer step 的局部稳态收益，不是收敛加速证据。
固定batch来自验证语料，仅用于性能负载，更新后的权重不保存；这些loss不作为泛化指标。
**首次step更慢**：baseline约18.5秒、candidate约37.3–37.6秒，包含编译/加载与
optimizer初始化，且磁盘JIT缓存已复用；不是纯编译计时，也不是全流程冷启动时间。
因此20步总耗时并未获益；若粗略以额外19秒和每步省约3.5ms估算，需约五千多步摊销，
这不是实测长程盈亏平衡点。默认训练仍不启用，尚未做长程验证与更多shape sweep。

复现（项目研究环境，不动系统Python）：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv-research/bin/python -m pytest tests/test_tied_embedding_transport.py -q
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. .venv-research/bin/python -m benchmarks.check_tied_embedding --tag NEW_UNIQUE_TAG --deterministic
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv-research/bin/python -m benchmarks.benchmark_tied_embedding_step
```

性能脚本为防覆盖已有结果，会拒绝重建已有v2目录；重跑应先修改该脚本的artifact tag。
它读取已存在的确定性v1门槛结果，未通过则拒绝运行。

### 系统优化筛选与训练接入（2026-11-05）

本节为新一轮完整筛选，不覆盖旧结果。研究环境 Torch2.10.0+cu128、Triton3.6、
HF4.51、FlashOptim0.1.4，A6000、真实Qwen3-1.7B、BF16。系统Python、原有basics和默认训练未改。
本轮没有新增大模型checkpoint，没有读取held-out test选择方案，没有启动收敛长作业。

#### Shape 与完整step矩阵

`artifacts/optimization_matrix_v1/summary.json`：4组shape、4种路径、各2轮，
共32个独立进程的20步性能实验，另有2个矩阵内新shape门槛任务，34项均结束。
S512/S2048的确定性门槛另有先前独立记录；四组embedding完整梯度检查均通过。
每轮前4步不纳入稳态中位数，正反顺序复测，完整step包括forward/backward/optimizer。
此矩阵共同设置`CUBLAS_WORKSPACE_CONFIG=:4096:8`，性能阶段并未打开确定性算法。
共享主机、未锁GPU时钟，故列出配对范围，不把波动大的单轮最高值当成结论。

| B/S | embedding耗时减少，两轮范围 | embedding + 小张量更新，两轮范围 | embedding峰值allocated变化 |
|---|---:|---:|---:|
| 1/128 | 3.01–12.50% | 6.51–11.77% | 减少约631 MiB |
| 1/512 | 2.74–7.37% | 7.72–9.34% | 减少728 MiB |
| 2/512 | 2.14–2.35% | 1.70–2.50% | 减少840 MiB |
| 1/2048 | 1.08–3.27% | 0.60–3.93% | **增加1103 MiB，不推荐默认使用** |

小张量更新单独在S512/B2下基本无收益，S2048两轮均更慢；组合百分比不能相加。
初始与扩展的源码哈希分别保存在progress.json的source_sha256/extension_source_sha256。
扩展前已将相同embedding kernel提升为training模块，新增opt-in大张量调度选项，
保留原记录而非冒充全过程源码未变；提升后的真实完整梯度再次验证通过。

#### 多张量 FlashOptim：减少launch并不保证端到端加速

实现：`training/multi_tensor_flashoptim.py::MultiTensorFlashAdamW`。
直接调用安装版本的官方JIT更新函数，以grid轴1选择参数，轴0保持原有local/persistent遍历。
不cat梯度，不跨参数合并量化group，不改变FP16/BF16/master-weight设置或状态格式。
不支持的dtype、ECC、未对齐输入等走官方step_param，不启用未经验证的指针对齐假设。
checkpoint可与官方双向交换，缺失梯度的逐参数step计数保持一致。

- 小张量合并：每step从310个更新launch变为197个官方kernel + 2个批量kernel，即199次。
  证据：matrix目录的r0_b1_s512_baseline/both.json中两步profiler，620→394+4。
- 全尺寸persistent合并：`artifacts/all_tensor_flashoptim_v1/`，两轮比原版慢约3%。
- 对齐版：`artifacts/all_tensor_flashoptim_vectorized_v1/`，单独更新约−0.23%/+0.49%，无稳定收益。
  叠加embedding的3.5–4.3%不能归功于optimizer。
- PTX核对：`artifacts/optimizer_vectorization_audit_v1.json`。指定4M BF16 case中，
  静态global load指令56→14，出现v4/v2 load，寄存器90→72；这是汇编证据，
  不是4倍吞吐证据，也不是完整Qwen已获得额外加速。
- 官方kernel的block/grid/warp扫描：`artifacts/flashoptim_launch_sweep_v1/`，
  大权重未找到可靠胜出配置，保留官方设置。
- 异步metadata版本用新pinned buffer及non_blocking copy，避免覆盖仍在DMA中的host buffer；
  已过64步中途无GPU同步的参数/量化状态逐位对照，性能证据另列于async目录。

#### CUDA Graph 训练：只捕获F/B，optimizer在graph外正常更新

实现：`training/graphed_forward_backward.py::GraphedForwardBackward`。
固定shape/地址，输入token每步copy到静态buffer；capture时grad=None，replay覆盖已捕获梯度buffer。
不把optimizer step/bias correction冻结在graph内，不在replay之间zero_grad(set_to_none=True)。
检查输入shape/dtype/device、训练模式、参数/梯度地址；错用直接报错。
暂不支持该graph模式的梯度累积、activation checkpointing、非零dropout或动态shape。
普通embedding路径的累积和activation checkpointing则已另外测试通过。

真实模型两组不同输入下，确定性普通执行与Graph replay的loss和**全部参数梯度误差为0**。
证据：`artifacts/graphed_training_v1/result.json`。以下为带共同workspace限制的历史对照：

| 路径 | 两轮step ms | steady peak reserved GiB |
|---|---|---:|
| compiled HF + FlashOptim | 109.41 / 108.87 | 11.688 |
| 上述 + Graph F/B | 102.76 / 103.33 | 11.744 |
| embedding优化 | 105.53 / 105.52 | 11.109 |
| embedding + Graph F/B | 99.26 / 99.46 | 11.027 |

组合相对本表原版少8.64–9.28%时间，Graph在embedding基础上再少5.75–5.95%。
**不能把Graph的max_memory_allocated下降直接叫工作空间节省**：replay会使用私有pool，
期间不重新更新allocator分配统计。应同时看reserved和capture阶段峰值。
原版→Graph-only的reserved反而增加约58 MiB；组合→原版的reserved少约0.66 GiB。

长序列反例：`artifacts/graphed_training_b1_s2048_v1/result.json`，完整梯度门槛仍为0误差，
但普通embedding 288.02/291.82 ms，Graph 288.98/290.02 ms，没有稳定收益；
reserved 12.930→13.275 GiB，增加354 MiB。不能把S512结论推广到S2048。

初次构建/更新成本另计：S512 Graph-native setup约11.4–11.6s、首次optimizer step约8.2s；
Graph-embedding setup约31.9–32.3s、首次step约8.2s。普通native首次step约19s，
普通embedding约39s。磁盘JIT cache已复用；以上不含全进程加载，不称作全冷启动。
20步短作业并未因组合稳态更快就获总耗时优势，不外推收敛加速。

#### 不同训练数据与验证集、接入/恢复测试

- `artifacts/optimized_trajectory_v1/comparison.json`：embedding + 小张量optimizer，
  256步变化训练batch、16个独立validation blocks、每64步验证，最大loss差0.00354。
- `artifacts/graphed_trajectory_v1/comparison.json`：embedding + Graph、原FlashOptim，
  同口径最大差0.00190；step256 validation 2.24243→2.24222。
  仍是单对短轨迹，不是到收敛、不使用test、不声称泛化或质量无损。
- 最终目标回归：**64 passed**，不是全项目所有测试的总数；
  `artifacts/systematic_optimization_final_v1.xml`。
  覆盖FP16/BF16/FP32、重复/热token、非连续/非对齐、两阶状态、双向恢复、
  无同步异步上传、梯度累积、checkpoint重算、Graph变输入/更新权重与错误buffer保护。
- 实际训练CLI的合成小模型中断→durable checkpoint→恢复→模式不匹配拒绝也已跑通，
  大小仅几十KB，未生成新的1.7B checkpoint。

#### 训练入口与适用范围

`training/flashoptim_convergence_job.py`新增可选`--execution graph`和`--optimizations`，
默认仍为原有eager/off；旧checkpoint仍以off模式恢复，恢复时检查execution/optimization一致。
embedding模式必须提供已通过、确定性、B1/seq匹配的真实梯度门槛JSON。
`training/tied_embedding.py`为可复用实现，benchmark不再充当实现模块。

Graph接入示例（可选原型，不是下方最终强基线复核后的推荐默认；本轮**未**启动此长作业）：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv-research/bin/python -m training.flashoptim_convergence_job \
  --protocol artifacts/flashoptim_convergence_v1/frozen_protocol.json \
  --output artifacts/my_graph_embedding_training_run \
  --optimizer flashoptim --execution graph --optimizations embedding \
  --embedding-gate artifacts/tied_embedding_promoted_gate_v1/result.json
```

S2048建议保留`--execution compile --optimizations off`，不要为约1–3%局部速度默认付出更高显存。
新长作业仍需单独做磁盘/真实I/O预算与收敛比较，不能由本节短测替代。

#### 淘汰项与观测限制

原生SDPA enable_gqa去KV复制，两版（strided/contiguous）均失败：
`artifacts/native_gqa_training_gate_v1/`及`native_gqa_contiguous_gate_v1/`。
确定性repeat误差0，候选全量3.37–3.41%、最差参数8.08–8.63%，超过2%/5%门槛，
不接入训练，不因此宣称GQA算法错误。MLP addmm dX也无加速，保留旧实现。

Nsight Compute 2020与CUDA研究环境不兼容；改用已有2023版后明确报
`ERR_NVGPUCTRPERM`，是在沙盒外运行的真实GPU计数器权限限制。
证据：`artifacts/flashoptim_bandwidth_ncu2023_v1.csv`。
未修改驱动权限、未锁GPU时钟，**未取得硬件DRAM字节计数，不捏造带宽利用率**。
当前归因基于Torch profiler、PTX与严格消融，不能替代硬件counter。

#### 最终强基线复核与推荐：以本段为准

**收紧此前Graph结论**：确定性梯度检查需要workspace配置，但普通性能对照不应只看
该配置。在同一未额外限制workspace的环境重新做两轮正反顺序比较：
`artifacts/graphed_training_unrestricted_v1/result.json`。

| B1/S512路径 | 两轮完整step ms |
|---|---|
| compiled HF + FlashOptim | 105.08 / 106.02 |
| 上述 + Graph | 104.74 / 104.44 |
| embedding | 101.81 / 101.95 |
| embedding + Graph | 100.83 / 101.01 |

此对照中Graph在embedding基础上只额外减少约0.92–0.96%，不足以默认承担固定buffer/
shape限制与维护成本。不能把历史受workspace约束的8.6–9.3%表述为普遍或最强baseline收益。
不同GPU/负载的对照不能用来单独断言workspace是全部差异原因，更不能跨实验相加倍率。

**最终有更清晰配对证据的是proxy生命周期优化**：
`patch_tied_embedding(model,consume_proxy=True)` 在head用完共享weight view后删除临时属性，
缩短图/Tensor的可观察生命周期，未原地覆盖输入梯度、未更改数学公式或optimizer。
两种模式都保留，默认全局训练仍为off。

S512完整模型门槛：`artifacts/tied_embedding_consume_s512_gate_v1/result.json`，
loss一致，全量梯度relative L2 0.0531%、共享权重0.2413%，其他参数梯度一致。
S2048也已单独验证；FP16/BF16/FP32及累积/重算等单测见发布回归。

| 反向顺序独立进程复测 | baseline step ms | 合并+消费proxy step ms | 耗时减少 | peak allocated GiB |
|---|---:|---:|---:|---|
| baseline→candidate→consume | 104.163 | 98.408 | 5.53% | 11.140→9.849 |
| consume→candidate→baseline | 104.469 | 97.595 | 6.58% | 11.140→9.849 |

证据：`artifacts/embedding_memory_lifetime_s512_v1/`与
`artifacts/embedding_memory_lifetime_s512_confirm_v1/`；
相对baseline少约1322 MiB，即1.29 GiB；相对先前embedding版再少约594 MiB。
这是BF16/no-master、原Flash SDPA与原FlashOptim下，完整forward/backward/optimizer稳态结果。
它与Graph的变动batch实验负载/物理卡不同，不能据此直接拼接绝对时间、证明跨实验排名。

分阶段显存诊断：S2048三种forward峰值基本相同、backward结束后allocated相同，
差异来自backward临时峰值。baseline/candidate/consume分别11.580/12.657/12.077 GiB，
及时释放proxy部分减轻回退，但仍比baseline多约509 MiB，故S2048仍推荐原生compile/off。
见`artifacts/embedding_memory_lifetime_v1/result.json`。
不能仅凭约一份大权重大小的差值，就断言所有内部allocator重用细节。

消耗proxy版另跑256步变化训练batch：
`artifacts/consumed_embedding_trajectory_v1/comparison.json`，独立validation最大loss差0.00151，
通过0.01短期门槛。依旧不是收敛比较，不写“质量无损/收敛更快”。
首次编译/初始化仍需摊销，以上step收益不能代替全作业含I/O结果。

异步metadata最终结果：`artifacts/all_tensor_flashoptim_async_v1/result.json`。
optimizer单独两轮约慢1.16%/快0.32%，无稳定收益；叠加embedding的改善不能归给optimizer。
已保留64步无中途同步的正确性回归，默认不采用全张量/异步变体。

**发布回归67 passed**，`artifacts/systematic_optimization_release_v1.xml`，
额外覆盖consumed入口自身的gate匹配、实际训练CLI中断/恢复及Graph组合。
这是目标回归，不冒称全项目全部测试。统一证据、源码/权重哈希和最终决定在
`artifacts/optimization_wave_final_v1.json`。

当前B1/S512可选接入命令（完整收敛作业仍未启动；先预算磁盘/checkpoint I/O）：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv-research/bin/python -m training.flashoptim_convergence_job \
  --protocol artifacts/flashoptim_convergence_v1/frozen_protocol.json \
  --output artifacts/my_consumed_embedding_training_run \
  --optimizer flashoptim --execution compile --optimizations embedding_consumed \
  --embedding-gate artifacts/tied_embedding_consume_s512_gate_v1/result.json
```

恢复时使用同样的execution/optimizations/gate与`--resume`，不要跨模式覆盖旧作业。
未通过相同B1/seq/consume_proxy门槛的配置会被拒绝，不要求用户手动修改模型调用链。
