# 论文驱动的 Qwen 优化实验（2026-09-28）

目标不是把所有算子都改成 Triton，而是寻找尚未覆盖的瓶颈，并用本机
RTX A6000 / Qwen3-1.7B 的结果决定是否接入。以下区分论文结论、工程参考和本地实验；
不能把其他 GPU/模型上的论文加速比写成我们的结果。

## 1. 文献筛选

| 来源 | 方法与适用条件 | 本项目处理 |
|---|---|---|
| [Deep Kernel Fusion for Transformers，2026-02](https://arxiv.org/html/2602.11808v1) | gate/up 两路投影与 SwiGLU 融合；按 shape/hardware 选择布局与 kernel。论文主要在 TP=4 的 A100/H100、Llama-3.1-70B 上评测。融合范围不是整个 MLP：down projection 仍独立。 | 过去偏重大 M 训练；新增小 M decode 实验，使用实际 Qwen 第一层权重。实现一个独立 GEMV 候选，不宣称复现论文 kernel。 |
| [FlashInfer，MLSys 2025](https://arxiv.org/abs/2501.01005) | KV 布局、可定制 attention、动态负载均衡并兼容 CUDA Graph。 | 后续长 KV decode / split-KV 调度的参考；本轮没有复现服务调度器。 |
| [FlashInfer fused_add_rmsnorm 官方接口](https://docs.flashinfer.ai/generated/flashinfer.norm.fused_add_rmsnorm.html) | 将残差加法与 RMSNorm 融合。属于工程实现参考，不能称为该论文的新贡献。 | 实现保留 HF 舍入的非原地版本，返回 residual 与 normalized 两个输出。完成算子及真实模型消融。 |
| [FlashAttention-4，2026-03](https://arxiv.org/abs/2603.05451) | 针对 Blackwell 的流水线、tensor memory、2-CTA MMA 等设计。 | A6000 没有对应硬件能力，不能直接移植或引用 B200 上的收益。暂不替换现有 Flash SDPA。 |

## 2. 新增 Add + RMSNorm

- 实现：`ops/triton/add_rms_norm.py`。
- 数值顺序：低精度 residual sum → FP32 RMS reduction → cast → 乘权重。
- 不修改输入；后续 MLP 残差仍需使用 sum，不能把它丢掉。
- 仅合并每层 attention 后的 add 与 post-attention norm，未改变层间接口。
- `qwen3_triton_inference(model, fuse_residual=True)` 显式开启；默认关闭。

BF16、hidden=2048 的 CUDA Graph GPU 中位数（μs）：

| B,S | 独立 add + 已有 Triton norm | 融合 | torch.compile |
|---|---:|---:|---:|
| 1,128 | 5.63 | 4.20 | 4.40 |
| 2,512 | 30.82 | 26.73 | 27.65 |

单算子确实减少一次 launch 和 residual 的重复读取。但两输出依然要分配，
相对已有 Triton norm，新增峰值分配没有下降；不能宣称这一步减少了中间张量分配。

第一轮真实模型交错测量（BF16，20 次有效采样，prompt 128/512）没有稳定的
wall-time 收益：相对已有 Triton 路径，差异在约 -0.3% 到 +0.9% 之间。
因此保留实验开关，不修改默认。GPU 微秒级节约不等于 Python 驱动的完整模型明显提速。

证据：`artifacts/add_rms_norm_ablation.json`、`artifacts/residual_ablation_bfloat16.json`。

## 3. 小 batch decode：融合 GEMV + SwiGLU

- 实现：`ops/triton/decode_gate_up.py`。
- 一个 program 处理一个 token 和 1/2/4/8 个输出 channel，直接归约 hidden 维。
- 不为 M=1 填充 Tensor Core GEMM 的 M tile；两个投影累加后，按 HF 顺序 cast、SiLU、乘 up。
- 权重读取量仍在，不能说融合消除了两份权重；M 增大时重复读取权重会变贵。
- 未融合 down projection，不支持 backward；默认模型路径不改变。
- `qwen3_triton_inference(model, decode_mlp=True)` 只在单 token、batch≤2、hidden=2048 时启用，其他 shape 走原路径。

单层真实权重、合成输入的 BF16 初步测量（CUDA Graph GPU 中位数，μs）：

| M | 两次 cuBLAS + Triton 激活 | GEMV BN=1 | 结论 |
|---|---:|---:|---|
| 1 | 81.31 | 74.85 | 值得进入真实模型验证 |
| 2 | 81.41 | 74.75 | 值得进入真实模型验证 |
| 4 | 81.10 | 99.53 | 不启用 |
| 8 | 82.02 | 212.48 | 不启用 |
| 16 | 82.74 | 429.77 | 不启用 |

也记录 packed cuBLAS + contiguous copies 对照；这是带复制的实现，不代表最佳
packed GEMM epilogue。上述表没有声称胜过最优 packed/compiled MLP。
FP16/BF16 × M=1/2/4/8/16 × 四种 BN 均检查相对 L2；
不以 near-zero 元素的相对误差替代整体和绝对误差报告。

证据：`artifacts/decode_gate_up_ablation.json`。真实模型四路径交错消融见
`artifacts/fusion_ablation_<dtype>.json`，不能用算子表代替整模型结论。

### 完整模型结果

下面只比较本次新增 GEMV 与已有 Triton 路径，不能将这部分增益重复加到旧报告的百分比上。
单位 ms，为 20 次交错测量的同步 wall-time 中位数：

| dtype | B,prompt | 现有 Triton decode | + GEMV decode | 延迟下降 |
|---|---|---:|---:|---:|
| FP16 | 1,128 | 22.833 | 22.298 | 2.34% |
| FP16 | 2,128 | 24.368 | 23.982 | 1.59% |
| FP16 | 1,512 | 21.755 | 21.370 | 1.77% |
| BF16 | 1,128 | 27.062 | 26.512 | 2.04% |
| BF16 | 2,128 | 30.412 | 29.471 | 3.10% |
| BF16 | 1,512 | 23.009 | 22.540 | 2.04% |

这是一个规模较小但方向一致的实验收益；未固定 GPU 时钟、未测并发服务、未与完整
compiled HF 比较。Prefill 没启用 GEMV，其时间差异应看作测量噪声，而非该 kernel 的收益。

新增分支相对 HF 的 logits 相对 L2 最大：FP16 约 0.081%，BF16 约 0.784%。
**重要反例：** BF16 B=2,prompt=128 的第一个 decode 步，至少一个 batch 元素的
argmax 与 HF 不同；原有 Triton、残差融合、GEMV 三条路径均在这一步出现差异。
其余本次记录的 argmax 检查一致。JSON 中该字段是每个 step 对整个 batch 的
`all()`，不是逐 token 一致率。不能声称通用质量无损，也不能把这次差异完全归因于新 GEMV。
FP16 本轮所有 argmax 检查一致。后续如要求严格生成行为，应增加自然文本集与近似并列
logit 的诊断；默认仍关闭新增实验开关。

## 4. 测量与复现

```bash
cd /data1/tzh/proj/qwen_operator_lab
export PYTHONPATH=.
/data1/tzh/proj/systems/.venv/bin/python -m pytest -q tests/test_add_rms_norm.py tests/test_decode_gate_up.py tests/test_hf_inference.py
/data1/tzh/proj/systems/.venv/bin/python benchmarks/benchmark_add_rms_norm.py
/data1/tzh/proj/systems/.venv/bin/python benchmarks/benchmark_decode_gate_up.py
/data1/tzh/proj/systems/.venv/bin/python benchmarks/benchmark_residual_ablation.py --dtype bfloat16 --decode-mlp
/data1/tzh/proj/systems/.venv/bin/python benchmarks/benchmark_residual_ablation.py --dtype float16 --decode-mlp
```

完整模型采用强制 Flash SDPA、logits_to_keep=1；(B,S) 为 (1,128)/(2,128)/(1,512)。
每轮正反交替顺序测量，四轮预热后 20 个样本；context 安装不计入时间。
decode 每次新建等长 cache，prepare 在计时外；计时包含 cache update。
数值检查含每层 hidden states、4 步固定 token logits、full vs incremental，
相对 L2 门槛 2%，另报 max_abs 和 argmax；不是任务质量评估。

## 5. 尚未覆盖、值得后续研究的工作

1. 长 KV decode 的 split-KV 与 GQA head 共享：先扫 cache 长度，再判断并行度是否不足；短 prompt 不代表长上下文。
2. packed QKV/gate-up projection 的无复制 epilogue：当前 cuBLAS packed 对照还包含激活布局复制，应先完善这个基线再宣称优于 packed。
3. CUDA Graph 整模型执行：现有 wall-time 包含 Python；固定/静态 cache 与捕获接口必须先验证，不能直接拿算子 graph 加速比推算模型收益。
4. 训练的 LM-head + cross-entropy 融合：针对全 token 大词表 logits，可节省显存，但与当前 logits_to_keep=1 的推理瓶颈不同，应另建训练 benchmark。

本轮优先验证前面两个实际新增算子，不把量化、分页服务、训练优化混为一个收益口径。

## 6. 论文算法可行性实验（2026-11-01）：FASA

**状态：不接入默认路径。32K 上出现速度收益，但本次两个配置都没有通过数值门槛。**
这是算法迁移的实测负结果，不是对论文在其他模型上的结论的否定。

### 实现范围与回退

参考 [FASA v3，2026-02](https://arxiv.org/html/2602.03152v3) 的 Eq.4 / Algorithm 1，
独立实现 Qwen3-1.7B 的 FASA-C 风格实验，并非直接运行作者代码。
对每层每个 query head，按校准集上的 top-256 token 集合重合度排序 frequency chunks；
Qwen3 使用 `(i, i+64)` 配对。选择 16/32 对后预测 token 排名，选出 512/1024 个 token，
再用完整 128 维重新计算其 attention，不能直接用 proxy scores 当最终 softmax。

- `ops/reference/frequency_sparse.py`：FP32 校准、候选选择与 attention oracle。
- `ops/triton/frequency_sparse.py`：增量 proxy K packing、低维打分、按索引直接读原 KV 的 split-KV attention。
- `model/frequency_sparse.py`：显式 `FrequencySparseRunner`，与现有 `StaticDecodeRunner` 分离。
- `benchmarks/benchmark_frequency_sparse.py`：校准、诊断、128 步 teacher-forced 误差、分阶段及完整计时。
- `tests/test_frequency_sparse.py`：split-half/GQA/strides/FP16/BF16/FP32、无效 cache 尾部、图回放及模型恢复。

没有修改默认模型或 `model/static_decode.py`，没有改宿主 Python、现有虚拟环境依赖或 checkpoint。
所有实验 context 退出后恢复原 forward；用户原有命令继续走原来的 dense/Flash 路径。
未通过的研究代码保留为独立、默认不导入的实验，不将失败分支塞入生产调用链。

### 数据与实验协议

- GPU 0：RTX A6000 48GB；FP16、B=1、Qwen3-1.7B 原权重，不混用 INT8 MLP/LM head。
- Torch 2.5.1+cu121 / Triton 3.1.0 / Transformers 4.51.0，没有安装论文旧依赖。
- 本地缓存 `Salesforce/wikitext`、`wikitext-103-raw-v1`，revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`；parquet SHA256 在 JSON manifest 中。
- calibration：validation split 的 4 个不重叠 8192-token **窗口**，每个窗口 8 个 teacher-forced query，
  共每层 32 个 query。参数名 `calibration-docs` 是窗口数，不代表独立完整文章；这是小规模可行性校准。
- test：独立 test split，8K/16K/32K prompt 后各 128 个真实 continuation token。
  三种 prompt 长度都从 test offset=0 开始，因此是嵌套窗口，不是三个独立数据集。
- 质量：相同 teacher-forced token 输入，报告 logits L2、KL、top-1 一致率、continuation NLL/PPL ratio。
  KV 随各候选自己的 hidden states 更新，包含跨层/跨 decode 步的误差传播。
- 时间：现有 exact-weight Triton 前后处理 + 原生 Flash + static cache + CUDA Graph 是 baseline；
  2 轮预热后 8 轮随机交错测量，包含完整 prefill、proxy packing、128 次 greedy decode、
  LM head、argmax、cache 更新和同步；排除初始化、JIT/capture。候选生成内容可能分歧，但步数固定相同。
- 稀疏打分、`torch.topk`、indexed attention 均计入候选耗时，不能只报最后一段 kernel。
- 预设初筛条件：pipeline 至少 1.05x、paired CI 下界 >1，最大 logits L2 <2%、
  top-1 >=99%、平均 KL <=0.005、continuation PPL ratio <=1.01；此外仍需独立任务质量验收。
  这是工程筛选门槛，不是“近似 attention 必须逐 token 完全相同”的学术判据。

### 端到端结果

单位 ms，表中时间为每路径中位数；变化率由表中中位数计算，非 paired-ratio 中位数。
top-1 是各自 128 个 teacher-forced 位置的模型输出一致率，**不是任务准确率**。

| Prompt | dense/Flash pipeline | F16/K512 pipeline | 相对 dense | top-1 一致率 | F32/K1024 pipeline | 相对 dense | top-1 一致率 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 1280.08 | 1361.43 | 慢 6.36% | 83.59% | 1434.48 | 慢 12.06% | 96.09% |
| 16384 | 1959.40 | 2066.69 | 慢 5.48% | 78.12% | 2192.57 | 慢 11.90% | 94.53% |
| 32768 | 3920.00 | 3628.02 | 快 7.45% | 75.78% | 3828.45 | 快 2.34% | 86.72% |

32K F16/K512 的 decode 部分：1455.62 → 1152.28 ms，约减少 20.84%；
完整流程 paired speedup 1.083x，95% bootstrap CI [1.078, 1.084]。
但最大 logits L2 为 **142.28%**，不是可用的无损优化，**不写成简历加速成果**。
F32/K1024 的 32K 最大 logits L2 也达到 36.56%。部分窗口 NLL 略低，不能据此忽略明显的行为偏差。

### 瓶颈与算法观察

8K、layer 14 的 Graph 算子中位数（μs）：

| 路径 | proxy score | top-k | indexed full-dim attention | 含 proxy update 的完整调用 |
|---|---:|---:|---:|---:|
| 原生 Flash | — | — | — | 66.84 |
| F16/K512 | 19.64 | 63.36 | 13.98 | 91.90 |
| F32/K1024 | 32.03 | 61.49 | 20.53 | 109.59 |

独立阶段时间不会严格加和等于整段时间（缓存/调度不同）；整段直接测量是最终依据。
32K 相应完整调用约为 207.77 / 125.62 / 181.89 μs。
top-k 的实测耗时不随长度线性增长，不从这些数值反推未验证的内部算法。

F16 在 8K 的五层诊断样本上，平均保留 attention mass 约 91.43%，
高于随机 frequency chunks 的 52.11% 和简单 sink+recent 的 84.01%；oracle top-512 为 94.98%。
这支持“校准的频率有筛选价值”，但不等于当前参数足以维持模型行为。
甚至 oracle top-k 也会丢弃剩余概率质量，不能把全部误差归因于 kernel 或 selector。

当前 compact proxy 按 query head 存储，不是完整 KV 的 GQA 展开，也不是无额外内存：
32K+129 容量下，F16/F32 分别额外占约 **899.5 / 1799.1 MiB**；
完整 K/V 仍保留。不能宣称 FASA-C 实验节省了 KV resident memory。

### 证据与复现

- `artifacts/frequency_calibration_pilot.json`：每层每 head 的校准分数，固定用于后续长度。
- `artifacts/frequency_sparse_pilot.json`：8K。
- `artifacts/frequency_sparse_long.json`：16K/32K。
- `artifacts/frequency_wikitext_data.json`：由现有公开 parquet 缓存转换的数据及来源摘要。

```bash
cd /data1/tzh/proj/qwen_operator_lab
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
/data1/tzh/proj/systems/.venv/bin/python benchmarks/benchmark_frequency_sparse.py \
  --tag new_run --seqs 8192,16384,32768 --steps 128 --repeats 8 \
  --calibration artifacts/frequency_calibration_pilot.json
```

脚本拒绝覆盖同名结果，复测需新 `--tag`。JSON 包含运行时源码哈希；首轮以后只补了输入
contract 检查和脚本清理/CPU 线程设置，核心 kernel 数学未变化，不能冒称各轮源码完全相同。
WikiText 小样本并非 LongBench、长文本检索或推理质量评测；没有跑的任务不作结论。

## 7. 块跳过对照（2026-11-01）：BLASST-inspired

**状态：保留默认 Flash；本次小阈值候选通过本地数值初筛，但没有达到 5% 的性能采用门槛。**

参考 [BLASST v3](https://arxiv.org/html/2512.12087v3) 的 running-max 判据。
这是 MLSys 2026 论文，但 arXiv 首次提交于 2025-12，不能标成“2026 首发”。
本项目只迁移算法思想到 A6000/Triton，不复现其 Hopper/Blackwell CUDA batched-load pipeline。

实现 `ops/triton/block_skip_decode.py`：每个 split 独立维护 online-softmax 的 running max，
逆序处理自己的 KV tiles；若 `block_max - running_max < log(threshold)`，跳过该块
的 V load、softmax 与 PV。**K 和 QK 仍全部读取/计算**，不是同时跳过 K/V。
固定 `splits=16, block_n=128`；`threshold=0` 是不做剪枝的同版 dense 对照。
`model/block_skip.py::BlockSkipRunner` 为独立 opt-in runner。

### 先做算子筛选

`benchmarks/benchmark_block_skip.py` 在真实模型的 0/7/14/21/27 层 decode states 上，
扫描 splits/block-size 和 threshold=0/1e-6/1e-4/1e-2，报告真实 skipped fraction、输出 L2 和
Graph GPU 延迟；数据为 test offset=0，8K/32K prompt 后 16 步的最后 query。

不能只对比“自己的稠密 kernel”：真正的性能基线是原生 Flash。
在 S16/BN128 下，1e-4 阈值实际跳块比例为：8K 五层约 0–0.29%，32K 约 0.15–2.55%。
增到 1e-2 后跳过更多，但一些层输出 L2 达到约 4–5%，不能据此直接接入模型。
某些**没有实际跳块**的样本中，带条件分支的 kernel 也比 threshold=0 编译分支更快，
因此不能把两版之间的全部差异归因于稀疏度；需要编译器/调度层面的额外分析才能解释。

### 独立窗口的完整验证

固定 1e-4，不根据后续结果改阈值。移到 test offset=65536 的独立窗口，
每种长度 128 步 teacher-forced 质量检查；另做 2 warmup + 8 randomized paired
greedy pipeline round。baseline 与 FASA 实验一致，但**文本窗口不同，不能跨实验直接比较绝对时间**。

| Prompt | Flash pipeline ms | 自写、不跳块 ms | 自写、小阈值跳块 ms | 跳块相对 Flash | top-1 一致率 | 最大 logits L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 1274.69 | 1287.41 | 1265.60 | 快 0.71% | 99.22% | 0.844% |
| 32768 | 3916.08 | 3989.26 | 3949.21 | 慢 0.85% | 100% | 0.991% |

小阈值候选平均 KL：8K 约 1.66e-5，32K 约 1.83e-5；continuation PPL ratio 约
0.999987 / 1.000314。符合本次的初筛条件，但不是长文本任务质量保证。
作为 FP 舍入对照，不跳块版本的最大 logits L2 也有 0.771% / 0.605%，
8K top-1 同样为 99.22%，因此不能把所有差异都算在剪枝上。

完整 pipeline paired speedup：8K 为 1.0075x，95% CI [1.0051,1.0092]；
32K 为 0.9947x，CI [0.9925,0.9976]。
这是同次测量的重采样区间，没有固定 GPU 时钟、没有跨日复测；小幅收益不等于稳定可推广的改进。
更不能因它比**自己的不跳块 kernel**快，就宣称超过现有 Flash pipeline。

证据：

- `artifacts/block_skip_screen.json`：算子阶段完整扫描与反例。
- `artifacts/block_skip_e2e_confirm.json`：独立窗口端到端验证。
- `tests/test_block_skip_decode.py`：零阈值 dense parity、实际发生跳块的人工样例、
  随机非零阈值逐块 PyTorch oracle、完整模型 graph decode 与 context 恢复。

```bash
cd /data1/tzh/proj/qwen_operator_lab
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
/data1/tzh/proj/systems/.venv/bin/python benchmarks/benchmark_block_skip.py --tag new_screen
/data1/tzh/proj/systems/.venv/bin/python benchmarks/benchmark_block_skip_e2e.py \
  --tag new_confirm --seqs 8192,32768 --steps 128 --repeats 8 --test-offset 65536
```

## 8. 本轮采用决定与后续范围

两条候选都**不启用**；原来的命令、默认模型路径、依赖版本、checkpoint 保持不变。
研究代码/原始 JSON/测试保留用于复现，不删除负结果，也不把这些数值加入简历的有效加速项。
门槛自动核算及原默认源码未变化的检查由 `benchmarks/summarize_sparse_research.py` 完成。

本轮没有实现 KVarN 量化 KV，也没有接入/训练 DFlash drafter。
这两个方向分别需要量化表示和误差传播验证、目标模型匹配的 draft checkpoint 与多 token 验证/cache 提交机制，
不能从这两次 sparse-attention 实验推断其有效或无效。

如果继续 FASA，下一轮应先检查更大预算、保留 sink/local tokens、分 head/layer 预算的质量—速度曲线，
并扩大校准和独立任务集；不能只继续压缩 kernel 时间来掩盖已经观察到的行为偏差。
如果继续块跳过，首先解释“几乎没跳块却有局部提速”的编译/调度差异，以及提高长上下文下实际
安全可跳过的比例；当前证据不足以支持默认替换。

### 最终回归与审计

完整项目回归：**190 passed，2 warnings，474.09 s**。
其中本轮新增 25 项测试；两条 warning 为 cuBLAS 首次 CUDA context 初始化、TF32 未启用提示，
没有为追求速度改变精度设置。报告：`artifacts/sparse_research_regression.xml`。

`artifacts/sparse_research_audit.json` 汇总每个候选的质量/性能门槛，校验原默认
`model/hf_inference.py`、`model/static_decode.py` 的源码哈希从实验开始到结束未变，
并在**最终审计时**记录本地 checkpoint/tokenizer 文件 SHA256。
这不是声称所有文件哈希均在每轮 benchmark 开始前采集；原始 benchmark manifest 保留不改。

```bash
cd /data1/tzh/proj/qwen_operator_lab
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 /data1/tzh/proj/systems/.venv/bin/python -m pytest -q
# 使用新的输出名，避免覆盖已经留存的审计：
/data1/tzh/proj/systems/.venv/bin/python benchmarks/summarize_sparse_research.py \
  --output artifacts/sparse_research_audit_recheck.json
```

## 9. 完整 Qwen3 训练优化：实现与验收证据

本节与前面的 inference 实验分离。实现、三轮复测、预算扫描、数值验收及完整回归均已完成；
完成实验不等于所有候选可以采用。不能将早期顺序运行的加速比当作稳定结论。
训练对象为本地完整 28 层 Qwen3-1.7B；没有改 checkpoint。

### 已实现

- `benchmarks/benchmark_full_training.py`：HF / compile / RMSNorm / SwiGLU /
  QK-Norm+RoPE / 算子组合 / 分块 loss / fused loss；记录 forward、backward、
  fused AdamW、step wall time、tokens/s、peak allocated 和 peak delta。
- BF16 参数、梯度、Adam moments，**没有 FP32 master weights**；固定 validation 文本，
  `use_cache=False`，强制原生 Flash SDPA。比较时 batch、累积、checkpointing 必须相同。
- Tiny loss 统一先 cast FP32，避免与 HF loss 的精度口径不同。
- `training/hf_operators.py`：可恢复的训练替换 context，attention/GEMM 保留原生。
  旧训练 kernels 的 cast 边界与 HF 不完全相同，必须经过 loss/全参数梯度门槛。
- `training/chunked_loss.py`：按 token 分块、backward 重计算、FP32 完整 dW 累加，无梯度过滤。
- `ops/triton/linear_ce_forward.py`：GEMM tile 内归约 LSE/target logit，只输出
  `[N,ceil(V/128)]` 摘要；`training/fused_loss.py` 接入完整重计算 backward。
  **backward 仍使用分块 logits/native GEMM，不是全阶段单 kernel。**
- `ops/triton/ce_gradient.py`：直接写低精度 dlogits，避免完整 FP32 softmax 概率张量。
- `benchmarks/sweep_training_budget.py`：相同 28 GiB **分配器**预算、S=2048、有效 batch=4，
  microbatch=1/2/4 与累积=4/2/1，独立进程、随机执行顺序，保留 OOM 日志。
- `benchmarks/validate_full_training.py`：完整模型逐参数梯度检查，正在运行；
  它不是收敛性或任务质量测试。

### 已观察到的初筛结果

S=512，B=1，无 checkpointing 的顺序诊断计时（ms/step）：
HF 249.98、compile 203.88、RMSNorm 239.20、SwiGLU 266.05、QK 231.81、
组合 226.23、旧分块 296.26。各报告为 `artifacts/full_training_*s512_v1.json`。
这些轮次没有随机交错，且部分后来测量采用 steady 计时，**不能混在一张最终 speedup 表里**。

S=4096 的 steady 初筛中，compile 的峰值 allocated 比 HF 更低；自写 fused loss
虽减少完整 logits 存储，backward 重计算和 FP32 dW 仍有成本，未胜过 compile。
同预算扫描已观察到 fused loss microbatch2 OOM、compile microbatch2 成功。
前者 OOM 出现在 backward 的 dW dtype cast，说明“前向不保存 logits”并不保证更大 batch。

### 验收范围与证据映射

1. 九组预算全部结束，只有成功/OOM、无未解释错误：`training_budget_v1.json` 和对应日志。
2. 完整模型逐参数梯度及原生重复对照已完成；每轮 compile counters 为单一 captured graph，
   首次编译列在 warmup，而不是纳入 steady 中位数。
3. 七条完整模型路径、三轮随机进程顺序、同输入指纹复测：`full_training_repeated_steady_v1.json`。
4. 同配置及同预算结论均保留负结果，见下方最终表。
5. 完整回归与自动审计：`full_training_regression_v1.xml`、`full_training_final_audit_v1.json`。

已有小型测试证据：`fused_loss_boundaries_v2.xml` 共 26 项，覆盖真实词表、
FP16/BF16/FP32、全忽略标签、非连续输入、部分梯度、共享 embedding 权重；
`training_operators_validation_v1.xml` 共 9 项。小型测试不能替代完整模型验收。

### 新增验收结果（最终性能复测仍在运行）

完整回归 `artifacts/full_training_regression_v1.xml`：**231 passed，2 warnings，48.76 s**。
两条 warning 仍是 cuBLAS 初始 context 和未启用 TF32 提示；没有改精度或依赖来消除它们。

`artifacts/full_training_gradient_validation_v1.json` 使用真实预训练 28 层模型、BF16、
S=128、同一批文本，比较 loss 和**所有参数**梯度。采用前预设门槛为 loss abs<0.01、
每参数 relative L2<5%，不是仅比较全模型平均。结果：

| 候选 | 所有参数中的最大梯度相对 L2 | 本次门槛 |
|---|---:|---|
| RMSNorm | 149.44% | 不通过 |
| SwiGLU | 120.52% | 不通过 |
| QK-Norm+RoPE | 173.56% | 不通过 |
| 三者组合 | 160.80% | 不通过 |
| Fused loss | 4.55% | 通过初筛 |
| torch.compile | 69.71% | 不通过 |

这不是说 compile 的训练目标改变，也不是宣称上述算子数学公式错误；它说明本机 pinned
栈与这些 pretrained BF16 states 上的梯度不能按当前门槛视为一致，不能只看 loss 接近就采用。
`full_training_repeat_control_v2.json` 的原生 HF 重复计算对照所有参数梯度 L2=0，
排除了该次参考重复计算本身的波动；不能据此猜测具体哪个 cast/kernel 是全部偏差的原因。
Fused loss 只是该配置数值初筛通过，没有完整训练收敛或全任务质量的保证。

预算扫描 `artifacts/training_budget_v1.json` 九组全部结束：

| 路径 | microbatch 1 / 累积4 | microbatch 2 / 累积2 | microbatch 4 / 累积1 | 最高已测试可用 |
|---|---|---|---|---:|
| HF eager | 成功 | OOM | OOM | 1 |
| compile | 成功 | 成功 | OOM | 2 |
| Fused loss | 成功 | OOM | OOM | 1 |

有效 batch=4、S=2048、分配器上限28 GiB、无 checkpointing；不是硬件总显存上限，
也没有证明在任意 batch 值下的全局最大值。成功配置各8个 steady 样本，
HF microbatch1约4826 tokens/s，compile microbatch1/2约6390/6829 tokens/s，
fused loss microbatch1约3828 tokens/s。只描述这次预算测量；compile 未通过数值门槛，
不能把最高吞吐自动当作最终可采用方案。

当前采用决定：**保留原生 HF 训练路径，新增训练 context/融合 loss 均 opt-in，不覆盖原模型权重。**
后续不是把门槛调宽让最快的候选过关，而是保留数值反例与复现入口。

### 最终同配置随机复测

B=1、S=512、BF16、无 checkpointing、AdamW/累积配置相同。三轮中每轮重载原始权重，
随机进程执行顺序，每进程2 warmup+8 steady step，表为三轮 step 中位数的中位数。
GPU event 分阶段计时；steady 模式不在 forward/backward 间强制同步，完整 step 包含 optimizer。

| 路径 | step ms | tokens/s | 最大 peak allocated GiB | 真实模型数值门槛 | 采用决定 |
|---|---:|---:|---:|---|---|
| HF Flash SDPA | 270.66 | 1892 | 14.00 | 参考 | 保留 |
| HF + compile | 204.42 | 2505 | 14.01 | 不通过 | 仅性能对照 |
| Triton RMSNorm | 247.72 | 2067 | 14.00 | 不通过 | 不默认启用 |
| Triton SwiGLU | 279.54 | 1832 | 14.00 | 不通过 | 不默认启用 |
| Triton QK-Norm+RoPE | 224.66 | 2279 | 14.00 | 不通过 | 不默认启用 |
| 三算子组合 | 214.92 | 2382 | 14.00 | 不通过 | 不默认启用 |
| Fused linear CE | 275.03 | 1862 | 13.42 | S=128 初筛通过 | 省显存、未加速 |

数值门槛列对应上面的 S=128 完整模型测试，不冒称验证了任意 shape 的数值一致性。
同轮 paired speedup 均保存在 audit。HF 本身轮间也有波动，三个进程级 round 不是
跨日置信区间；不固定 GPU 时钟、不排他占用整机，不宣称服务吞吐或生产最优。
Fused loss 相对 eager 在此配置仅省约0.58 GiB；同预算 microbatch并未增大。

第三部分的 profiler 驱动决策：保留原生 Flash/大 GEMM；试验已有 norm/QK/activation
训练替换后做全参数梯度门槛，不接受 loss 接近但梯度失败的路径。对分块 loss，
profiling 指向 FP32 dW 临时输出/累加，随后改为直接 addmm_ 累加并融合 CE 梯度生成，
S2048 的旧/新 profiler 与计时保留；这些改善没有最终超过 compile，不包装成全面加速。

### 入口与复现

```bash
cd /data1/tzh/proj/qwen_operator_lab
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# 全部输出要求新文件名，不覆盖原证据
/data1/tzh/proj/systems/.venv/bin/python benchmarks/benchmark_full_training.py \
  --variant hf --seq 4096 --steps 8 --timing steady --output artifacts/train_hf_new.json
/data1/tzh/proj/systems/.venv/bin/python benchmarks/benchmark_full_training.py \
  --variant fused_loss --seq 4096 --steps 8 --timing steady --output artifacts/train_fused_new.json
/data1/tzh/proj/systems/.venv/bin/python benchmarks/repeat_full_training.py --tag new --rounds 3
/data1/tzh/proj/systems/.venv/bin/python benchmarks/sweep_training_budget.py --tag new --budget 28
/data1/tzh/proj/systems/.venv/bin/python benchmarks/validate_full_training.py \
  --seq 128 --output artifacts/train_gradient_new.json
/data1/tzh/proj/systems/.venv/bin/python -m pytest -q
```

Loss API 接收已做 causal shift 的 `hidden [N,H]`、`weight [V,H]`、`labels [N]`：
`training.fused_loss.fused_linear_cross_entropy(..., chunk_size=128, ignore_index=-100)`。
不要在其外再次 shift；示例模型接入位于 benchmark 的 `loss_fn`。支持共享 embedding 权重的
梯度正常合流；全忽略标签按 native mean CE 返回 NaN loss、零梯度。

Liger 是可选工程对照，本轮未安装或改动其依赖；没有将第三方未执行 benchmark 当作成果。
本轮实现不含梯度过滤、量化、完整训练收敛验证、分布式训练或全融合 backward。
如果继续研究，下一步是定位 HF 低精度舍入与替换 backward 的整模型偏差；不能先改宽数值门槛。

## 10. 后续迭代：HF 舍入对齐与 AdamW（端到端原型贡献已验收）

### 数值定位和保留的负结果

`training_layer_diagnostic_float32_v1.json` 与 `training_layer_diagnostic_bfloat16_v1.json`
在完整28层模型记录每层输出及输出梯度。FP32/math 下已有算子通过原门槛；BF16/Flash
在首个 block 就出现差异。FP32 与 BF16 backend 不同，不能把结果当作严格的纯 dtype 因果消融。
`training/hf_rounded.py` 保留 HF 的 cast 边界，先用原生重计算 backward 隔离问题。
SwiGLU 的该版本在 S128 完整模型 loss/参数梯度逐项一致，RMSNorm 仍未过关。

随后 `ops/triton/hf_swiglu_backward.py` 保留 `grad*up` 与 SiLU 的低精度边界，
完整模型最大参数梯度 relative L2 为3.85%；64步轨迹的最大训练loss差0.00577、
独立test平均loss差0.00055，整体参数relative L2约4.83e-5，均通过预设初筛。
但 S512 三轮 whole-step 与原生基本持平，不把数值修复自动包装为加速。
相关证据：`hf_rounded_backward_unit_v2.xml`、`training_swiglu_hf_backward_diagnostic_v1.json`、
`rounded_swiglu_trajectory_v1.json`、`full_training_repeated_rounded_bwd_s*_v1.json`。

### 新的性能方向：单独替换 AdamW

完整step中原生fused AdamW约120 ms，明显大于单个activation阶段。
`training/triton_adamw.py` 保持相同BF16参数/梯度/动量存储，不量化、不加FP32 master，
只替换optimizer；forward/backward仍为原生HF+Flash，避免把之前的算子梯度偏差混入。

核对本机 `ATen/native/cuda/fused_adam_utils.cuh`，native超参数接口为double，
而候选用FP32标量算术。**公式/目标相同不代表bitwise相同**，不能隐瞒这项差别，
也不能仅根据源码接口就宣称所有原生慢点都是FP64造成；尚未做原生SASS归因。
候选以一个elementwise kernel读取参数、梯度、两类动量，计算后直接写回三者。

legacy初稿20步参数/动量测试通过，但更严格的每步更新量检查出现3.24%相对误差，
超过预设3%界限。该失败保存在 `triton_adamw_update_validation_v2.xml`。
rounded版本预先计算beta补数、关闭FMA并使用round-to-nearest除法，不放宽门槛，
通过 `triton_adamw_rounding_validation_v3.xml`（9项）和
`triton_adamw_api_validation_v4.xml`（14项）。旧版不作为当前推荐实现。

rounded版本64步不同文本窗口训练：最大train loss差0.00595，独立test平均loss差0.000583，
最终参数global relative L2为4.49e-5；`triton_adamw_rounded_trajectory_v2.json`。
这是短期轨迹证据，不是收敛到同质量所需时间或完整任务准确率保证。

使用 `TritonAdamW(..., math_mode='rounded')`，支持普通step/zero_grad/state_dict流程；
checkpoint保留math_mode。限制：单参数组、连续dense CUDA张量，不支持AMP scaling、
AMSGrad/maximize、稀疏梯度、CUDA Graph capture、分布式兼容性承诺。
默认HF训练入口不变。README给出接入示例。

最终回归 `adamw_contribution_regression_v1.xml`：**257 passed，2 warnings**。
`audit_adamw_contribution.py` 要求两个序列长度各三轮成功、输入/设备一致、rounded标记正确，
并检查更新量测试、轨迹和全回归；S512/S2048三轮均已结束，
`artifacts/adamw_verified_contribution_v1.json` 的最终审计已通过。

额外原生foreach AdamW对照：S512初筛约254.55 ms/step，其中optimizer约105.91 ms，
`full_training_foreach_adamw_s512_v1.json`。它不是随机三轮结果，也没有与native fused完全相同的
低精度中间舍入，不用它替代主要对照或宣称普遍最优。原始负结果与所有日志均保留。

### 最终完整训练 step 结果

Qwen3-1.7B完整28层、RTX A6000、B=1、BF16参数/梯度/Adam moments，
无FP32 master、无checkpointing、相同输入/权重/有效batch。每形状三轮随机进程顺序，
每进程2 warmup+8 steady step；表为三轮中位数的中位数。优化路径保留HF原生
forward/backward，只换optimizer，不使用先前未通过梯度门槛的算子替换。

| 序列长度 | HF + native fused AdamW | HF + compile + native AdamW | HF + rounded Triton AdamW | 相对HF耗时下降 |
|---:|---:|---:|---:|---:|
| 512 | 265.36 ms | 202.90 ms | 179.07 ms | 32.52% |
| 2048 | 493.83 ms | 397.92 ms | 409.59 ms | 17.06% |

同轮相对HF speedup：S512为1.404/1.490/1.480x，S2048为1.204/1.206/1.206x。
Optimizer阶段三轮中位数：native约116–121 ms，候选约36.4–36.5 ms。
显存基本持平，不把此优化写成显存压缩；S512 peak allocated约14 GiB，S2048约19.61 GiB。
相对compile：S512耗时约下降11.75%，S2048反而慢约2.93%。
结论是**在已测配置中，对常规HF训练存在完整step原型收益**，不是普遍胜过compile。
三轮不构成跨日复现，也未锁GPU时钟或独占整机；数据加载、tokenization、checkpoint写盘
不在计时内，不能把step提升直接等同于完整预训练作业加速。

### 使用、验收与复现

```python
from training.triton_adamw import TritonAdamW
optimizer = TritonAdamW(model.parameters(), lr=1e-5, math_mode="rounded")
# 保持原来的 forward、loss.backward()、step()、zero_grad() 和 checkpoint 流程。
```

```bash
cd /data1/tzh/proj/qwen_operator_lab
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
/data1/tzh/proj/systems/.venv/bin/python benchmarks/repeat_full_training.py \
  --tag adamw_new --seq 512 --rounds 3 --variants hf,compile,hf_adamw --adamw-math rounded
/data1/tzh/proj/systems/.venv/bin/python benchmarks/validate_training_trajectory.py \
  --candidate triton_adamw --adamw-math rounded --steps 64 --seq 512 \
  --output artifacts/adamw_trajectory_new.json
/data1/tzh/proj/systems/.venv/bin/python -m pytest -q
```

结果文件要求新tag/文件名，不能覆盖旧证据。完整复测文件为
`full_training_repeated_adamw_rounded_s512_v2.json` 和
`full_training_repeated_adamw_rounded_s2048_v2.json`，每个run指向独立原始JSON/log。
`audit_adamw_contribution.py` 检查九个run/形状、相同输入与GPU、同配置、rounded模式、
compile单图无graph break、每轮相对HF>1.05x，以及数值/轨迹/全回归。

接口加固期间改过checkpoint和unsupported-option检查，benchmark始终显式传入rounded；
未混用legacy算术结果。每个run保存其源码指纹，不声称所有历史run源码字节完全相同。
当前仍为opt-in，不改用户默认optimizer或安装依赖。保留单卡B1、64步轨迹、无AMP scaling/
多参数组/AMSGrad/分布式保证等限制；本节历史阶段尚未验证长程收敛，不能写成“训练质量完全无损”。

## 11. 完整训练作业已完成（2026-11-02）

后续实际跑完两组完整 Qwen3-1.7B/WikiText-103 持续预训练作业，按同一个预先冻结的验证平台规则停止：
Native 15750 步、Triton AdamW rounded 17500 步。不是固定1000步测试，也不是安全超时终止。
同等验证质量的实际观察耗时7208.42→5912.05秒（1.219×，耗时下降17.98%）；
各自平台停止且完成最终评估的完整进程耗时8320.38→7682.28秒（下降7.67%）。
选定checkpoint测试loss为2.39936/2.39764，最后checkpoint为2.39739/2.39594。

只替换optimizer，HF前后向与Flash SDPA保持不变。单GPU顺序作业对、一个种子，
共享预处理未计时并排除，不宣称从数据下载到收敛的总流程加速，也不宣称优于compiled HF。
数据哈希、全部验证记录的停止规则重放、四份checkpoint元数据均已验证；最终GPU回归270项通过。
详细协议、结果、checkpoint选择语义、计时修复及限制见
[完整训练报告](TRAINING_CONVERGENCE_RESULTS.md)，原始证据在`artifacts/convergence_comparison_v1/`。
