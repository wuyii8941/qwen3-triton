# Qwen3：Static KV Cache / CUDA Graph / QKV packing / MLP 融合

这一轮在**真实 Qwen3-1.7B、28 层、FP16、RTX A6000** 上测试四个方向。
保留原参数及旧路径，全部新增选项默认关闭；没有改训练 backward，也没有改环境依赖。

## 1. 四个实验分别改变什么

| 实验 | 实现位置 | 实际改变 | 代价或限制 |
|---|---|---|---|
| Static cache + CUDA Graph | `model/static_decode.py` | 预分配 BSHD cache；捕获单 token 的完整模型 forward，包含 LM head | 固定 batch/capacity/dtype；初始化与捕获有成本 |
| Packed QKV | `model/hf_inference.py`、`model/hf_attention.py` | 离线拼接 Q/K/V 权重；每层三次 Linear 变一次；split 是 view | 当前保留原权重，额外 **448 MiB** |
| Prefill GEMM + SwiGLU | `make_mlp_forward()`、`ops/triton/gemm_tuning.py` | 使用已有调优的 Triton fused gate/up GEMM + activation，保留 down projection | 仅 hidden=2048、sequence>1；独立开关，不保证所有 shape 更快 |
| QK norm/RoPE 直接写 K cache | `ops/triton/hf_qk_norm_rope.py` | K 分支直接写预分配位置；不生成临时 K 再 index_copy | 依赖 static cache；V 仍单独写入 |

QK kernel 增加输入 stride，直接读取 packed QKV 的切片，**没有为了 packing 又做三份 contiguous copy**。
K 的写入位置来自 GPU 上的 `cache_position`；RoPE 的位置仍由 HF 生成的 cos/sin 决定。

### 为什么不是直接换成 HF StaticCache

当前 PyTorch 2.5.1 + Transformers 4.51.0 下，HF StaticCache 返回容量长度的 K/V，
并生成显式 mask。这个路径不能直接沿用本实验强制的 Flash SDPA。

这里采用预分配 `[B, capacity, Hkv, D]` 缓冲区，以及 PyTorch 自带 Flash 的
**varlen 私有接口** `aten._flash_attention_forward`。累积偏移描述每个 batch 的存储区间，
GPU `seqused_k` 描述实际有效长度。未使用的尾部不参与 attention。

这是有意的、锁定版本的实验依赖，而非通用公共 API：

- [PyTorch 2.5.1 Flash 分发源码](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/transformers/cuda/attention.cu)
- [Flash varlen 接口源码](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/transformers/cuda/flash_attn/flash_api.cpp)

注意：`seqused_k` 只在传入 cumulative sequence offsets 的 varlen 分支生效。
不能给 dense 分支加上这个参数后就假定尾部被屏蔽。

## 2. 测量口径

- `hf`：未修改的 HF + 强制 native Flash SDPA。
- `best`：上一轮最佳路径，`decode_mlp=True, fuse_qk=True, compact_gqa=True`。
- `static`：上述基线 + static cache，**无 Graph**。
- `graph`：static cache + Graph，无 packed QKV、无直接写 K。
- `graph_packed` / `graph_direct`：分别只增加一项，用于独立归因。
- `graph_both`：Graph + packed QKV + 直接写 K。
- 第二轮增加 `graph_all`：在 `graph_both` 上再开启 prefill MLP 融合。

每轮 16 次随机交错测量，另有 3 次预热；独立进程复测时更换输入随机种子。
每个请求包含一次 prefill 和 **16 次单 token greedy decode forward**，不是把固定 KV 状态重复执行 16 次。
decode 含 argmax、cache 更新、输入复制、position/length 更新、Graph replay 和最终同步。
每侧都计算最后位置 logits，attention 都使用原生 Flash，没有替换成 math backend。

`pipeline` 是完整模型 wall-time，包含 prefill 到 decode 边界的一次同步，用于区分 TTFT；
不包含 tokenizer、模型加载、权重 packing、JIT、Graph capture。
这些 setup 成本、缓存常驻字节数和额外 packing 字节数另行记录。
Graph 只捕获 decode，**没有捕获 prefill**。

CUDA Events 与 wall-time 分开记录。Profiler 单独运行，不参与计时。
JSON 包含所有样本、随机顺序、paired bootstrap CI、GPU/软件版本及源码/权重/config SHA256。
CI 只描述这一进程内的交错样本，不是跨机器/跨天置信度；没有锁 GPU 时钟。

## 3. 第一轮结果

完整 pipeline 中位数，单位 ms；随机种子 20260930。

| B / prompt | 上一轮 best | static，无 Graph | graph | graph_both | best / graph_both |
|---|---:|---:|---:|---:|---:|
| 1 / 128 | 263.872 | 270.590 | 113.542 | 110.098 | 2.40× |
| 2 / 128 | 296.555 | 324.553 | 122.718 | 116.960 | 2.54× |
| 1 / 512 | 311.100 | 323.374 | 121.008 | 117.503 | 2.65× |

这里的倍数是中位数之比；JSON 的 paired median speedup 是另一统计量，不要求逐位相等。

几个重要的负结果/边界：

- **静态 cache 单独使用没有加速**。它主要为稳定地址和 Graph 捕获提供条件。
- 大部分端到端收益来自 Graph 减少 Python/CUDA 调度间隙，不能全部归功于 Triton kernel 本体。
- Packed QKV 和直接写 K 是增量优化，不是上述 2.4–2.6× 的主要来源。
- Prefill MLP 在 B=1/S=512 下，prefill 22.433 → 20.565 ms；但 B=1/S=128 的 CI 跨过 1，不能宣称所有 shape 都有稳定收益。
- 只优化 prefill MLP、保持 eager decode 的整段 16-step pipeline，没有稳定额外收益。

第一轮 FP16 所有候选通过已有相对 L2 < 2% 的门槛，最大约 **0.354%**。
固定随机输入的 prefill + 8 步 teacher-forced logits argmax 一致，自然文本 32-token smoke 一致。
这些不是任务质量评测，也不是逐位一致的承诺。

### 第二轮：不同输入种子、最终代码与 profiler

独立进程，种子 20261001；增加防止在错误 context 中使用 runner 的检查，
并增加 `graph_all` 的组合验证。源码哈希与当前文件一致。所有候选再次通过数值门槛。

| B / prompt | best pipeline | graph_both pipeline | graph_all pipeline | graph_all / step（仅 decode） |
|---|---:|---:|---:|---:|
| 1 / 128 | 268.268 ms | 110.463 ms | 110.153 ms | 5.852 ms |
| 2 / 128 | 332.792 ms | 118.844 ms | 118.715 ms | 6.126 ms |
| 1 / 512 | 322.324 ms | 117.806 ms | 115.962 ms | 5.975 ms |

第二轮组合路径约为此前 best 的 **2.44× / 2.80× / 2.78×**。
两轮中 `graph_both` 的完整 pipeline 中位数之比约 **2.40–2.80×**。
主机调度存在波动，不将两轮结果合并成一个精确到小数点后的普适加速比。

相对 `graph` 的单项消融（第二轮 paired median，pipeline）：

- Packed QKV：约 **1.022–1.039×**；需要额外 448 MiB，是否值得取决于显存预算。
- 直接写 K：约 **1.007–1.010×**；不需要额外 packed 权重。
- 两项组合：约 **1.030–1.051×**。
- Prefill MLP 在 S=512 时进一步把组合 pipeline 的 117.806 ms 降至 115.962 ms；
  短 prompt 差别很小，保留为可选项。

第二轮 `prefill_mlp` 的独立 prefill：B=1/S=512 为 22.491 → 21.231 ms。
两轮都观测到长一些的 prefill 收益，但不能声称它显著改变 decode 性能。

Profiler 独立采集 B=1、prompt=128 后的一步 decode：

| 路径 | GPU kernel 数量 | kernel 时间之和（仅诊断） |
|---|---:|---:|
| best，动态 cache | 461 | 6105 μs |
| static，无 Graph | 462 | 6048 μs |
| graph | 465 | 5964 μs |
| graph_packed | 409 | 5882 μs |
| graph_direct | 437 | 5903 μs |
| graph_both | 381 | 5822 μs |

Packed QKV 少 **56** 次投影 kernel，即每层 3 → 1。
直接写 K 少 **28** 次 `index_copy` kernel，即每层省一次 K 复制，V 的复制仍在。
Graph 路径的额外输入/状态维护也计入 profiler，所以 kernel 总数没有凭 Graph 自动减少。
这些 GPU kernel 总时间只有约 6 ms，而 eager best 单步 wall-time 约 15.7 ms：
与 static/eager/Graph 的消融一起看，主要收益来自调度间隙减少，而不是 attention 算法被换掉。

第二轮 FP16 最大 logits 相对 L2 约 **0.344%**；两轮最大约 **0.354%**。
所有 teacher-forced argmax 和自然文本 smoke 都与 HF 一致。
在所测 logits 上，单独 static / Graph / direct K 相对旧 best 为逐位一致；
packed QKV 和 MLP GEMM 可以改变浮点累加顺序，不要求逐位一致。

显存/初始化需要单列：本轮 static cache 容量 256/256/1024，对应三个 case 的
cache 本体 **28 / 56 / 112 MiB**。Graph 还保留中间缓冲区；setup 的 allocated delta
记录在 JSON，但会受共享 workspace 的初始化顺序影响，不等于独立部署的总显存。
第二轮 `graph_both` 的 setup 约 177–208 ms，含一次 prefill、warmup 和 capture，
不含独立的 packing；第一轮冷 JIT 更慢，不能把 warm setup 当作首次启动耗时。

## 4. 如何调用

从项目根目录运行；模型必须是 CUDA FP16、eval 状态。不要给这个 runner 传 padding mask。

```python
import torch
from transformers import AutoModelForCausalLM
from torch.nn.attention import sdpa_kernel, SDPBackend
from model.hf_inference import qwen3_triton_inference
from model.static_decode import StaticDecodeRunner

model = AutoModelForCausalLM.from_pretrained(
    '/data1/tzh/models/Qwen/Qwen3-1.7B', local_files_only=True,
    torch_dtype=torch.float16, attn_implementation='sdpa',
).cuda().eval()
# input_ids: 无 padding 的 [B, S] CUDA int64，同 batch 中长度相同。

with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION), \
     qwen3_triton_inference(model, decode_mlp=True, fuse_qk=True,
                           compact_gqa=True, static_decode=True,
                           packed_qkv=True, direct_k_cache=True):
    runner = StaticDecodeRunner(model, input_ids.shape[0], capacity=1024)
    logits = runner.prefill(input_ids)
    runner.capture(logits[:, -1].argmax(-1, keepdim=True))  # setup，不放进稳态计时
    for _ in range(16):
        token = logits[:, -1].argmax(-1, keepdim=True)
        logits = runner.decode(token)
```

`capacity >= prompt_length + decode_steps`。不足会在 host 检查时报错，不会静默覆盖 cache。
同一个 runner 可重新 `prefill()`，复用已捕获 Graph，但 batch、模型、dtype、capacity 不可改变。
若不想增加 448 MiB，关闭 `packed_qkv`；Graph 和直接写 K 仍可使用。
默认不启用 `prefill_mlp`，可单独开启观察对应 workload。

Graph 输出 logits 是复用缓冲区；跨 replay 保存结果请 `.clone()`。
权重 packing 是推理快照，修改/重新加载模型参数后必须重建，不能复用旧 packed 权重。
`pack_qkv_weights(model)` 可提前准备并作为 `packed_weights=` 传入 context，避免反复 packing。

## 5. 复现与限制

```bash
cd /data1/tzh/proj/qwen_operator_lab
/data1/tzh/proj/systems/.venv/bin/python -m pytest -q
/data1/tzh/proj/systems/.venv/bin/python -m benchmarks.benchmark_static_decode_e2e \
  --repeats 16 --steps 16 --tag _new
```

结果文件已存在时会拒绝覆盖，请选择新的 `--tag`。

- 实现支持 FP16/BF16 小模型测试，但本次真实模型验收限定 **FP16**。
- 先前 BF16 全模型已有超过 2% 的误差反例，本次没有通过改门槛将其“验收”。
- 仅初始 prefill + 连续单 token decode；不支持 chunked prefill、padding、不同请求长度、beam、paged cache、滑动窗口或动态 RoPE scaling。
- 独立 runner，不是 HF `generate()` 的通用 StaticCache 替代品；当前固定长度 decode 不做 EOS early exit。
- 全局 RoPE patch 仍限定单线程实验；不支持并发推理、训练或权重热更新。
- CUDA Graph 是运行时调度优化，native Flash 是复用的；不是“全模型全 Triton”，也没有与 compiled HF/vLLM 对比。
- 内存 peak delta 不含已预分配的 cache、graph pool 和 packed 权重；不能把它当成总推理显存。

证据：`artifacts/static_decode_e2e.json`（第一轮）；
`artifacts/static_decode_e2e_recheck.json`（第二轮，含 profiler 和新增组合）。
第二轮复现参数：`--repeats 16 --steps 16 --seed 20261001 --profile --tag _new_recheck`。
新增测试：`tests/test_static_decode.py`，覆盖有效前缀/污染尾部、改变输入与位置的 Graph replay、
full vs incremental logits、短 prefill、重新 prefill、容量边界、异常恢复与真实宽度 MLP。
完整回归：**137 passed**；有一个旧 attention backward 测试的 cuBLAS context 初始化 warning。
