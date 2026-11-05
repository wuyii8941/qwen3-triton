# 现有 Triton kernel 内部优化：实测与取舍

日期：2026-09-28。RTX A6000，PyTorch 2.5.1+cu121，Triton 3.1.0。
本轮实际尝试了此前讨论的五项：split-KV、GQA 复用、GEMM tile/流水线、
GEMM program 排序、RMSNorm/RoPE 多行处理。不是仅新增几个算子名称。

## 1. 测量口径

- 共 496 条算子候选记录，均先检查数值，再计时。包含 FP16/BF16。
- CUDA Graph 中捕获 10 次调用，交替正反顺序 replay 9 轮，以每次调用的 GPU 时间中位数比较。
- 记录首调用/JIT 时间、9 个样本、最小/最大值、相对 L2、最大绝对误差与 peak allocated delta。
- 首次编译不计入 steady-state；计入 kernel 中的归并与复制，不计 Python 调度。
- warm-cache 固定形状，未锁定 GPU 时钟；不是完整 Qwen 的速度，也不是服务吞吐。
- 每个 case 的最佳值来自有限候选扫描，属于在这些 shape 上的调参结果，不代表泛化最优。
- JSON manifest 含种子、环境、driver 和相关源码 SHA256；该 hash 标识测量时版本。

## 2. Decode attention：split-KV + GQA 共享

代码：`ops/triton/decode_split_kv.py`，入口 `attention_decode_split`。

旧版每个 `(batch,query_head)` 只有一个 program，顺序遍历整段 cache。
新版把 KV tiles 分给多个 program，输出 FP32 partial 与 log-sum-exp，再用一个
归并 kernel 计算最终输出。不能直接平均各段 attention 输出。

```text
Q + KV 分段 → 每段 normalized output 与 LSE
           → 以 exp(LSE - max_LSE) 为权重合并
```

`group_heads=True` 时，一个 program 同时处理共享 K/V 的 query heads；
`False` 则每个 query head 独立。两种都不构造重复 K/V。
支持静态 cache 的 stride 与非零 storage offset，不先复制完整 cache。

### BF16、B=1、Hq/Hkv=16/8、D=128，μs

| KV 长度 | 旧 Triton | 最佳新版 | 配置 | 旧/新 | Flash SDPA |
|---:|---:|---:|---|---:|---:|
| 128 | 9.216 | 9.933 | 不分组，2 splits | 0.93× | 10.138 |
| 512 | 27.546 | 13.312 | 不分组，8 splits | 2.07× | 9.830 |
| 2048 | 150.835 | 36.250 | 分组，16 splits | 4.16× | 22.733 |
| 8192 | 593.818 | 120.934 | 分组，32 splits | 4.91× | 60.928 |

结论：长 KV 的旧 Triton 得到明显改善，但仍未超过 Flash SDPA；短 KV 不值得强制拆分。
不能把最后一列隐藏后宣称超过成熟 attention 库。

### GQA 共享本身的消融

保持分片数相同，BF16 B=1：

| KV/splits | 不共享 | 共享 | 观察 |
|---|---:|---:|---|
| 512/8 | 13.312 | 13.619 | 无收益 |
| 2048/16 | 41.062 | 36.250 | 约 12% 更低延迟 |
| 8192/32 | 130.867 | 120.934 | 约 8% 更低延迟 |

共享减少 K/V 加载，但增加 program 内的计算/状态，不能推断所有长度都更快。
尚未用硬件计数器证明 register occupancy 或 DRAM 流量变化，这里只报告延迟证据。
32 splits、B=1 的 partial/LSE 为 258 KiB，加输出后实测新增峰值分配 262 KiB。
这是用小 workspace 换并行度，不是零 workspace 优化。

接入已有 pipeline（默认仍为 legacy；不替换真实 HF 的 Flash SDPA）：

```python
from ops.triton.attention_pipeline import CompactGQAAttentionPipeline
pipeline = CompactGQAAttentionPipeline(
    q_heads=16, kv_heads=8, decode_impl='split_kv',
    decode_splits=32, decode_group_heads=True,
)
# pipeline.prefill(...) 不变；pipeline.decode(...) 使用新路径。
```

限制：一个 query token、所有缓存位置可见、无 padding/mask、无 backward；分组比例支持 1/2/4/8。
独立 FP32 oracle 测试包含 S=1/129/513/1025、不同 batch/GQA、静态 cache 连续追加；
benchmark 还覆盖长度 8192，并包含完整 merge 时间。

## 3. GEMM + SwiGLU：tile、warps、stages

扩展的是已有 `ops/triton/gate_up_swiglu.py` 的同一个 kernel。
`ops/triton/gemm_tuning.py::gate_up_tuned` 绕过 autotune，显式指定配置，便于消融。
旧训练/autograd 调用继续使用原默认参数。

扫描包含：BM=16/32/64，BN=32/64/128，BK=32/64/128，warps=4/8，stages=2/3/4
的选定组合（**不是全笛卡尔积**）。当前候选新增 `HF_ROUND=True`，在投影与 SiLU 后
保留低精度舍入；旧 `original_autotune` 仍保留历史 FP32 中间值语义。因此还提供
相同 HF 舍入下的固定 tile 对照，避免只把两套语义不同的代码视作完全相同实现。

### BF16、N=6144、K=2048，μs

| M | 旧 autotune | 新版最佳 | 配置 BM/BN/BK/warps/stages | cuBLAS×2 + Triton 激活 |
|---:|---:|---:|---|---:|
| 1 | 126.362 | 75.366 | 16/64/64/4/3 | 79.974 |
| 16 | 126.669 | 76.083 | 16/64/64/4/3 | 81.408 |
| 128 | 132.608 | 110.080 | 64/64/64/4/3 | 137.011 |
| 512 | 282.522 | 253.030 | 64/128/32/8/3 | 330.547 |
| 1024 | 506.778 | 485.171 | 64/128/32/8/3 | 564.736 |

FP16 M=128/1024 的 cuBLAS 对照更快；不能从 BF16 表推断所有 dtype/shape 都胜过 cuBLAS。
这里是随机权重和输入的算子实验，不是完整 MLP（down projection 不在范围内），也不是完整模型。
对照不是最佳 packed GEMM epilogue，更不是 compiled HF。

同一套 HF 舍入、BF16 M=1 时，32/64/32/4/3 与 16/64/64/4/3 分别为
116.429 与 75.366 μs，约 1.54×，可以隔离旧版中间值舍入差异。
调整 warp/stage 也没有单调收益，例如 M=1024，固定
32/64/64/4 时 stages=2/3/4 为 942.39/869.38/978.23 μs。增大 stages 并非越多越好。

示例：

```python
from ops.triton.gemm_tuning import gate_up_tuned
with torch.inference_mode():
    y = gate_up_tuned(x, gate_weight, up_weight,
                     bm=16, bn=64, bk=64, warps=4, stages=3)
```

此入口仅 forward/inference；没有让旧 backward 偷偷对应新的 HF 舍入公式。

## 4. GEMM program 排序：负结果

同 tile 32/64/64、warps=4、stages=3，BF16 M=512：

| 顺序 | μs |
|---|---:|
| 旧二维 grid | 457.01 |
| GROUP_M=1 | 1075.81 |
| GROUP_M=4 | 465.72 |
| GROUP_M=8 | 463.87 |

这次没有证明 grouped ordering 更快。旧二维 grid 的 M 维本来就能连续遍历，
并不是完全没有权重复用；调换顺序反而可能破坏局部性。没有 L2 计数器证据，
不把原因写成“已证明命中率下降”。保留参数用于实验，生产选择不启用它。

## 5. RMSNorm / RoPE 多行处理

代码：`ops/triton/rowwise_tuning.py`。

### BF16 RMSNorm，μs

| M,N | 旧版 | 新版最佳 | 行数/program |
|---|---:|---:|---:|
| 2048,128 | 4.608 | 3.277 | 4 |
| 16384,128 | 19.354 | 14.131 | 16 |
| 2048,2048 | 27.034 | 26.419 | 8 |
| 16384,2048 | 198.861 | 198.758 | 1 |

值得保留的是大量短行 Q/K norm，约 1.37–1.41×；N=2048 的变化很小。
真实 HF 的可选开关 `qwen3_triton_inference(model, multirow_norm=True)` 只在
N=128、M≥2048 时尝试多行，其他情况保留旧 kernel。默认关闭。
接入验证不是完整模型加速的保证，单独保留其端到端消融报告。

精简公共 wrapper 检查、在模型中直接 launch 后，完整 Qwen prefill 的同步
wall-time 如下（ms，20 次交错采样，use_cache=True、logits_to_keep=1）：

| dtype | B,S | 现有 Triton | 多行 norm |
|---|---|---:|---:|
| FP16 | 1,128 | 23.108 | 23.313 |
| FP16 | 2,128 | 24.241 | 24.037 |
| FP16 | 1,1024 | 35.812 | 35.999 |
| BF16 | 1,128 | 20.959 | 21.457 |
| BF16 | 2,128 | 24.457 | 24.520 |
| BF16 | 1,1024 | 37.388 | 37.424 |

**未观察到稳定端到端收益，因此不启用默认替换。** 首次走公共 wrapper 的
版本在短 prefill 也略慢；精简后仍不能支持“完整模型更快”的结论。
没有进一步的 CPU profiler 因果证据，不能把全部差异归因于 Python 调度。

RoPE 已有 HF kernel 使用 512 元素扁平 tile，本身已经处理多行。
再次扫描 block=128/256/512/1024/2048 与 warps=4/8 后：BF16 B=2,S=512
由 15.462 降到 14.950 μs，其他形状基本持平。因此不替换现有默认。
新测试使用非连续输入和非对称 cos/sin，验证与 HF 逐位一致。

## 6. 复现与证据

```bash
cd /data1/tzh/proj/qwen_operator_lab
export PYTHONPATH=.
/data1/tzh/proj/systems/.venv/bin/python -m pytest -q
/data1/tzh/proj/systems/.venv/bin/python benchmarks/benchmark_kernel_tuning.py decode
/data1/tzh/proj/systems/.venv/bin/python benchmarks/benchmark_kernel_tuning.py rows
/data1/tzh/proj/systems/.venv/bin/python benchmarks/benchmark_kernel_tuning.py gemm
/data1/tzh/proj/systems/.venv/bin/python benchmarks/benchmark_hf_multirow.py --output hf_multirow_direct.json
```

不要同时运行多个 GPU benchmark。

- `artifacts/kernel_tuning_decode.json`：176 条。
- `artifacts/kernel_tuning_rows.json`：160 条。
- `artifacts/kernel_tuning_gemm.json`：160 条。
- `artifacts/hf_multirow_ablation.json`：首次通过公共检查包装器接入的模型消融。
- `artifacts/hf_multirow_direct.json`：精简包装、直接 launch 后的模型复测。

所有候选先通过误差门槛再计时。107 项回归覆盖历史路径与新增路径，但不代表
所有任意 shape、训练梯度、任务质量或并发服务都已验收。
