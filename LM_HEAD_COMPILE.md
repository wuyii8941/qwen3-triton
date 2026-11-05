# LM head、全图编译与分阶段执行实验

环境：RTX A6000，Qwen3-1.7B，FP16，PyTorch 2.5.1+cu121 / Triton 3.1.0 /
Transformers 4.51.0。原权重、宿主 Python 和默认推理开关均未改动。
所有性能数值是本机指定 workload，不代表其他 GPU 或生产服务。

## 1. 为什么继续看 LM head

大词表输出投影为 `[B,2048] @ [2048,151936]`。原 FP16 权重约 593.5 MiB，
小 batch 下读取权重占比很大。已有 Graph decode 每步 325 个 kernel，继续减少
小算子 launch 的空间有限；减少这个大投影的权重搬运更值得验证。

新增 `ops/triton/linear_decode.py::linear_decode`：

- FP16/BF16 激活，原权重或可选分组 INT8 权重；FP32 reduction。
- 支持 M=1..8、K≤8192；本次真实 shape 为 K=2048、N=151936。
- 每个 program 处理若干输出行；词表 tile 放在 grid.x，避免 grid.y 的 65535 限制。
- `share_batch=True`：权重 tile 加载/反量化一次，在同一个 program 内处理多个 batch 行。
- INT8 oracle 是显式反量化后的权重；与原模型的量化误差另验，不能混为一谈。

真实 checkpoint 权重的独立算子扫描，单位 μs，取各自扫描中的最快配置：

| Batch | 原生 Linear，连续输入 | W8 G64，不复用 batch | W8 G64，复用 batch |
|---:|---:|---:|---:|
| 1 | 881.6 | 471.5 | 470.0 |
| 2 | 885.3 | 937.0 | 512.3 |
| 4 | 889.8 | 1865.9 | 731.6 |

这是两次独立参数扫描，不是同一组交错对照；它解释了为何不能把 B1 的实现直接推广到 B4。
最终接入使用 B≤2 的 BN=4 / 4 warps，B≥4 的 BN=8 / 4 warps。
原 FP16 权重的自写 kernel 对连续输入与 cuBLAS 基本持平，没有强加速结论。

G64 权重+scale 约 315.3 MiB；B1 470 μs 对应约 703 GB/s 的估算权重读取带宽。
这个估算忽略小激活/输出流量，不是硬件带宽计数器。收益主要来自减少读取字节，
不是 INT8 Tensor Core：当前 kernel 反量化后仍做浮点乘加。

证据：`artifacts/lm_head_tuning.json`、`lm_head_tuning_shared.json`。

## 2. 无量化机会：最后一个 token 的输入布局

HF 的 `hidden_states[:, -1:, :]` 在 B>1 时仍保留跨 batch 的 `S*H` stride。
因此原生 `Linear` 在被测版本走了 `aten::bmm`。对这个很小的激活先 `.contiguous()`，
可转成 `aten::mm`，避免该 workload 中低效的批量投影路径。

| Batch / S=128 | 原生 strided | 连续化 + 原生 Linear | 连续化 + 自写 FP16 |
|---:|---:|---:|---:|
| 1 | 881.25 μs | 881.36 μs | 875.93 μs |
| 2 | 1935.15 μs | 889.54 μs | 882.57 μs |
| 4 | 3860.81 μs | 893.01 μs | 889.13 μs |

表中**包含复制成本**，输入来自真实模型 hidden states，使用相同原权重。
B4 连续化仅搬运 16 KiB 激活。Profiler 确认 B2/B4 的 `bmm → copy + mm`。
这也修正了归因：自写 FP16 LM head 在某些 prefill 中变快，不全是 kernel 本身更好，
主要是原生 baseline 的输入布局问题。decode 输入本来连续，没有同等收益。

接口：`qwen3_triton_inference(..., lm_head_impl='contiguous')`。
证据：`artifacts/lm_head_layout.json`；含算子 dispatch 和全部重复计时。

## 3. 真实模型 pipeline，不只看算子

基线 `graph_ref` 已包含此前 Graph、packed QKV、QK/RoPE、直接 K/V cache 写入、
prefill fused GEMM 和 residual/norm 融合；不是最原始 HF。
完整 prefill + 16 步 greedy decode，包含 LM head、argmax、cache/input 更新和同步；
排除加载、JIT、packing、量化快照和 Graph capture。初始化成本在 JSON 单独记录。

第一轮 seed=20261007，16 个交错 round，单位 ms：

| B / prompt | 已优化 Graph | 仅 LM head G64 | 旧 MLP G16 分流 | 两者组合 |
|---|---:|---:|---:|---:|
| 1 / 128 | 108.887 | 102.004 | 98.201 | 90.902 |
| 2 / 128 | 116.847 | 109.070 | 116.938 | 109.249 |
| 1 / 512 | 114.226 | 107.376 | 104.521 | 97.759 |
| 1 / 2048 | 170.315 | 163.028 | 161.126 | 154.626 |
| 4 / 128 | 123.447 | 117.756 | 123.591 | 117.526 |

LM head 单项相对已有 Graph 的完整延迟约下降 4.3–6.7%，五组 paired CI 都高于 1。
组合在 B1/S128 约下降 16.5%，但包含**两种近似量化**，不能写成数值等价 fusion 收益。
B2/B4 的 MLP 分流不执行量化，所以组合实质上只启用了 LM head 量化。
Profiler 仍是每步 325 kernels；这次不是减少 launch 数量。

数值与反例：

- 第一轮各候选随机 logits 均满足原有 relative L2 < 2% 门槛。
- 仅 LM head G64 最大约 0.535%，被测 argmax 与自然文本 smoke 均匹配 HF。
- G32 在 B1/S512 有一次 argmax 不同；更小的总体 L2 并不保证 top-1 相同。
- 组合最大约 0.784%，但 32-token 自然文本 smoke 与 HF 分歧，已保留原 tokens/text。
  前段解释一致，后续续写开始不同。不能称“质量无损”或“输出完全一致”。
- 2% 是工程回归门槛，不是量化质量认证；尚无公开任务准确率或困惑度验收。

显存：保留原始 tied embedding/LM head 权重，G64 **额外**常驻 315.3 MiB 快照，
G32 333.8 MiB。叠加 MLP G16 再额外 840 MiB，packed QKV 另占 448 MiB。
因此这不是减少模型总显存的方案；权重变更后必须重建快照。

### 第二个种子的复测

seed=20261008，12 个交错 round；新增原生连续化消融。完整流程 ms：

| B / prompt | 已优化 Graph | 原生 LM head 连续化 | 仅 LM head G64 | 两种量化组合 |
|---|---:|---:|---:|---:|
| 1 / 128 | 108.948 | 109.003 | 102.074 | 91.320 |
| 2 / 128 | 116.592 | 115.780 | 108.846 | 108.910 |
| 1 / 512 | 113.801 | 113.770 | 106.856 | 96.821 |
| 1 / 2048 | 169.671 | 169.916 | 162.963 | 154.734 |
| 4 / 128 | 123.916 | 120.824 | 117.930 | 117.607 |

连续化在 B1 无明显收益，B2 完整流程约下降 0.7%，B4 约下降 2.5%。后两者 paired CI
高于 1；B4 prefill 为 23.115→20.107 ms。该路径相对 `graph_ref` 最大 logits L2 约
0.00365%，argmax/smoke 一致；mm/bmm 的累加路径有细微差别，不宣称逐位一致。
这比自写 FP16 kernel 更简单，且不量化，适合优先保留为可选开关。

LM head G64 的完整延迟再次下降约 4.0–6.6%，五组 CI 高于 1；本轮最大 L2 为 0.543%，
argmax/smoke 匹配。组合最大 L2 为 0.667%，自然文本仍分歧。两轮随机检查不是广泛质量保证。
新增峰值分配 delta 与基线相同，**未包含量化快照的额外常驻内存**；不要据此宣称节省总显存。
证据：`artifacts/static_decode_e2e_lm_head_v2.json`，第一轮为 `static_decode_e2e_lm_head.json`。

## 4. 完整 compile 对照：成功编译也不一定更快

纯 PyTorch reference 仅适配 static cache / compact GQA；norm、RoPE、MLP、投影均为
HF/PyTorch。它不是未经修改的 `HF generate()`，但与 Triton 路径共享相同 cache/Flash 契约。
所有路径都使用手动 CUDA Graph decode；compiled 版本同时编译 prefill 和 decode。
不把某个路径首次编译时间算进稳态，也不让另一条路径独占 CUDA Graph。

实现：`model/torch_static_reference.py`、`ops/compile_ops.py`、
`benchmarks/benchmark_compile_pipeline.py`。

修复了两个真实障碍：

1. PyTorch 2.5.1 Triton mutation analysis 未正确处理 `enable_fp_fusion=False`，
   错把所有输入（包括权重）当作被修改。用有明确 mutation schema 和 fake 实现的
   custom ops 封装；保留所有舍入边界，没有移除该选项，也没改安装的 PyTorch。
2. 每轮重新绑定 context 中的 forward closure 会触发重新编译。改为稳定的独立模块树，
   共享只读原参数，测试期间 context 不反复重绑。

`fullgraph=True`；无回退；统计 backend 调用次数，稳态若发生重编译则标为无效。
Custom ops 是 compiler 的不透明边界，**不声称跨它们完成了编译器融合**。
失败尝试见 `artifacts/compile_failure_notes.json`，没有把其耗时当作性能结果。
准确声明输入修改和禁止输出 alias 的要求，参见
[PyTorch 2.5.1 custom-op 官方实现](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/_library/custom_ops.py)。
当前编译验收覆盖报告中的无量化 fused-QK/static-cache 路径，不承诺所有历史实验开关均可 fullgraph 编译。

第一轮有效全图对比（12 rounds，ms）：

| B1 / prompt | PyTorch + Graph | PyTorch compile + Graph | Triton + Graph | Triton compile + Graph |
|---|---:|---:|---:|---:|
| 128 | 155.906 | 108.540 | 108.483 | 113.753 |
| 512 | 168.706 | 121.956 | 114.804 | 119.402 |

四条路径均过数值门槛；compiled 路径每 case 都是 prefill/decode 两张图，计时中零重编译。
单步 GPU kernel 数分别 1648 / 430 / 325 / 327。
短 prompt 的完整流程与 compiled PyTorch 基本持平，不能再把对 eager 的巨大倍率写成
对 compile 的倍率。S512 的 Triton Graph 更快；Triton 再 compile 并没有带来新收益。
编译/缓存查找 + warmup + capture 的一次性成本约 26–51 秒，**不保证全冷编译**，
因为本机保留编译缓存；模型加载、packing 不包含在该 setup 数字中。

原理线索：S128 时 compiled PyTorch prefill 12.389 ms，Triton prefill 16.194 ms；
decode 则 Triton 更快。S512 时现有 Triton prefill 反而更好，因此不能默认“一律 compile”。

证据：`artifacts/compile_pipeline_custom_ops.json`。

## 5. 混合路径：编译 prefill + Triton Graph decode

新增 `model/hybrid_decode.py::HybridDecodeRunner`，两棵独立模型模块树共享原始 Parameter，
prefill 和 decode **共用同一个 StaticDecodeCache 对象与 K/V 地址**，没有把 KV 再拷贝一遍。
只编译纯 PyTorch prefill；Triton decode 仍使用已有 CUDA Graph，不重复编译。
两个 inference context 必须在运行期间保持开启，不支持并发使用这两个 runner。

完整 Qwen3-1.7B 的第二轮全图对照，B1、12 个交错 round，ms：

| Prompt | PyTorch compile + Graph | 现有 Triton + Graph | 混合路径 |
|---:|---:|---:|---:|
| 128 | 109.103 | 108.911 | 104.808 |
| 512 | 121.723 | 113.976 | 117.597 |

- S128：相对现有 Triton pipeline 延迟下降约 **3.8%**，相对 compiled PyTorch 约 **3.9%**。
  相对 Triton 的 paired speedup 1.037×，95% CI [1.036, 1.042]。
- S512：相对现有 Triton **变慢约 3.2%**，因为原 fused GEMM prefill 更快。
  不能将混合路径默认推广到所有 prompt，更不能由两个 shape 宣称通用最优阈值。
- 两种长度的混合路径最大 logits L2 分别约 0.188% / 0.084%，所测 argmax 全部匹配。
  没有改变权重量化精度；由于浮点实现路径不同，不保证逐位一致。
- 混合路径每 case 只编译一张 prefill 图，计时阶段零重编译；四个原对照路径也零重编译。
- 稳态测量包含真实 prefill、共享 cache 的 16 步 decode 和 host 调度，
  **不是把分别测出的 prefill/decode 中位数相加**。

这轮无量化的选用建议：短 prompt 可试混合路径；S512 的本次测量继续用既有 Triton prefill。
B>1 的 last-token 连续化可独立试验。尚未交叉验证“混合 + LM head 量化”，
不把它们的百分比相加，也没有默认启用这种未验收组合。
证据：`artifacts/compile_pipeline_hybrid.json`，包含数值、计时、profiler、编译计数与 setup。

## 6. 接口与复现

在已有 `qwen3_triton_inference(..., static_decode=True, ...)` 参数上额外指定：

```python
# 不量化，仅改善 last-token 输入布局
lm_head_impl='contiguous'

# 可选近似实验；不要与上一个参数同时指定
lm_head_impl='int8', lm_head_group=64
```

`quantize_groupwise(model.lm_head.weight, 64)` 可离线创建快照，通过
`quantized_lm_head=` 复用。原模型参数不变，context 退出恢复方法，Graph 持有快照引用。
此适配器为 inference-only、单线程实验，不支持训练、并发、多请求 padding 或 beam search。

```bash
cd /data1/tzh/proj/qwen_operator_lab
/data1/tzh/proj/systems/.venv/bin/python -m pytest -q
/data1/tzh/proj/systems/.venv/bin/python -m benchmarks.benchmark_static_decode_e2e \
  --suite lm_head --seed 20261008 --repeats 12 --steps 16 --profile \
  --cases 1:128,2:128,1:512,1:2048,4:128 --tag _my_lm_head
/data1/tzh/proj/systems/.venv/bin/python -m benchmarks.benchmark_compile_pipeline \
  --cases 1:128,1:512 --repeats 12 --hybrid --tag _my_compile
```

现有 JSON 不覆盖；复测请使用新的 `--tag`。各 JSON 保存当时源码指纹，不应把后续
代码改动后的文件散列冒充为历史结果的版本。单算子两项扫描脚本为一次性实验入口，
若已有结果会拒绝覆盖。

## 最终回归

完整测试 **165 passed，2 warnings，349.58 s**。警告分别是旧 attention backward 测试的
cuBLAS context 初始化，以及 TF32 可用但未启用；没有改变精度配置来消除警告。
新增测试覆盖 FP16/BF16 LM head、batch 权重复用、大词表 grid、tied 参数恢复、Graph 快照
生命周期、原生连续化、custom-op schema/fake、全图编译和共享 KV 的增量 decode。
最终两份端到端 JSON 的全部已记录源码 SHA256 已与当前文件核对一致。
测试摘要：`artifacts/lm_head_compile_validation.json`。
