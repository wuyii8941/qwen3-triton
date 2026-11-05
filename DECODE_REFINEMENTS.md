# CUDA Graph 之后：KV 写入融合、残差融合、W8A16 实验

## 动机与基线

本次基线 `graph_ref` 是上一阶段 Graph + packed QKV + K 直接写 cache + prefill MLP，
已经保留原生 Flash，不能再拿早期未优化 Python 路径当作这批小优化的唯一对照。
实际 profile 中，gate/up 约占 28 层合计 2.05 ms，另有 down projection 和 LM head；
因此增加两类实验：减少剩余小 kernel，以及减少 decode 投影读取的权重字节数。

环境仍为 RTX A6000 / FP16 / torch 2.5.1 / Triton 3.1.0 / Transformers 4.51.0。
没有覆盖 checkpoint、没有改宿主 Python、没有把量化设成默认。

## 不改变权重的优化

- `direct_v_cache=True`：在融合 QK kernel 的 K 分支顺便将 V 写入 BSHD cache。
  V 不经过 norm/rotation。输入支持 packed projection stride，不再单独 `index_copy_`。
- `fuse_residual=True`：复用已有 Add+RMSNorm kernel，重新验证其在 Graph 下的收益。
- 也比较了退回原生 cuBLAS gate/up + Triton activation；本轮没有比原小 batch GEMV 更快。

新增第一轮（seed=20261002），pipeline 为 prefill + 16 步 decode，ms：

| B / prompt | graph_ref | + V 写入融合 | + residual/norm |
|---|---:|---:|---:|
| 1 / 128 | 111.491 | 110.682 | 111.419 |
| 2 / 128 | 120.000 | 118.474 | 119.658 |
| 1 / 512 | 117.434 | 115.531 | 117.138 |
| 1 / 2048 | 172.535 | 171.309 | 171.689 |

V 融合的 paired speedup 约 1.007–1.014×，四组 CI 均高于 1。
残差融合的单项 wall-time 收益较小，多数组 CI 跨 1；不能称为强结论。
Profiler 单步：381 kernels → 353（单独 V 或 residual 融合）。两项作用不同，可以组合，
但组合收益以独立实测为准，不把百分比相加。

## W8A16：正确不代表快，更不代表无损

仅对 decode 的 MLP gate/up 使用对称分组 INT8 权重，scale 为 FP32，激活仍是 FP16。
prefill、down projection、QKV、LM head 仍使用原权重。
公式：每个输出行沿 K 维分组，`scale=max(abs(w))/127`；量化到 [-127,127]，
在 kernel 内反量化为原激活 dtype 后做 FP32 累加，保留 HF projection/SiLU 舍入。

两种验收不能混淆：

1. Kernel 与“显式反量化权重”oracle 对齐，证明索引与计算正确。
2. 完整模型与原 checkpoint logits 对齐，评估量化引入的误差。

第一版逐元素读取重复 scale，完整 decode **反而显著变慢**：B1/S128 的 16 步
93.785 → 153.997 ms；B2 更差。这不是可以隐藏的结果。

之后新增显式 group-scale 加载/广播与二维 reduction，另试 Tensor Core 版本。
参数扫描先过 kernel oracle 再计时。某些 TC 配置所需 shared memory 大于本机
101376 bytes，记录为 unsupported，没有用别的实现假装它运行成功。
Grouped GEMV 在这次扫描中更好：B1 单算子约 43.32 μs，原 FP16 对照 74.55 μs。
这只是 G64 的算子调参证据，不是 G64 通过了模型数值验收。

数值反例保留：

- G64：seed=20261002、B1/S512 最大 logits L2 **2.368%**，失败。
- G32：seed=20261003、B1/S128 最大 logits L2 **2.024%**，失败。
- 没有把 2% 门槛改成 2.5%，也没有删除这些输入。
- G16 在上述两轮四种配置通过，但仍属于近似实验，不宣称任务质量无损。

新版真实模型（seed=20261003），仅展示 G16；单位 ms：

| B / prompt | graph_ref pipeline | G16 pipeline | 观察 |
|---|---:|---:|---|
| 1 / 128 | 110.538 | 99.755 | 有收益 |
| 2 / 128 | 119.177 | 126.065 | 退化，不适合该 batch |
| 1 / 512 | 116.212 | 105.869 | 有收益 |
| 1 / 2048 | 172.198 | 163.550 | 有收益，prefill 占比变大 |

因此新增 `decode_int8_max_batch=1`：只在 B1 使用量化路径，B≥2 回到原 FP16 decode MLP。
这是明确的 workload 分流，不是 benchmark 中遇到错误后偷偷换后端。

## 最终分流验证：新的 seed=20261004

每种配置 16 个随机交错 round，3 个 warmup round。三个路径使用相同输入与 Flash 后端：

- `graph_ref`：上一阶段的最佳 Graph 路径。
- `graph_exact`：再加 K/V 写入融合及 residual/norm 融合，不量化。
- `graph_selective`：在 `graph_exact` 上仅 B1 启用 G16 grouped W8A16；其余 batch 保留 FP16。

完整 prefill + 16 步 decode，中位数 ms：

| B / prompt | graph_ref | 不量化新组合 | 可选分流组合 | 分流组合 / decode step |
|---|---:|---:|---:|---:|
| 1 / 128 | 110.783 | 109.088 | 98.376 | 5.098 ms |
| 2 / 128 | 118.628 | 116.296 | 116.417 | 6.038 ms |
| 1 / 512 | 116.173 | 114.256 | 103.755 | 5.239 ms |
| 1 / 2048 | 171.828 | 169.996 | 161.331 | 5.755 ms |
| 4 / 128 | 125.292 | 123.107 | 123.099 | 6.299 ms |

相对 `graph_ref`，无量化组合的 paired speedup 1.011–1.020×，五组 CI 均高于 1。
B1 分流组合的 paired speedup：1.126× / 1.122× / 1.066×；对应完整流程延迟约下降
11.2% / 10.7% / 6.1%（中位数之比口径）。B2/B4 的分流路径就是 FP16，细小时间差属于采样波动。

Profiler：`graph_ref` 每步 381 个 GPU kernel，两个融合组合均为 325，少 **56** 个。
量化没有再减少 kernel 数量，收益来自投影实现变化，不能说是又融合掉了几十个 launch。

这一轮所有数值门槛通过；无量化新组合的被测 logits 相对 `graph_ref` **逐位一致**。
分流组合相对原 HF 的最大 logits relative L2 约 **0.496%**，所测 argmax 与
32-token 自然文本 smoke 全部一致。G16 在本轮及前两轮已测试输入中的最大 L2 约
**1.375%**，不等价于“任意输入误差都小于这个值”或任务质量无损。
G64/G32 的失败仍留在报告中；最终分流没有使用它们。

若显存预算紧张或不接受近似权重，使用 `graph_exact`，不要启用 `decode_int8_group`。
若要把量化用于实际应用，应先补自然文本集 NLL/任务准确率，不只看这批随机 logits。

### 显存代价

这是保留原权重的实验快照，**不是减少总模型显存的部署方案**。
gate/up quantized snapshot 的额外常驻大小：G64 714 MiB，G32 756 MiB，G16 840 MiB。
此外 packed QKV 仍额外 448 MiB。减少的是被量化 decode 的权重读取量，不是当前原型的总显存。

## 接口和复现

无量化路径：

```python
with torch.inference_mode(), qwen3_triton_inference(
    model, decode_mlp=True, fuse_qk=True, compact_gqa=True,
    packed_qkv=True, prefill_mlp=True, static_decode=True,
    direct_k_cache=True, direct_v_cache=True, fuse_residual=True,
):
    # 沿用 StaticDecodeRunner.prefill / capture / decode。
    ...
```

可选近似路径额外传：

```python
decode_int8_group=16,
decode_int8_impl='grouped',
decode_int8_max_batch=1,
```

`quantize_decode_mlp(model, 16)` 可离线建立快照，通过 `quantized_mlp=` 复用。
更改原权重后必须重建。Graph 显式持有量化快照，防止 context 退出后地址失效。
训练/梯度开启时拒绝量化调用，context 退出恢复原模块方法，不修改参数。

```bash
cd /data1/tzh/proj/qwen_operator_lab
/data1/tzh/proj/systems/.venv/bin/python -m pytest -q
/data1/tzh/proj/systems/.venv/bin/python -m benchmarks.benchmark_static_decode_e2e \
  --suite dispatch --seed 20261004 --repeats 16 --steps 16 --profile \
  --cases 1:128,2:128,1:512,1:2048,4:128 --tag _new_dispatch
```

证据：`static_decode_e2e_refine.json`、`static_decode_e2e_refine_v2.json`、
`weight_only_tuning.json`、`static_decode_e2e_dispatch.json`（最终分流）。均位于 `artifacts/`，
失败门槛和 unsupported 配置也保留。源码 SHA256 对应该阶段版本；当前已新增
[LM head 与 compile 实验](LM_HEAD_COMPILE.md)，不要与历史源码指纹混用。
完整回归当前为 **150 passed**，另有一条既有 cuBLAS context 初始化 warning。
关于 kernel tests：包括 FP16/BF16、G16/32/64/128、零权重行、多输出行、原权重不变；
模型测试覆盖快照恢复、batch 回退、Graph 权重持有以及融合 K/V 的增量一致性。
