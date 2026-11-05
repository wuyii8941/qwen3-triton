# 真正产生端到端收益的优化：QK 融合 + compact GQA

## 结论

这次不是把算子加速比当作模型加速比，而是在本地完整、预训练的 Qwen3-1.7B
上做五路径消融。第一轮 **FP16** 相对此前已有最佳 Triton 路径：
prefill 延迟下降 **16.4–31.0%**，decode 下降 **25.2–32.4%**。
范围是 A6000、固定输入、B=1/2、prompt=128/512；不能泛化为任意配置或生产服务吞吐。

独立进程复测中，decode 仍下降约 **26.3–31.5%**；prompt=128 的 prefill 下降
约 **26.2–30.6%**。prompt=512 的 prefill 收益降到 **4.5%**，波动明显，
所以当前更稳健的结论是 **decode 与短 prefill 有约四分之一到三分之一的收益**，
不能只挑第一轮数据宣称长 prefill 也稳定改善 16%。

重要数值边界：FP16 本轮通过原有 2% 相对 L2 门槛；BF16 的一组输入连旧基线也
达到 2.14%，因此该组只保留诊断结果，不计为通过验收的加速方案。没有放宽门槛或换输入避开它。

## 1. 为什么这次更有用

之前优化 RMSNorm 的行数，每层仅节约少量 GPU 时间，完整模型的 Python/launch
开销仍在。本次改变整条 attention 调用路径：

1. **Q/K RMSNorm + RoPE 的单次 Triton launch**。
   旧 HF 适配路径每层有 Q norm、K norm、Q RoPE、K RoPE 四次 launch。
   新版在一个 kernel 内处理两种 head 数及各自的 norm 权重，并保留 HF 的低精度
   cast 边界，直接输出 BHSD。28 层少 84 次 launch，也少了 norm 中间张量。
2. **不再把 K/V 扩展到 query heads**。
   本机 Transformers 4.51.0 的 `integrations/sdpa_attention.py` 先执行
   `expand(...).reshape(...)` 复制 K/V，然后 SDPA。新版直接向 PyTorch 2.5.1
   的 SDPA 传 `enable_gqa=True`；实际 benchmark 强制 Flash 后端。
   每层少两份 K/V 扩展复制，28 层少 56 次 clone。

这不是自写 Triton attention 胜过 FlashAttention。Flash attention 本身保留；
Triton 优化发生在它前面的 QK norm/位置编码处理，GQA 改动属于调用链优化。

## 2. 实现位置和使用接口

- `ops/triton/hf_qk_norm_rope.py`：`hf_qk_norm_rope` 公共校验入口及 `_qk_norm_rope` kernel。
- `model/hf_attention.py`：保留 Linear、HF cache update、mask 与 output projection 的适配路径。
- `model/hf_inference.py`：新增 `fuse_qk`、`compact_gqa` 独立开关；退出 context 自动恢复。
- `tests/test_hf_attention_fusion.py`：cast、GQA、非整齐维度、cache、mask、attention weights fallback、恢复测试。

```python
# model 已经是 CUDA FP16、eval 的本地 Qwen3，attn_implementation='sdpa'
with torch.inference_mode(), qwen3_triton_inference(
    model,
    decode_mlp=True,  # 之前已有的小 batch MLP 优化，也包含在本次比较基线中
    fuse_qk=True,
    compact_gqa=True,
):
    prefill = model(input_ids, use_cache=True, logits_to_keep=1)
    decode = model(next_token_ids, past_key_values=prefill.past_key_values,
                   use_cache=True, logits_to_keep=1)
```

需要与实验相同的 attention 后端时，外层加
`sdpa_kernel(SDPBackend.FLASH_ATTENTION)`。非空 padding mask 不保证该版本 Flash 可运行；
单元测试允许 SDPA 正常选择支持 mask 的后端，但性能报告只有无 padding 的 forced Flash。
`output_attentions=True` 回退原 HF eager attention 语义。滑窗配置拒绝启用此优化。
仅限 pinned Transformers 4.51.0、单线程 inference；不支持训练、量化或 offload。
未把任何新开关设为默认，也没有改写模型权重。

## 3. FP16 完整模型延迟

单位 ms，同步 wall-time 中位数，包含 Python 调度与 cache update，不使用 CUDA Graph。
五条路径在每轮随机排列，4 轮预热后 24 轮有效采样。构建/预热 cache 不计入 decode。

| B,prompt | phase | 已有最佳 Triton | 新组合 | 延迟下降 |
|---|---|---:|---:|---:|
| 1,128 | prefill | 22.460 | 16.642 | 25.9% |
| 1,128 | decode | 22.633 | 16.934 | 25.2% |
| 2,128 | prefill | 29.547 | 20.395 | 31.0% |
| 2,128 | decode | 29.302 | 19.817 | 32.4% |
| 1,512 | prefill | 27.769 | 23.218 | 16.4% |
| 1,512 | decode | 26.256 | 19.282 | 26.6% |

示例消融（FP16 B=1,prompt=128）：

| 路径 | prefill ms | decode ms |
|---|---:|---:|
| 原始 HF + Flash SDPA | 30.005 | 30.062 |
| 已有 Triton + 先前的 decode MLP 优化 | 22.460 | 22.633 |
| 仅加 QK 融合 | 18.909 | 19.119 |
| 仅加 compact GQA | 19.761 | 19.780 |
| 两项一起 | 16.642 | 16.934 |

新组合的相对收益不是通过选一个未优化的基线制造的。相对 HF 的总收益也不能再
与旧简历数字相加。JSON 中保留原始样本、每轮顺序和 paired speedup 的 bootstrap
95% 区间；该区间只描述本次测量，不证明跨日期、跨机器都稳定。

独立进程复测（相同输入，保留原结果，不覆盖）：

| B,prompt | phase | 已有最佳 Triton ms | 新组合 ms | 延迟下降 |
|---|---|---:|---:|---:|
| 1,128 | prefill | 21.924 | 16.170 | 26.2% |
| 1,128 | decode | 21.903 | 15.890 | 27.5% |
| 2,128 | prefill | 24.209 | 16.807 | 30.6% |
| 2,128 | decode | 24.494 | 16.771 | 31.5% |
| 1,512 | prefill | 24.461 | 23.355 | 4.5% |
| 1,512 | decode | 21.757 | 16.029 | 26.3% |

绝对时间随主机调度/硬件状态变化，不跨轮直接相减；主要使用各轮内部配对比较。

## 4. Profiler：减少的是什么

对 FP16 B=1,S=128 分离采集一次 untimed prefill：

| 指标 | 已有最佳 | 新组合 |
|---|---:|---:|
| CUDA kernel 事件数 | 629 | 489 |
| `_norm` | 113 | 57 |
| `_rotary` | 56 | 0 |
| `_qk_norm_rope` | 0 | 28 |
| `aten::clone` | 84 | 28 |
| 独立 CUDA kernel 执行时间之和 | 7.792 ms | 7.492 ms |

kernel 数量下降 140，符合 84 次融合节约 + 56 次复制消除。
实际 GPU kernel 总时间只下降约 4%，小于 wall-time 改善；结果支持收益主要来自
调用/调度与中间操作减少，而非矩阵计算吞吐突然提升。没有把嵌套 profiler 的
`aten::linear`、`aten::mm` 等重复求和；这里只有 CUDA 设备事件之和。

显存：prefill 新增峰值分配未下降（仍由其他激活决定）；decode B=1,prompt=128
由约 1.15 MiB 降到 0.40 MiB，B=1,prompt=512 由约 7.94 MiB 降到 4.90 MiB。
这是 allocated peak delta，不是模型总显存、KV cache 总量或 reserved memory。

## 5. 数值验收与反例

- 固定 seed=20260929，随机输入；检查每层 hidden states、prefill 最后位置 logits、
  四步固定 token decode logits、full vs incremental。
- FP16 新组合相对 HF 的 logits 相对 L2 最大约 **0.173%**；本次全部 argmax 检查一致。
- FP16 新组合相对 `current_best` 的被测 logits **误差为 0**；BF16 本次对应对比也为 0。
  这是有限测试数据的观测，不是对所有输入的 bitwise 保证。
- 单个自然语言 greedy smoke prompt，HF/旧路径/新组合的 token 序列一致；不代表任务质量评测。
- BF16 B=1,prompt=128，旧路径与新组合相对 HF 均出现 **2.1387%** logits 相对 L2，
  超过预设 2% 门槛。`accepted=false` 写在 JSON 中，不计作通过数值验收的 BF16 方案。
  另外两组 BF16 通过，但仍不宣称整体 BF16 已完成验收。

下一步若要发布 BF16 默认配置，应先定位旧路径的误差来源并增加自然文本验证；
本轮已验证的端到端收益结论以 FP16 为主。

## 6. 复现和审计

```bash
cd /data1/tzh/proj/qwen_operator_lab
export PYTHONPATH=.
/data1/tzh/proj/systems/.venv/bin/python -m pytest -q
/data1/tzh/proj/systems/.venv/bin/python benchmarks/benchmark_hf_attention_e2e.py --dtype float16 --profile
/data1/tzh/proj/systems/.venv/bin/python benchmarks/benchmark_hf_attention_e2e.py --dtype bfloat16 --profile
/data1/tzh/proj/systems/.venv/bin/python benchmarks/benchmark_hf_attention_e2e.py --dtype float16 --tag _recheck
```

- `artifacts/hf_attention_e2e_float16.json`：主结果、逐层误差、profiler、自然语言 smoke。
- `artifacts/hf_attention_e2e_bfloat16.json`：保留失败门槛与通过门槛的所有情况。
- `artifacts/hf_attention_e2e_float16_recheck.json`：独立进程复测，不覆盖第一轮数据。

manifest 保存 GPU/driver、依赖版本、源文件与 checkpoint SHA256。比较模型均同权重、
同输入、相同 `logits_to_keep=1`、相同 cache 长度；没有 tokenizer 或采样耗时混入。
这是完整模型 forward/prefill/单步 decode，不包含网络服务和请求队列，未与
compiled HF、vLLM、CUDA Graph 服务引擎比较。

最终全项目回归：**120 passed**（另有一次 cuBLAS CUDA context 初始化 warning）。
