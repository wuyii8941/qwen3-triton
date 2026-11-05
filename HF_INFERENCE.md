# 真实 Qwen3 inference 接入

## 用法

本地模型 `/data1/tzh/models/Qwen/Qwen3-1.7B`，Transformers 严格校验 4.51.0。权重不改写，不下载。默认使用原 HF model、attention、cache API、Linear，仅替换 RMSNorm、RoPE、SwiGLU。下文的额外实验选项会改变部分路径，均默认关闭。

```python
import torch
from transformers import AutoModelForCausalLM
from model.hf_inference import qwen3_triton_inference

model = AutoModelForCausalLM.from_pretrained(
    '/data1/tzh/models/Qwen/Qwen3-1.7B',
    local_files_only=True, torch_dtype=torch.bfloat16,
    attn_implementation='sdpa',
).cuda().eval()

with torch.inference_mode(), qwen3_triton_inference(model):
    prefill = model(input_ids, use_cache=True, logits_to_keep=1)
    decode = model(next_token_ids, past_key_values=prefill.past_key_values,
                   use_cache=True, logits_to_keep=1)
```

此 context 会在正常退出或异常时恢复方法。RoPE 的函数替换在 context 内是模块全局的，限单线程实验，不适用于并发服务。禁止在训练/启用梯度时使用。支持 CUDA FP16/BF16，同设备同 dtype 模型；不支持量化、分片/offload、其他 Transformers 版本的未经验证接入。

## 为什么增加 HF 专用兼容 kernel

新增实验开关（均默认关闭）：`fuse_residual=True` 融合 attention residual add 与
post-attention RMSNorm；`decode_mlp=True` 在单 token、batch≤2、hidden=2048 时
尝试融合 gate/up GEMV + SwiGLU（此时不再保留这两次 Linear 调用）。其他 shape
保留原 MLP 路径。动机、消融和限制见 [论文驱动实验](RESEARCH_OPTIMIZATIONS.md)。

`multirow_norm=True` 为另一个独立实验开关：仅优化大量 `head_dim=128` 的 Q/K
norm 行，不改 hidden RMSNorm、attention 或 MLP。其算子和端到端测量须区分，
见 [kernel 内部调优报告](KERNEL_TUNING_RESULTS.md)。默认关闭。

`fuse_qk=True, compact_gqa=True` 为新的 attention 路径优化：单次 Triton launch
完成 Q/K norm + RoPE，再直接调用 native GQA SDPA，避免 HF 的 K/V 扩展复制。
FP16 有完整模型端到端收益；扩展输入中的 BF16 旧基线出现超过 2% 的误差，
因此不要把 BF16 当作已全面验收。接口、消融和反例见
[端到端优化报告](E2E_ATTENTION_OPTIMIZATION.md)。仍默认关闭。

新增 `packed_qkv`、`prefill_mlp`、`static_decode`、`direct_k_cache` 独立实验选项。
其中 static cache / CUDA Graph 要通过 `model/static_decode.py` 的 `StaticDecodeRunner`
调用，不应直接交给 HF `generate()`。锁定 PyTorch 2.5.1 的私有 native Flash varlen API，
限定无 padding、等长 batch、连续 decode。接口、端到端证据与显存成本见
[Static decode 实验报告](STATIC_DECODE_OPTIMIZATION.md)。

进一步可选：`direct_v_cache=True` 将 V cache 复制并入融合 QK kernel；
配合 `fuse_residual=True`，在已有 Graph 上再减少 56 个 GPU kernel。
`decode_int8_group=16, decode_int8_impl='grouped', decode_int8_max_batch=1`
是单独的**近似、仅 B1**实验路径，会增加量化权重快照显存，不默认开启。
G64/G32 的已知误差反例、分流规则及精度/性能证据见
[Decode refinement 报告](DECODE_REFINEMENTS.md)。完整工作索引见
[优化总表](OPTIMIZATION_SUMMARY.md)。

`model/hf_inference.py` 显式保留 HF 低精度中间结果舍入：RMSNorm 在乘权重前 cast，RoPE 两项乘积先 cast 再相加，SwiGLU 先 cast SiLU 再乘 up。相比早期纯 FP32 中间计算，更接近真实预训练模型的数值语义。浮点 reduction/exp 仍可能有差异，因此不声称完整模型逐位一致。

RoPE kernel 直接接受 HF 的非连续 BHSD view，通过 stride 寻址并将多个元素/行分给同一 program，避免为输入 transpose 调用 contiguous。输入 head_dim 方向连续；Q/K head 数不必相同。GEMM、Flash SDPA 和 HF KV cache 保持现有实现。

## 验收与复现

```bash
/data1/tzh/proj/systems/.venv/bin/python -m pytest -q tests/test_hf_inference.py
/data1/tzh/proj/systems/.venv/bin/python benchmarks/benchmark_hf_inference.py --dtype bfloat16 --profile
/data1/tzh/proj/systems/.venv/bin/python benchmarks/benchmark_hf_inference.py --dtype float16 --profile
```

报告：`artifacts/hf_inference_bfloat16.json`、`artifacts/hf_inference_float16.json`。

- 原始 checkpoint 权重 SHA256、config SHA256、adapter SHA256、GPU/driver/库版本记录于 manifest。
- 固定随机 seed 和输入 token，B=1/2，prompt=128，额外 8 步增量 decode；当前不是自然语言任务质量评估。
- 比较 prompt 的每层 hidden states、prefill 最后位置 logits、8 步 decode logits；检查完整 causal forward 最后位置与增量 decode 一致性。
- 预设数值门槛为相对 L2 < 2%，逐项记录最大绝对误差；argmax 一致率另报，不替代数值检查。此门槛不是逐元素 assert_close，也不是质量无损承诺。
- 测量 full model forward，`use_cache=True`、`logits_to_keep=1`，强制 Flash SDPA；预热后测 10 次，报告中位数与范围。
- Prefill 两侧都包含 HF cache 创建；decode 两侧都从新建的等长 prompt cache 开始，只计时一次 token forward（含 cache update），不把 prepare/prefill 时间算入 decode。
- 使用同步后的 wall time，包含 Python/kernel launch。没有拿 CUDA Graph 时间与未捕获时间混比。显存用该次调用的 allocated peak 增量，非 allocator reserved。
- Profiler 独立采集一次 prefill，不混入计时；事件名称/count/device time 存在 `profile_events`。

范围之外：自由生成文本/任务准确率、left-padding/多请求变长、paged cache、完整训练性能、并发服务吞吐，以及与 torch.compile 整模型/vLLM 的比较。项目中的自写 attention/GEMM backward 是实验路径，不用于这个推理结果。

## 2026-09-28 实测记录

RTX A6000；完整 28 层本地 Qwen3-1.7B，prompt=128。同步 wall-time 中位数（ms），每组预热后 10 次；包含 Python 调度，未使用 CUDA Graph。

| dtype | batch | HF prefill | Triton prefill | HF decode | Triton decode |
|---|---:|---:|---:|---:|---:|
| FP16 | 1 | 31.228 | 27.869 | 30.376 | 28.279 |
| FP16 | 2 | 40.305 | 29.929 | 38.999 | 31.378 |
| BF16 | 1 | 28.594 | 26.505 | 28.901 | 27.197 |
| BF16 | 2 | 39.295 | 29.763 | 41.549 | 31.885 |

这轮测量的 prefill 延迟下降约 7–26%，decode 下降约 6–23%。单 GPU、少量固定配置的原型结果，存在主机调度/时钟波动，不应扩大为任意 batch/seq 的保证，也没有与 compiled HF/vLLM 比较。

数值：两组 batch 的 FP16 logits 相对 L2 最大约 0.10%，BF16 最大约 1.06%；prefill+8步固定输入 decode 的 argmax 一致率均为 100%。自然语言 smoke demo 的 28 个输出 token（包含 EOS）也完全一致，结果见 `artifacts/hf_generation_demo.json`。这不是广泛任务质量评测。

完整项目回归：62 passed。新增测试覆盖低精度 HF 算子语义、非连续 RoPE 输入和异常时恢复原模型方法。

Profiler 在优化 prefill 中记录 `_norm` 113 次、`_rotary` 56 次、`_activation` 28 次，与 28 层结构及最终 norm 对应。事件与源码/权重指纹保存在 JSON 中，支持核对实际执行的 kernel。

运行自然语言 demo：

```bash
/data1/tzh/proj/systems/.venv/bin/python demo_hf_inference.py
```

## 面试展示建议

先展示一个 kernel 及其输入布局，再展示 eager/compile/Triton 基准，解释为什么省去中间张量/复制能降低流量，最后展示真实 checkpoint 的误差和 prefill/decode 延迟。明确 GPU 时间、wall time、单算子和端到端的边界。已知负结果（compile 接近或优于手写、GEMM fusion 训练变慢）同样保留。
