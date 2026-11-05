# Qwen/Triton 推理优化实践

更新：真实 checkpoint 接入、prefill/decode 结果及生成 smoke test 已补齐，见 [HF_INFERENCE.md](HF_INFERENCE.md)。下文是独立算子和随机 block 实验的原始范围，数值不能与真实模型 wall-time 混用。

## 当前实验范围

RMSNorm、split-half RoPE、SwiGLU 的 FP16/BF16 推理验证，以及随机初始化的 Qwen 结构 decoder block 集成。Block 使用 hidden=2048、intermediate=6144、Hq=16、Hkv=8、D=128；GEMM 保留 PyTorch，attention 显式使用 PyTorch Flash SDPA。

这不是预训练 Qwen 完整模型或生成服务的吞吐测量。S=1 的独立算子测试不是带 KV cache 的模型 decode。不能据此声称完整 Qwen 加速同样倍数。

## 可复现命令

从项目根目录执行，使用现有独立虚拟环境：

```bash
/data1/tzh/proj/systems/.venv/bin/python benchmarks/benchmark_inference_evidence.py
/data1/tzh/proj/systems/.venv/bin/python benchmarks/benchmark_qk_layout.py
```

产物：`artifacts/inference_evidence.json`、`artifacts/qk_layout_ablation.json`。

前者覆盖 batch/seq=(1,1)、(1,128)、(2,512)，两种 dtype、三类算子、eager/compile/Triton 三种后端。每例先验证输出，再计时。compile 使用 fullgraph 并逐 case reset，避免编译缓存上限导致静默退回 eager。首次调用时间单列，包含可能的编译/autotune，不与 steady-state 混算。

计时同时提供 CUDA Graph 重放 GPU 时间和不捕获的主机可见批量调用时间。前者用于隔离 Python 调度影响，是固定地址重复 workload，可能受缓存复用影响；后者包含 Python/调度影响。记录七次 GPU 样本的中位数及范围。显存报告单次调用相对 baseline 的 allocated peak delta，包含返回输出，不代表纯 workspace。

数值容差是 rtol=0.02、atol=0.02，同时记录 max absolute error；低精度 eager 中间舍入与融合 FP32 中间计算存在差异。不能将通过这一容差表述为逐位一致。布局消融使用完全相同的算术，其输出要求逐位一致。

## 本轮优化：消除 QK-Norm+RoPE 布局复制

原路径：BSHD → transpose+contiguous → BHSD kernel → transpose+contiguous → BSHD。

新路径：BSHD kernel → BSHD。程序直接处理 projection 输出中连续的 head_dim，沿相邻 head 分配 program。保留原 kernel 的 BHSD 模式作消融基线，backward 暂不改变。

对 Q 和 K 分别节省两次布局复制；每次复制需要读写整个 tensor。对 B=2、S=512、Hq=16、Hkv=8、D=128、两字节 dtype，理论减少约 24 MiB 的复制读写流量，不含缓存效果。算术量未变，主要收益来自省去复制 kernel 和中间张量。

## 项目结论与简历口径

2026-09-28 实测：RTX A6000，PyTorch 2.5.1+cu121，Triton 3.1.0。62 条比较记录均先通过数值检查，再完成计时；相关既有 kernel/model 回归测试 14 passed。

BF16、B=2、S=512，GPU CUDA Graph 中位数（微秒）：

| 算子 | eager | compile | Triton |
|---|---:|---:|---:|
| RMSNorm，hidden=2048 | 135.48 | 14.44 | 14.34 |
| SwiGLU，intermediate=6144 | 96.97 | 57.65 | 57.65 |
| RoPE，Hq=16/Hkv=8/D=128 | 105.37 | 22.12 | 28.26 |

这组结果支持相对 eager 的融合收益，不支持“普遍优于 Inductor”：RoPE 还慢于 compile。

随机 Qwen 结构 block（使用相同权重、Flash SDPA）的 GPU CUDA Graph 中位数：

| dtype | B/S | eager ms | 三类算子替换后 ms | 加速比 |
|---|---|---:|---:|---:|
| FP16 | 1/128 | 0.334 | 0.231 | 1.45× |
| FP16 | 2/512 | 1.700 | 1.206 | 1.41× |
| BF16 | 1/128 | 0.381 | 0.274 | 1.39× |
| BF16 | 2/512 | 1.747 | 1.265 | 1.38× |

这是所有算子替换的组合收益，不是仅由本轮布局改动贡献。较大 block 的样本存在波动，详见 JSON 的 min/max。

布局消融仅测 Q 分支、H=16、D=128：BF16 B=2/S=512，从 55.40 μs 降至 29.08 μs（1.90×），新增分配峰值从 12 MiB 降为 4 MiB；旧、新结果逐位一致。FP16 对应 55.60→28.77 μs。

可使用的阶段性简历表述：

> 基于 Triton 实现 RMSNorm、RoPE、SwiGLU 推理算子，完成 FP16/BF16、多 batch/sequence 配置下的数值验证；建立 eager/Inductor/Triton 三后端基准，区分 GPU 执行和主机调度开销。消除 QK-Norm+RoPE 的布局复制，在指定 BF16 workload 下使融合 Q 分支延迟由 55.4 μs 降至 29.1 μs；集成到 Qwen 结构 decoder block，固定形状 CUDA Graph 测试中相对 eager 获得约 1.38–1.45× 加速。

可以描述实现三类 kernel、建立三后端数值/性能对照、完成 Qwen 结构 block 集成和布局复制优化。具体倍数必须带 shape、dtype、计时口径，并链接 JSON 数据。当前 block 未与 compiled block 比较，不能称为超过 Inductor 的端到端加速。

历史 SwiGLU benchmark 的 compile cache limit warning 使部分结果无效；历史 MLP 多 variant 常驻及梯度残留的总 peak 不可直接比较。历史 attention prefill 中两侧是否包含 cache 分配也不一致。上述结果不作为此报告的证据。

下一验收阶段：真实 checkpoint/固定 Transformers 版本 parity、真实模型 inference 接入、多随机种子和边界输入测试、profiler 的 kernel/copy 证据、分离 prefill/decode 的公平端到端比较。推理阶段优先复用成熟 GEMM/attention，只对实测有效的算子启用替换。
