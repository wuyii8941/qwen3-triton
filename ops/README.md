# 当前训练算子库

`reference/` 是数学参照，`triton/` 是可训练实现；保留 forward、手写 backward 和 autograd 接口，不代表全部被真实 8B 默认训练采用。

| 文件 | 当前职责 |
|---|---|
| `rms_norm.py` | RMSNorm forward/backward |
| `rope.py` | Qwen split-half RoPE forward/backward |
| `silu_and_mul.py` | SwiGLU 激活部分，不包含 GEMM |
| `fused_qk_norm_rope.py` | Q/K RMSNorm + RoPE 融合 |
| `attention.py` | 整段 causal GQA forward 与 autograd；prefill 是同计算的形状约束入口 |
| `attention_backward.py` | 单一当前路径：寄存器累积 dQ + fused dK/dV |
| `gate_up_swiglu.py` | 可训练 GEMM + SwiGLU，Tiny 的显式可选后端，不是默认性能赢家 |

推理专用 decode、静态 cache、INT8、稀疏算法、参数调优版本和旧 attention backward 基线都在 `experiments/legacy/ops/`。这些是保留的实验，不会被主运行时偷偷导入。

从 [目录导航](../REPOSITORY_LAYOUT.md) 查看训练路径与测试命令。注意 PyTorch 数学 reference 与 HF 的低精度中间 cast 语义不是同一个验收口径。
