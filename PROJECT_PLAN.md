# Qwen Operator Lab 项目计划

## 0. 项目目标

本项目不是重新实现整个 CS336 Transformer，而是：

1. 用 Hugging Face Qwen3 作为真实模型 reference。
2. 用一个可训练的 Tiny-Qwen 作为快速开发和调试平台。
3. 只补 Qwen 相比标准 Transformer 的结构差异。
4. 对关键算子建立 PyTorch reference、Triton 实现和 benchmark。
5. 最终把验证过的算子接回真实 Qwen，比较端到端效果。

核心验证链路：

```text
Hugging Face Qwen3 reference
          ↓
Tiny-Qwen PyTorch baseline
          ↓
单算子 reference
          ↓
Triton custom kernel
          ↓
Tiny-Qwen integration
          ↓
真实 Qwen integration
```

---

## 1. 当前已经完成的内容

### 1.1 Hugging Face Qwen baseline

文件：`profile_qwen.py`

当前默认加载：

```text
/data1/tzh/models/Qwen/Qwen3-1.7B
```

调用接口：

```python
AutoTokenizer.from_pretrained(model_path, local_files_only=True)
AutoModelForCausalLM.from_pretrained(...)
```

当前功能：

- 加载本地 Qwen3-1.7B 权重
- 固定文本输入
- 自动选择 CUDA / CPU
- 对 `model.model.layers` 注册 forward hook
- 记录每个 Transformer block 的 input/output shape、dtype、device
- CUDA warmup
- CUDA synchronize 后测量 forward latency
- 记录 peak allocated / reserved CUDA memory
- 保存 JSON baseline 报告

运行：

```bash
uv run --no-sync python profile_qwen.py
```

输出：

```text
artifacts/qwen_baseline.json
```

当前已验证：

```text
28 个 Transformer blocks
输入 shape: [1, 15]
dtype: bfloat16
forward median: 约 30 ms
peak allocated memory: 约 3.2 GiB
```

### 1.2 Tiny-Qwen 配置

文件：`model/config.py`

默认配置：

```text
vocab_size = 4096
hidden_size = 256
intermediate_size = 768
num_hidden_layers = 2
num_attention_heads = 4
num_key_value_heads = 2
max_position_embeddings = 1024
```

这个配置用于快速运行训练和单元测试，不用于加载 1.7B 权重。

### 1.3 Tiny-Qwen PyTorch baseline

文件：`model/qwen_tiny.py`

当前已实现：

- token embedding
- Qwen-style RMSNorm
- Q/K/V projection
- GQA 的 query heads / KV heads 划分
- RoPE 的基础 contiguous-position 版本
- PyTorch `scaled_dot_product_attention`
- causal attention
- SwiGLU MLP
- pre-norm residual block
- final RMSNorm
- tied embedding / LM head
- causal language modeling loss

主要接口：

```python
model = TinyQwenForCausalLM(config)
result = model(input_ids, labels=input_ids)

logits = result["logits"]
loss = result["loss"]
loss.backward()
```

### 1.4 最小训练入口

文件：`training/train_tiny.py`

当前训练链路：

```text
固定 toy token pattern
    ↓
TinyQwenForCausalLM
    ↓
cross entropy
    ↓
backward
    ↓
AdamW
    ↓
checkpoint
```

运行：

```bash
uv run --no-sync python -m training.train_tiny --steps 20
```

当前已验证 loss 可以下降，并且能够保存：

```text
artifacts/tiny_qwen_checkpoint.pt
artifacts/tiny_qwen_checkpoint.json
```

### 1.5 当前单元测试

文件：`tests/test_tiny_qwen.py`

当前测试：

1. RMSNorm 数学定义
2. Rotary embedding shape 和有限值
3. GQA attention shape 和 backward gradient
4. 完整模型 forward / loss / backward
5. optimizer step 是否更新参数

运行：

```bash
uv run --no-sync pytest -q
```

当前结果：

```text
5 passed
```

---

## 2. 当前明确不做的事情

以下内容已经在 CS336 中学习过，不作为本项目第一阶段的重复实现目标：

- 标准 Linear
- 标准 embedding
- 基础 cross entropy
- AdamW
- 普通 residual add
- 普通 PyTorch training loop
- CS336 已经完成的标准 Transformer 组件

我们会直接使用 PyTorch 实现作为 reference，重点放在 Qwen 特有结构和性能优化。

---

## 3. 下一步需要补齐的 Qwen 结构 TODO

### 3.1 RoPE 接口完整化

当前接口：

```python
RotaryEmbedding.forward(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]
```

下一步接口目标：

```python
RotaryEmbedding.forward(
    position_ids: Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]
```

需要完成：

- 支持任意 `position_ids`
- 支持不同 batch 的 position
- 对照 Qwen3 的 rope theta
- 核对 Qwen3 的 long-context scaling
- 加入 position offset，用于 KV cache

验收标准：

- 与 Hugging Face Qwen3 同一层的 cos/sin 对齐
- 不同 sequence length 下 shape 正确
- forward 和 backward 数值稳定

### 3.2 Q/K normalization 和 projection 配置

当前实现默认：

- q/k/v projection 无 bias
- 没有额外 Q/K normalization

下一步需要从本地 Qwen3 配置和源码确认：

- q_proj/k_proj/v_proj/o_proj 是否有 bias
- 是否包含 q_norm/k_norm
- norm 放在 RoPE 前还是后
- head_dim 是否固定为 128

验收标准：

给定相同权重和输入时，单层 Q/K/V 中间结果与 HF reference 对齐。

### 3.3 Attention mask 和 cache 接口

当前接口：

```python
attention(hidden_states) -> hidden_states
```

目标接口：

```python
attention(
    hidden_states: Tensor,
    attention_mask: Tensor | None = None,
    position_ids: Tensor | None = None,
    past_key_value: tuple[Tensor, Tensor] | None = None,
    use_cache: bool = False,
) -> tuple[Tensor, tuple[Tensor, Tensor] | None]
```

需要完成：

- 显式 attention mask
- prefill 路径
- decode 路径
- past key/value 拼接
- cache position
- batch 内不同长度请求

### 3.4 权重映射

当前 Tiny-Qwen 不能直接加载 Qwen3-1.7B 权重。

下一步增加：

```python
load_qwen_state_dict(
    model: TinyQwenForCausalLM,
    state_dict: dict[str, Tensor],
) -> None
```

但 Tiny-Qwen 的 hidden size、layer 数和 vocab 通常不同，因此第一步不是直接加载完整 1.7B，而是：

1. 建立一个与 Tiny-Qwen 配置完全相同的 HF-compatible 小模型；或者
2. 只做单层、单算子权重映射测试。

---

## 4. 单算子开发规范

每个新算子都必须有三层实现：

```text
reference implementation
    ↓
custom PyTorch implementation
    ↓
Triton implementation
```

每个算子至少需要四类测试：

1. forward output
2. input gradient
3. parameter gradient
4. 多种 shape / dtype

通用比较方式：

```python
torch.testing.assert_close(
    custom_output,
    reference_output,
    rtol=...,
    atol=...,
)
```

不要直接从完整模型 loss 判断算子是否正确。先在独立 tensor 上定位错误，再接回 block。

---

## 5. Triton 具体实现目标

Triton 不需要重写所有 PyTorch 算子，只重写有明确性能收益的热点或融合路径。

### 5.1 RMSNorm Triton

目标接口：

```python
rms_norm_triton(
    x: Tensor,
    weight: Tensor,
    eps: float,
) -> Tensor
```

计算目标：

```text
variance = mean(x * x, dim=-1)
y = x * rsqrt(variance + eps) * weight
```

测试：

- `[batch, seq, hidden]` 多种 shape
- float32 / bfloat16
- forward
- backward
- 与 QwenRMSNorm 对比

### 5.2 Fused SwiGLU Triton

目标接口：

```python
swiglu_triton(
    gate: Tensor,
    up: Tensor,
) -> Tensor
```

计算目标：

```text
output = silu(gate) * up
```

后续可以进一步把 projection 和 activation 融合，但第一版只实现 elementwise fusion。

### 5.3 RoPE Triton

目标接口：

```python
apply_rope_triton(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]
```

验收重点：

- interleaved pair 旋转正确
- q/k shape 支持不同 head 数
- 不产生非法越界读写
- 与 PyTorch reference 对齐

### 5.4 GQA / FlashAttention Triton

目标接口：

```python
attention_triton(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    causal: bool = True,
    attention_mask: Tensor | None = None,
) -> Tensor
```

第一版只支持：

- contiguous input
- causal attention
- fixed head dimension
- no KV cache

之后再扩展：

- GQA，不显式 materialize K/V repeat
- variable sequence lengths
- KV cache
- decode kernel
- backward kernel

不能一开始就写完整 FlashAttention。先写 forward、验证数值，再做 benchmark，最后考虑 backward。

---

## 6. Benchmark 规范

每个算子至少比较：

```text
PyTorch eager reference
torch.compile（如果适用）
Triton custom
```

每次 benchmark 必须：

1. warmup 多次
2. 正式测量多次
3. CUDA 前后调用 `torch.cuda.synchronize()`
4. 报告 median / mean / p90
5. 记录 peak allocated memory
6. 固定 batch、sequence length、hidden size、dtype

报告格式建议：

```text
operator: rms_norm
shape: [1, 128, 2048]
dtype: bfloat16
reference median: ... ms
triton median: ... ms
speedup: ...x
reference memory: ... MiB
triton memory: ... MiB
max abs error: ...
```

一个算子变快不代表完整模型一定变快，因此还要测：

```text
完整 block
完整 Tiny-Qwen
真实 Hugging Face Qwen3
```

---

## 7. Integration 计划

当前模型使用 PyTorch baseline。下一步为每个算子增加实现选择：

```python
model = TinyQwenForCausalLM(
    config,
    rms_norm_impl="torch",
    attention_impl="torch",
    swiglu_impl="torch",
)
```

之后可以切换：

```python
model = TinyQwenForCausalLM(
    config,
    rms_norm_impl="triton",
    attention_impl="triton",
)
```

集成顺序：

1. 单算子 custom forward 通过
2. 单算子 backward 通过
3. 替换 Tiny-Qwen 中的一个模块
4. Tiny-Qwen forward/loss/backward 通过
5. Tiny-Qwen training smoke test 通过
6. 对照 HF Qwen3 的单层输出
7. 最后替换真实 Qwen 的模块

不要直接修改 `transformers` 源码。使用模块替换、adapter 或显式 dispatch。

---

## 8. 当前阶段的运行命令

进入项目：

```bash
cd /data1/tzh/proj/qwen_operator_lab
export PATH=/data1/tzh/envs/cs336_basics/bin:$PATH
export UV_PROJECT_ENVIRONMENT=/data1/tzh/proj/systems/.venv
export UV_NO_SYNC=1
```

运行单元测试：

```bash
uv run --no-sync pytest -q
```

运行 Tiny-Qwen 训练：

```bash
uv run --no-sync python -m training.train_tiny --steps 20
```

运行 Hugging Face baseline：

```bash
uv run --no-sync python profile_qwen.py
```

---

## 9. 近期里程碑

### Milestone 1：Baseline 完整性

- [x] HF Qwen forward profiling
- [x] Tiny-Qwen forward
- [x] loss / backward
- [x] optimizer step
- [x] checkpoint 保存
- [x] 基础单元测试

### Milestone 2：Qwen 结构对齐

- [x] 增加 Qwen3 q/k norm
- [x] 使用显式 `head_dim`
- [x] 修正 Qwen3 split-half RoPE
- [x] 模型级共享 position embeddings
- [ ] 完善 position_ids
- [ ] 完善 attention mask
- [x] HF Tiny 与自研 Tiny 的 logits/loss parity
- [ ] 完成单层输出对照
- [ ] 完成最小权重映射测试

### Milestone 3：第一个 custom kernel

- [ ] PyTorch RMSNorm reference
- [x] RMSNorm Triton forward
- [x] RMSNorm Triton backward
- [x] RMSNorm correctness tests
- [x] RMSNorm benchmark
- [x] 接入 Tiny-Qwen

### Milestone 4：融合 MLP

- [ ] SwiGLU reference test
- [ ] fused SwiGLU Triton
- [ ] forward/backward 对照
- [ ] block-level benchmark

### Milestone 5：Attention

- [ ] GQA reference 对照
- [ ] Triton forward attention
- [ ] causal mask
- [ ] benchmark
- [ ] KV cache
- [ ] decode path

### Milestone 6：真实 Qwen 集成

- [ ] HF 模块替换 adapter
- [ ] 单层输出对齐
- [ ] 完整 logits 对齐
- [ ] 端到端延迟 benchmark
- [ ] 峰值显存 benchmark

---

## 10. Debug 原则

每次只改变一个变量：

```text
一个算子
一个 dtype
一个 shape
一个实现后端
```

出现错误时按以下顺序定位：

1. shape
2. dtype
3. device
4. forward 数值
5. backward gradient
6. optimizer 更新
7. block 输出
8. 完整模型 logits
9. 性能

正确性永远先于性能。Triton kernel 如果更快但输出错误，不算完成。

---

## 11. 评审意见采纳决策（2026-08）

对外部评审建议的处理不是全部立即实现，而是分为“立即采纳、确认后采纳、后置扩展”三类。

### 11.1 立即采纳

以下内容直接影响当前 correctness，应优先进入 Milestone 2：

- 显式增加 `head_dim`，不再只用 `hidden_size // num_heads` 推导。
- 增加 Qwen3 的 head-dimension `q_norm` 和 `k_norm`。
- 核对 projection bias，Qwen3-1.7B 当前配置为无 bias。
- 修正 RoPE 为 Qwen3 的 split-half `rotate_half` convention：
  `x[..., :D/2]` 与 `x[..., D/2:]` 配对，而不是 adjacent-pair interleave。
- 在模型级计算一次 `(cos, sin)`，传给每一层，而不是每层重复创建。
- 将 shape capture 和 performance timing 分成两个路径。
- 所有训练 benchmark 显式使用 `use_cache=False`。
- 将 training、prefill、decode 作为三个不同 workload 设计测试和 benchmark。
- 在写 Triton 前，增加同配置 Hugging Face Qwen3 与 Tiny-Qwen 的 parity harness。

### 11.2 确认后采纳

这些建议有价值，但需要先用当前安装的版本和本地 checkpoint 验证：

- Qwen3 的 q/k norm 具体调用顺序和权重形状。
- `position_ids`、`position_embeddings` 和 RoPE broadcasting 的 batch 语义。
- `logits_to_keep` 在当前 Transformers 版本中的可用范围。
- PyTorch 当前版本的 SDPA `enable_gqa` 与后端选择。
- HF `Cache.update` 的具体签名和 layer index 顺序。

验证方式优先采用本地安装的 `transformers==4.51.0` 源码和一个小 Qwen3-compatible config，不直接依据不同版本的网络源码猜测行为。

### 11.3 后置扩展

这些内容不进入当前可训练 baseline 的阻塞路径：

- long-context RoPE scaling（当前 Qwen3-1.7B 配置为 `rope_scaling=null`，先保留配置入口）。
- dynamic/static/paged KV cache 的完整实现。
- decode 专用 Triton kernel。
- batch 内不同长度请求和 continuous batching。
- `torch.library` custom-op 注册和 `opcheck`。
- effective bandwidth、CUDA Events 等高级 benchmark 指标。
- 独立 qwen 项目虚拟环境和完整 lockfile 重构。

### 11.4 修订后的执行顺序

```text
冻结 Transformers / PyTorch / Triton 版本
    ↓
Qwen3-compatible Tiny config（显式 head_dim）
    ↓
q_norm / k_norm + split-half RoPE
    ↓
HF Tiny 与自研 Tiny 的中间结果 parity
    ↓
分离 shape capture / timing
    ↓
RMSNorm Triton
    ↓
QK-Norm + RoPE Triton
    ↓
SwiGLU 实验
    ↓
training/prefill attention
    ↓
decode + KV cache
```

这意味着当前最先要修的不是 FlashAttention，而是 Tiny-Qwen 的 Qwen3 结构 parity。
