# 算子与推理性能证据

## 三种口径不能混用

- 单算子：固定shape的CUDA Graph GPU时间或非捕获wall time，首次JIT/autotune与steady-state分开。
- 随机block：相同权重的Qwen结构block，不是预训练模型；单token算子不等于带KV cache的decode。
- 真实HF：完整checkpoint、cache与logits语义必须对齐，prefill/decode分开，profiler不混入计时。

benchmark_inference_evidence.py覆盖B/S=(1,1)、(1,128)、(2,512)，FP16/BF16与eager/compile/Triton；逐case检查输出并重置compile，避免cache上限静默fallback。显存用单次调用allocated peak delta，不是reserved或纯workspace。

## 固定环境下的实测例子

以下是2026-09-28、RTX A6000、Torch2.5.1+cu121/Triton3.1.0的已保存测量，不是当前Torch2.10重新测得的数字。

BF16、B2/S512，GPU CUDA Graph中位微秒：

| 算子 | eager | compile | Triton |
|---|---:|---:|---:|
| RMSNorm，hidden2048 | 135.48 | 14.44 | 14.34 |
| SwiGLU，intermediate6144 | 96.97 | 57.65 | 57.65 |
| RoPE，Hq16/Hkv8/D128 | 105.37 | 22.12 | 28.26 |

这支持融合相对eager的收益，不支持普遍超过Inductor。随机Qwen block在指定FP16/BF16、B1/S128与B2/S512形状下，三类算子替换的组合相对eager约1.38–1.45倍；没有相同block的compiled强基线，不能称端到端超过compile。

真实28层Qwen3-1.7B、prompt128、10次同步wall-time中位毫秒：

| dtype / batch | HF prefill | 替换后prefill | HF decode | 替换后decode |
|---|---:|---:|---:|---:|
| FP16 / 1 | 31.228 | 27.869 | 30.376 | 28.279 |
| FP16 / 2 | 40.305 | 29.929 | 38.999 | 31.378 |
| BF16 / 1 | 28.594 | 26.505 | 28.901 | 27.197 |
| BF16 / 2 | 39.295 | 29.763 | 41.549 | 31.885 |

这是早期HF eager参考，use_cache=true、logits_to_keep=1，未捕获Graph。不能外推服务吞吐、任意shape或与compiled HF/vLLM的比较。固定prompt+8步decode的logits相对L2门槛为2%，不是逐元素零误差或通用任务质量证明。

## 核心kernel最新契约

新增36项FP16/BF16矩阵覆盖RMSNorm的N128/2048、RoPE的D64/128和GQA、SwiGLU的257/6144，B1/B2、S1/17/129，随机上游梯度与输入不变性。FP32数学oracle只在最终输出/梯度舍入，不替代HF中间cast边界parity。连同既有测试，单卡4GiB cap实测52项通过。

## 未采用的结果

split-KV/GQA复用改善旧手写decode attention，但没有普遍超过Flash SDPA；GEMM融合没有稳定训练收益。量化/稀疏算法有形状依赖或自然文本误差反例，保持可选。旧的cache分配不公平计时、compile fallback和多variant常驻显存结果不用于结论。

## 入口与记录

~~~bash
python benchmarks/benchmark_inference_evidence.py
python benchmarks/benchmark_qk_layout.py
~~~

原始本机记录包括 artifacts/inference_evidence.json、artifacts/qk_layout_ablation.json、artifacts/hf_inference_float16.json、artifacts/hf_inference_bfloat16.json。它们不发布，不得把不可取得的原始记录伪装成克隆后可下载的数据。当前可运行接口见 [HF接入](HF_INFERENCE.md)。
