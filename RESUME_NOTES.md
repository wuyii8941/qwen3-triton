# 简历建议与证据索引

## 新增训练侧可选描述（2026-11-05）

- 为Qwen3-1.7B实现Triton共享embedding/LM-head梯度合并，结合autograd临时Tensor生命周期优化；
  在A6000、BF16、B1/S512下，相对compiled HF + FlashOptim，两轮完整训练step耗时降低约
  **5.5–6.6%**，peak allocated由约11.14降至9.85 GiB（少约1.29 GiB）。
- 建立4组batch/sequence、4路径、正反顺序的完整step消融，补充完整梯度、
  checkpoint恢复与256步变动数据/独立validation检查；本轮目标回归67项通过。

必须说明：BF16/no-master、原Flash SDPA不变；这是稳态step，不是新增收敛作业的总耗时。
短期validation最大差0.00151不等于质量无损。S2048仍有内存回退，不外推全部shape。
Graph在更强、无额外workspace限制的对照下未显示足够额外收益；多张量optimizer亦保留负结果，
不能把约9%的早期受限环境结果或单算子3.86×写成普遍端到端收益。
实现：`training/tied_embedding.py`；统一证据：`artifacts/optimization_wave_final_v1.json`；
细节与启动方式：[训练优化筛选](TRAINING_REPLICATION_PLAN.md#最终强基线复核与推荐以本段为准)。

以下为历史阶段记录，其“当前回归”数字对应各自环境，不是本轮67项目标回归的替代总数。

最新完整工作清单：[优化总表](OPTIMIZATION_SUMMARY.md)。
当前回归 **165 passed，2 warnings**；历史阶段的测试数字在下方保留原口径。
新增 [LM head / compile 实验](LM_HEAD_COMPILE.md)：大词表 W8A16、跨 batch 权重复用、
last-token 连续化与完整 compile 强基线。量化组合已发现生成分歧，不写“无损”。
第一轮 B1/S128 的原有 Triton pipeline 与 compiled PyTorch 基本持平；
S512 更快。不能再将对 eager 的倍率描述成对 compile 的倍率。

本轮可选补充描述（只写自己能解释清楚的）：

- 通过 profiler 定位最后 token 的非连续布局导致 LM head 走低效 bmm；
  以激活连续化切换原生 mm，在 A6000、FP16、B4/S128 上使完整生成流程再降低约 2.5%。
- 将 compiled PyTorch prefill 与 Triton CUDA Graph decode 按阶段组合，共享 KV cache；
  B1/S128 的完整 prefill + 16 步生成相对已有 Triton Graph 降低约 3.8%，
  相对 compiled PyTorch 降低约 3.9%。S512 反而退化，保留 workload 适用范围。
- 探索 W8A16 LM head 和跨 batch 权重 tile 复用，两轮五配置端到端约 4–7% 改善；
  属于近似量化，保留原权重的快照额外占显存，叠加 MLP 量化时存在生成分歧。

以上百分比不能互相相加。编译路径数值、零重编译检查和负结果见新报告，
不是对 vLLM 的对比，也不是生产服务吞吐量。

以下开头数字保留上一阶段口径：
后续实验又将 V cache 写入、residual/RMSNorm 合并，每步 GPU kernel 从 381 降到 325，
相对此前 Graph 路径再改善约 1–2%，最终五组配置数值通过。
可选 B1 G16 W8A16 实验使较短 prompt 的 pipeline 再下降约 10–11%，但属于有权重误差的近似路径，
且额外保留 840 MiB 量化快照，不能包装成“无损量化/显存减半”。G64/G32 失败输入未隐去。
该阶段完整回归 **150 passed**；详细边界与数字见 [Decode refinement 报告](DECODE_REFINEMENTS.md)。
简历建议优先写无量化方案，量化部分作为能解释正负结果的补充实验。

## 新增：有完整 pipeline 收益的推理调度优化

可增加以下一条（明确是运行时 + kernel 集成，不把 Graph 的收益全部归功于 Triton）：

- 为 Qwen3-1.7B 构建静态 KV cache 与 CUDA Graph decode 原型，结合 packed QKV
  及 Triton QK-Norm/RoPE 直接写 cache；在 RTX A6000、FP16、B=1/2、prompt=128/512
  的三组配置中，完整 prefill + 16 步 decode 相对此前已优化 eager 路径达到约
  **2.4–2.8×** 加速；通过不同随机种子复测、数值一致性检查及 profiler 归因。

面试边界：Graph 是主要收益来源；packing + 直接写 K 相对 Graph 额外约 3–5%。
Packed 权重快照额外占用 448 MiB；使用锁定 PyTorch 2.5.1 的私有 native Flash 接口；
固定 batch/capacity、无 padding，并非生产级服务，也不是相对 compiled HF/vLLM 的结果。
FP16 两轮最大 logits 相对 L2 约 0.354%，不等价于广泛任务质量无损。BF16 仍有旧反例。

详细证据与复现：[Static decode 优化报告](STATIC_DECODE_OPTIMIZATION.md)。
最终代码完整回归 **137 passed**。以下保留此前各阶段的历史口径，不与新测量混用。

## 可使用的项目描述

**Qwen/Triton GPU Kernel 优化实践**

- 基于 Triton 实现 RMSNorm、RoPE、SwiGLU GPU kernel，建立 FP16/BF16、多 batch/sequence 配置的数值验证与 eager/torch.compile/Triton 三后端 benchmark。
- 优化 QK-Norm+RoPE 数据布局，消除输入/输出 transpose-copy；A6000 指定 BF16 workload 下，Q 分支 GPU 延迟由 55.4 μs 降至 29.1 μs，单次调用新增峰值分配由 12 MiB 降至 4 MiB。
- 完成三类 kernel 接入本地预训练 Qwen3-1.7B 的可恢复推理原型；保持 HF KV cache/Flash SDPA，验证逐层 hidden states、logits 及增量 decode；固定测试配置中观测到 prefill 延迟下降约 7–26%、decode 约 6–23%。

## 面试必须同时说明

- 第一项 shape matrix 为 (B,S)=(1,1)/(1,128)/(2,512)；真实模型 benchmark 为 B=1/2、prompt=128，固定单 token decode、预热后10次同步 wall-time。
- 55.4→29.1 μs 属于 Q 分支布局消融的 CUDA Graph GPU 时间，不是整模型 wall-time。
- FP16/BF16 的整模型 logits 相对 L2 最大分别约 0.10%/1.06%；argmax 在所测固定 token 序列上一致，不等价于全任务质量无损。
- 真实模型比较的是 HF eager+Flash SDPA，不是 compiled HF、独立 flash-attn 包或 vLLM。
- Kernel 推理接入局限于 Transformers 4.51.0、单线程 context；训练、并发服务、量化和变长/padding 尚未在该适配器验收。
- 保留负结果：RoPE 在部分 shape 慢于 Inductor；实验性 Triton GEMM fusion backward 未胜过 cuBLAS。不要使用早期 compile 回退造成的虚假优势。
- 扩大输入后的新反例：BF16 B=2、prompt=128 的一次 decode 中，原有及新增 Triton 路径均有 argmax 与 HF 不同。上面的“一致”只描述旧固定测试集，不应泛化。见 `RESEARCH_OPTIMIZATIONS.md`。

## 证据

### 新增训练侧端到端贡献（2026-11-02）

- 完整作业已实际结束：Qwen3-1.7B、WikiText-103 持续预训练、A6000、B1/S512，
  两组均按冻结验证平台规则停止（15750/17500步），不是固定1000步测试。
  同等验证质量累计耗时 7208→5912秒，**1.219×、耗时下降约18.0%**；
  各自平台停止并完成测试的完整进程耗时 8320→7682秒，下降约7.7%。
  选定checkpoint测试loss为2.39936/2.39764。仅一对顺序作业，不作跨种子无损保证。
  原生HF前后向+Flash SDPA未变，仅替换native fused AdamW；不是所有Triton算子共同加速。
  数据加载、验证、durable checkpoint均计入；共享预处理未计时且排除。
  协议/数据哈希、停止规则及checkpoint元数据审计已通过，最终回归270项通过。
  详见 [完整训练比较报告](TRAINING_CONVERGENCE_RESULTS.md)。

- 实现可接入原生HF Qwen3-1.7B训练循环的Triton AdamW原型，保持BF16参数/动量存储，
  针对beta补数、FMA及除法舍入进行数值消融，并验证更新量、动量、checkpoint恢复和64步短期轨迹。
- RTX A6000、B=1、完整28层forward+backward+optimizer，三轮随机顺序测量：
  S512 step 265.36→179.07 ms，S2048 493.83→409.59 ms，分别下降约32.5%/17.1%。
- 仅相对本机HF+native fused AdamW。S512胜过所测compile对照，S2048仍略慢于compile；
  上面新增的平台停止证据只覆盖该任务，仍无生产分布式保证。FP32标量算术与native double超参数表达式并非逐位一致。
- 全项目257项测试通过。证据：`artifacts/adamw_verified_contribution_v1.json`，
  实现与复现见 `RESEARCH_OPTIMIZATIONS.md` 第10节。不能将该贡献表述为“所有训练配置无损提速”。

新增可选项目亮点（限定 FP16、A6000 和报告中的固定配置）：将 Q/K RMSNorm + RoPE
融合为单次 Triton launch，并消除 HF GQA 的 K/V 扩展复制；完整 Qwen3-1.7B 的
prefill/decode 相对已有优化路径进一步降低延迟。两轮测量中 decode 约下降 25–32%，
短 prefill 约下降 26–31%；较长 prefill 收益波动约 4.5–16.4%。具体数据
与统计范围见 `E2E_ATTENTION_OPTIMIZATION.md`，不应外推为生产服务的普遍收益。
新增 BF16 反例中旧基线本身也超过 2% L2 门槛，不能宣称 BF16 全面质量无损。

- `artifacts/inference_evidence.json`：三后端算子与随机 block。
- `artifacts/qk_layout_ablation.json`：布局消融、逐位一致。
- `artifacts/hf_inference_float16.json` / `hf_inference_bfloat16.json`：真实模型计时、数值、profiler、版本/权重哈希。
- `artifacts/hf_generation_demo.json`：自然语言 greedy smoke test。
- `HF_INFERENCE.md`：复现接口、测量口径和局限。
- `RESEARCH_OPTIMIZATIONS.md`：2025/2026 文献筛选、新增 Add+RMSNorm 与 decode GEMV 消融；后者在指定配置相对已有 Triton decode 额外降低约 1.6–3.1% wall-time，仍为可选实验。
- `E2E_ATTENTION_OPTIMIZATION.md`：QK 融合/GQA 去复制的端到端五路径消融、kernel 数量证据、数值门槛与反例。

下一阶段若需要更强性能结论：增加 prompt length sweep、自然文本集、多轮交错测量与 clock/功耗控制，再对比 compiled HF。当前已经满足可复现的 kernel 优化与真实模型接入原型展示，不宜称为生产级推理引擎。
