# 实验结果与证据

各行是各自固定协议的完整请求或完整训练步中位数，不应跨协议直接归因。

| 实验 | 硬件和形状 | Triton / 方案 | 同轮 HF | 结论 |
| --- | --- | --- | --- | --- |
| V13 训练 | 4×A6000，每卡 B1/S512 | 4.5148 / 4.5176 秒/步 | 见 [V13 记录](history/OPTIMIZATION_V13.md) | 同轮耗时减少 14.75% / 13.80%；包含训练和通信策略。 |
| V14 完整 Triton 训练 | 4×A6000，每卡 B1/S512 | 5.9268 / 5.9269 秒/步 | 5.2760 / 5.2756 秒/步 | 功能验收通过；性能慢约 12.3%。 |
| V15 推理 Graph | 单 A6000，B1/S65，生成 8 token | 202.146 / 202.154 ms | 220.338 / 220.151 ms | 完整请求耗时减少 8.26% / 8.17%；Eager 慢于 HF。 |

V14 的证据覆盖 399 参数梯度、16 步质量、训练前后 512 块验证和 GPU 事件审计；主机 FSDP/autograd、CUDA 驱动及数据服务仍保留。V15 通过原 24 项数值检查、两条 16-token 生成、512 块 S512 和五段 GPU 审计；额外文本 logits 诊断有 23 项超阈值。训练结果绑定历史源码；当前源码包含后续推理兼容修改，尚未重新完成四卡训练全量验收。以上是有限范围验证，不代表长期收敛或通用数值等价。仓库保留小型 JSON 摘要；权重、语料、原始梯度和 profiler 原始 trace 不随源码发布。

- [V14 训练证据索引](../results/qwen3_triton_runtime_v14_evidence_index_v1.json)
- [V13 训练证据索引](../results/qwen3_triton_v13_evidence_index_v1.json)
- [V15 完整数值与审计报告](../results/qwen3_triton_inference_v15_reference_api_qualified_v3.json)
- [V15 六组完整请求样本](../results/qwen3_triton_inference_v15_reference_paired_request_v2.json)
- [V15 审查与边界检查](../results/qwen3_triton_inference_v15_final_review_v1.json)
- [历史实验、消融及失败记录](history/README.md)
