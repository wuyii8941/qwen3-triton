# 汇总结论

下表是清理前最终一次固定协议验收的历史数据，用于说明当前工程方向，不代表清理后的提交已经重新测量。

| 实验 | 条件 | Triton | HF 对照 | 结论 |
| --- | --- | --- | --- | --- |
| 完整训练 | 4×RTX A6000，每卡 B1/S512 | 5.9268 / 5.9269 秒/步 | 5.2760 / 5.2756 秒/步 | Triton 约慢 12.3%，collective 与 overlap 仍是主要优化方向。 |
| 固定 Graph 推理 | 单 RTX A6000，B1/S65，生成 8 token | 202.146 / 202.154 ms | 220.338 / 220.151 ms | 完整请求约快 8.2%；Eager 仍慢于 HF。 |

训练验收曾覆盖全部 399 个参数梯度、16 步短程质量、训练前后 512 块验证和 GPU 事件审计。推理验收曾覆盖 24 项数值检查、两条 16-token 生成、512 块 S512 验证和五段 GPU trace；另有 23 个逐步文本 logits 诊断项超过阈值。

原始 JSON、梯度、权重、语料和 profiler trace 已从当前树移除，可从 Git 历史追溯。当前可复现入口是 `scripts/prepare_corpus.py`、`scripts/configure_inputs.py`、`scripts/export_training_reference.py`、`scripts/train.py`、`scripts/generate.py` 和 `scripts/verify_inference.py`；固定配置是 `configs/input_recipe.json` 与 `configs/gemm_plan.json`。

因此当前结论是：推理 Graph 路径在所测形状有优势；训练尚无性能优势。下一阶段只应在同一正确性门槛下量化 collective、通信/计算 overlap，以及 TP/PP/DP 组合，确认有提升后再纳入稳定代码。
