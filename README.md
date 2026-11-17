# Qwen3-8B Triton 训练与推理

本仓库提供 Qwen3-8B 的 Triton 模型计算、训练运行时和单卡推理实现。

| 内容 | 入口 |
| --- | --- |
| 完整训练、输入准备与复现 | [训练指南](TRAINING.md) |
| 推理、KV cache 与 CUDA Graph | [推理指南](INFERENCE.md) |
| 数值、性能与验证范围 | [实验结果](docs/RESULTS.md) |
| Triton 算子 | [算子代码](qwen3_triton/kernels/) |

完整训练 GPU 数据通路涵盖模型前后向、优化器、张量通信和辅助数据操作；单卡推理涵盖 prefill、cache、decode 和贪心选词。Python/PyTorch 仍负责 autograd、FSDP 主机调度、请求控制和 CUDA 资源管理。

在固定的四卡训练对照中，完整 Triton 运行时比 HF＋Flash 慢约 12.3%。在单卡固定 Graph 推理对照中，完整请求耗时减少约 8%；Eager 仍慢于 HF。推理的额外逐步文本 logits 诊断有 23 个超阈值项。训练短程验证不证明长期收敛，当前代码也尚未重新完成四卡训练全量验收。具体条件见[实验结果](docs/RESULTS.md)。

[Tiny 模型配置](model/config.py)仅用于快速测试；正式训练和推理读取官方 Qwen3-8B 权重。第三方来源和许可证见[来源说明](qwen3_triton/PROVENANCE.md)；项目自有源码许可证尚未选定。

`qwen3_triton/kernels/` 是完整模型使用的算子，`qwen3_triton/runtime/` 管理训练中的 FSDP 通信、优化器和 GPU 辅助操作。`ops/` 包含独立算子及参考实现，供组件测试和 Tiny 模型使用。历史实验代码可从 Git 历史查阅。
