# Qwen3-8B Triton 训练与推理

本仓库提供 Qwen3-8B 的 Triton 模型计算、训练运行时和单卡推理实现。
当前结果来自固定硬件和短程验证，适用范围见下方说明。

| 需要什么 | 阅读或入口 |
| --- | --- |
| 训练：完整 Triton GPU 运行时、V13 性能路径 | [训练指南](TRAINING.md) |
| 推理：prefill、KV cache、decode 和 Graph | [推理指南](INFERENCE.md) |
| 性能、验证和限制 | [实验结果](docs/RESULTS.md) |
| Triton 算子代码 | [qwen3_triton/kernels](qwen3_triton/kernels/) |
| 过去的实验与失败记录 | [历史文档](docs/history/README.md) |

## 目前的结果

- **训练 V14**：完整 Triton GPU 数据通路通过 8B 短程功能验收；在 4×A6000、B1/S512 的对照中，5.927 秒/步，比 HF＋Flash 的 5.276 秒/步慢约 12.3%。
- **训练 V13**：保留 PyTorch 基础设施的性能方案，在其固定协议中比 HF＋Flash 快约 14%；该收益包含通信与训练策略，不全部来自 Triton 算子。
- **推理 V15**：单 A6000、B1/S65、生成 8 token 的固定 Graph 请求为 202.1/202.2 ms，同轮 HF Graph 为 220.3/220.2 ms，耗时减少 8.26%/8.17%。Eager 仍慢于 HF。

推理的 24 项原数值检查、512 块固定验证和生成对照通过；额外逐步文本 logits 诊断仍有 23 个超阈值项。训练结果来自历史版本的短程验证；当前源码包含后续推理兼容修改，尚未重新完成四卡训练全量验收，也不证明长期收敛。这些结果不代表所有输入、设备或服务场景。

早期 [Tiny 模型](model/config.py) 是快速测试用途；正式 8B 入口读取官方 Qwen3-8B 配置和权重。运行环境、数据和详细命令见 [训练指南](TRAINING.md) 与 [推理指南](INFERENCE.md)。

第三方来源和许可证见 [PROVENANCE](qwen3_triton/PROVENANCE.md)；项目自有源码的许可证尚未选定。
