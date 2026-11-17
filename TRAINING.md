# 训练

## 选择路径

| 路径 | 入口 | 使用范围 |
| --- | --- | --- |
| V14 完整 Triton GPU 数据通路 | [train_qwen3_runtime_v14.py](scripts/train_qwen3_runtime_v14.py) | 模型前后向、AdamW、GPU 张量通信和辅助数据操作由 Triton 实现；默认 `--runtime full`。 |
| V13 训练性能方案 | [train_qwen3_peer_rs_v13.py](scripts/train_qwen3_peer_rs_v13.py) | Triton 模型与优化通信；保留部分 PyTorch/FSDP 基础设施，固定实验中更快。 |
| HF 对照与旧版训练辅助 | [training/](training/) | 数据、验证、FSDP 主机调度和对照路径。 |
| 小模型快速测试 | [model/config.py](model/config.py) | 两层 Tiny 配置，与正式 8B 入口分开。 |

正式入口加载本地官方 `Qwen/Qwen3-8B` 权重，检查 36 层和 8,190,735,360 个参数。主机端仍使用 Python/PyTorch 的 autograd、FSDP 状态管理及 CUDA 驱动服务。

## 环境与运行

已测环境为 Python 3.11、PyTorch 2.10.0+cu128、Triton 3.6 和 4×RTX A6000；实验使用 BF16 计算及 FP32 归约/优化器状态。准备官方权重、固定语料与 input manifest 的步骤，以及四卡完整命令，见 [完整复现步骤](QWEN3_TRITON_REPRODUCE.md#v14-full-training-gpu-runtime)。输入路径和输出目录需要按本机环境配置。

```bash
# 查看完整训练入口参数；不会启动训练
CUDA_VISIBLE_DEVICES= bash scripts/research_python.sh scripts/train_qwen3_runtime_v14.py --help
```

使用前确认四张 GPU 可用。仓库的研究 wrapper 将缓存和临时文件放入 `.runtime-research`；历史控制脚本和 V14 入口的输出路径有 `/data1/tzh` 限制。权重、语料及训练产物不随源码发布。

## 已验证的结果

4×A6000、每卡 B1、序列 512 的固定协议中，V14 完整运行时约 5.927 秒/步，HF＋Flash 约 5.276 秒/步；V13 性能方案约 4.52 秒/步。V14 是完整 Triton 实现路径，V13 是该实验中的性能选择。

8B 训练对照覆盖全部 399 个参数的梯度检查、16 步质量比较和前后完整 512 块验证；这些是对应历史源码版本的短程检查；当前源码包含后续推理兼容修改，尚未重新完成四卡训练全量验收。检查不证明长期预训练收敛，也不保证其他硬件或形状的性能。详见 [结果汇总](docs/RESULTS.md) 与 [V14 验证记录](docs/history/TRITON_RUNTIME_V14.md)。
