# 当前训练运行时

从 `train.py` 开始读。这里只有一个真实多卡训练 CLI，不再混入参数搜索、监控、报告、1.7B 优化试验和不同轮次的 job。

- `data.py` / `evaluation.py`：数据与全量验证。
- `policy.py` / `stopping.py`：配置约束与停止规则。
- `best_state.py`：最佳参数的 CPU 快照与恢复。
- `fsdp/official.py`：默认官方 FSDP2。
- `fsdp/custom.py`：自研 block 分片，显式 `--backend custom`。
- `fsdp/communication.py`：显式开启的 Torch decoder exchange；不是 Triton 获胜结论。

启动命令、旧新命名对照、支持边界见 [目录导航](../REPOSITORY_LAYOUT.md)。历史完整训练、恢复/导出和单卡 FlashOptim 过程均在 [实验归档](../experiments/legacy/README.md)。
