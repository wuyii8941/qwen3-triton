# 训练复现流程

这是当前入口说明，不是排队计划。没有本轮仍在等待的长训练。

## 区分三种工作负载

1. Tiny-Qwen：模型和算子集成、loss/backward/optimizer smoke，不是大模型性能证据。
2. Qwen3-1.7B：compiled HF + FlashOptim，BF16/no-master；单卡优化结果不能套到FP32 master协议。
3. 真实8B：官方FSDP2或自研分片，BF16计算、FP32 master/AdamW/reduce，冻结WikiText任务协议。

环境、缓存和数据准备见 [复现边界](FSDP_REPRODUCIBILITY.md)。模型、数据和原始artifacts不随仓库发布。

## CPU与小模型检查

~~~bash
CUDA_VISIBLE_DEVICES='' GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  bash scripts/research_python.sh -m pytest -q \
  tests/test_tiny_qwen.py tests/test_hf_parity.py \
  tests/test_fsdp_convergence_policy.py tests/test_fsdp_validation.py

bash scripts/research_python.sh training/flashoptim_convergence_job.py --help
bash scripts/research_python.sh training/train_tiny.py --help
~~~

GPU门禁和真实post-update全量验证预检必须在长程前通过；不能只看无梯度forward能否运行。

## 重跑完整8B对照

在原始本机门禁、冻结输入和匹配环境均已准备好时，确认没有并发自有GPU作业，使用新输出目录：

~~~bash
bash scripts/research_python.sh -m scripts.campaign_fsdp_exchange_expandable \
  --output artifacts/quality_reproduction_new
~~~

监督器先重审前置证据，再串行执行native K36/every1和Torch decoder K32/every1，双方实际cap41.875GiB、expandable allocator；检查源码与协议未漂移、资源余量、每rank回调和实际分配器。不修改冻结protocol来追求更好结果。

每32更新评估全部validation，min_delta0.002、patience8、min_checks10，恢复实际最低validation参数后只做一次全量test。4096更新/6小时是安全截止，不算收敛；不在1000步人为截断。该任务没有保存完整optimizer checkpoint或导出长程最佳权重。

这不是从GitHub克隆后自动下载并一键训练的命令；前置本机数据/门禁缺失应失败，而不是降级成更小模型或更少验证样本。

## 结果选择

只用数值门禁与训练性能选择后继，不用test选赢家。不把单次最快值、只完成warmup或部分记录当成完整结果。最近两项增量尝试未通过稳定速度门槛，保持实验选项，不追加长程训练。

完整结果见 [8B证据](FSDP_8B_RESULTS.md)，原始记录复核命令见 [复现指南](FSDP_REPRODUCIBILITY.md)。恢复/导出与任务质量是不同证明，不能互相替代。
