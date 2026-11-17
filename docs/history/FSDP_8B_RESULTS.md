# 真实 8B 多卡训练结果

以下数字属于重构前源码。原始代码及相对路径保存在 `experiments/legacy/`，用根目录 `scripts/legacy_python.sh` 复核；不代表当前重构入口重新取得相同速度。见 [目录导航](REPOSITORY_LAYOUT.md)。

实际模型是 DeepSeek-R1-0528-Qwen3-8B：8,190,735,360 参数、36层、hidden4096、intermediate12288、Hq32/Hkv8、D128。不是原版 Qwen3-8B 或 7B，不是随机容量测试。

## 完整任务对照

四卡 RTX A6000，S512/B1/全局 batch4。BF16 计算，FP32 master、AdamW 状态及梯度归约，Flash SDPA。冻结 WikiText 续训协议每32步全量验证；min_delta=0.002、patience=8、min_checks=10；安全截止不算平台期。恢复实际最低 validation 的参数，再对全量 test 评估一次。

### 相同 expandable allocator、实际 cap 41.875 GiB/卡

| 指标 | 官方 native K36/every1 | Torch decoder exchange K32/every1 |
|---|---:|---:|
| 平台期更新数 / 验证次数 | 1152 / 37 | 1184 / 38 |
| 完整作业秒 | 8408.47973 | 7705.41240 |
| 记录 step 中位秒 | 6.14705 | 5.29121 |
| 最佳恢复后 test loss | 2.10615081 | 2.10707202 |
| allocated 峰值 GiB/卡 | 40.05251 | 38.61495 |
| reserved 峰值 GiB/卡 | 41.86719 | 41.78906 |

总耗时观测减少8.36%，step中位耗时减少13.92%；test delta +0.00092121。四rank实际分配器、回调、源码、历史覆盖和资源证据独立审计通过，本轮无OOM。候选本地打包/归约为Torch，不归因Triton。

证据：results/fsdp_exchange_expandable_full_quality_final_v1.json。此前非expandable作业在第129更新OOM，原始失败保留；本轮通过不等于已经隔离分配器根因。

### 独立 40 GiB、同 K34/every1 的 bounded/Triton 对照

| 指标 | native | bounded decoder |
|---|---:|---:|
| 平台期更新数 | 1152 | 1184 |
| 完整作业秒 | 8355.43846 | 7820.53875 |
| 记录 step 中位秒 | 5.87833 | 5.38240 |
| test loss | 2.10640750 | 2.10626447 |
| allocated 峰值 GiB/卡 | 39.33813 | 39.33813 |

观测总时间少6.40%，step中位耗时少8.44%；模型allocated峰值没有下降。带安全校验的有界scratch适配器与上一配置不同，不能叠加收益。证据：results/fsdp_scratch_bounded_quality_final_v1.json。

两项全部validation均512 blocks/261632 tokens，test均583 blocks/297913 tokens。单种子、共享硬件、停止步数不同，不证明统计等价、质量无损或纯实现因果加速。没有导出这些长程作业的最佳权重，也没有写完整optimizer checkpoint。

## profiler 与未采纳方案

最终配置四组ABBA诊断通过：候选step两顺序少7.80%/10.49%。只采rank0的一步trace，profile与稳态计时分开；NCCL时间含等待，不能当wire带宽。

降低到K24虽减少allocator重试/全流同步，却未带来稳定额外收益；固定K32将本地归约换Triton也未证明稳定收益。因此没有修改速度默认或追加这两项长训练。

证据：results/fsdp_exchange_profiles_final_v1.json、results/fsdp_retention_pressure_final_v1.json、results/fsdp_decoder_backend_final_v1.json。

## 其他已验证能力

- 自研同步分片原型已完成真实8B更新、HF导出、重新加载与生成；完整作业没有胜过官方FSDP，不采用为速度默认。见 [分片原型](MINIMAL_FSDP.md)。
- 真实8B、同world4的DCP恢复，两步重放输入/loss、399参数条目/rank的FP32 master/AdamW、buffers/groups及四类RNG精确一致。原生首次等待超时保留，不称reader加速或弹性恢复。见 [恢复指南](CHECKPOINT_RESUME.md)。
- S2048 K32/every1相对K8/隔层重算，两轮短测吞吐+26.12%/+32.89%；没有该组合完整质量证明。证据：results/fsdp_s2048_final_confirmation_v2.json。
- 1.7B双卡、S512/B1、24GiB预算，K28相对K0短测吞吐+29.7%/+26.5%；确定性三步完整梯度/master/AdamW状态哈希一致，不是长程质量证明。证据：results/fsdp_1p7b_two_card_screen_v2.json、results/fsdp_1p7b_full_state_gate_v2.json。

## 当前决定

默认保留官方实现，私有接口通信扩展仅作显式研究选项。HSDP、任意FP32梯度降精度、FP32微批累积、自定义除数、跨节点和弹性恢复都不在这些证明范围内。当前没有本轮活跃训练队列；不同工作负载没有一个可无条件通用的最优配置。
