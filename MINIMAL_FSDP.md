# Qwen 7B/8B 级全分片训练与部署

本次发布是基础版本：同步 all-gather、重算、FP32 reduce-scatter SUM 加 Triton 梯度平均。后续官方 FSDP 对照、NCCL AVG 消融等研究不包含在此版本，也不作为本版本的性能结论。

## 真实模型验证结果

实际使用本机 `/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`：它是 Qwen3ForCausalLM、36 层、hidden=4096、intermediate=12288、Hq=32/Hkv=8、D=128，共 **8,190,735,360 个参数**。这是 DeepSeek 的蒸馏 checkpoint，不是 `Qwen/Qwen3-8B` 原版权重，也不是 8B 随机初始化。用户的“7B”按 7B/8B 级规模处理，未换成更小模型。

当前四张可见 RTX A6000（每张约 48 GiB），Python 3.11 / Torch 2.10.0+cu128 / Triton 3.6.0 / Transformers 4.51.0。

- S128 / 每卡 B1 / 全局 batch=4：真实 token 数据上连续 5 次全参数 AdamW 更新通过，所有参数有有限梯度。不是 LoRA、量化或冻结层。
- 初始 logits 与未分片 HF BF16+SDPA 对照：四个 rank 最大误差均为 **0**。
- 每卡训练 allocated 峰值约 **35.75 GiB**，reserved 约 **39.31 GiB**。后续步约 7.34 秒，包含梯度有限性诊断，不包含数据读取/H2D；不是严格性能 benchmark，也未宣称快于官方 FSDP。
- 独立训练 3 步后导出 BF16 HF snapshot，普通 HF 重新加载后的最后位置 logits 最大误差 **0**，并完成 32 token 贪心生成；32 token 可能仅覆盖 reasoning 前缀，不是生成质量评测。
- **完整 S512 流程已通过**：4 卡 × 每卡 B1、真实 WikiText token 数据、3 次全参数更新，初始 HF 和导出后重新加载 logits 最大误差均为 **0**；每卡 allocated 峰值 **35.7568 GiB**、reserved **39.3359 GiB**。流程退出码为 0，`pipeline_status.json` 为 `verified`。
- S512 导出权重与原始 checkpoint 比较，第一层 Q projection 中约 **18.63%** 的 BF16 权重元素实际发生变化，最大更新 **3.0518e-5**；不是只加载并原样导出。
- 小型可分享数值清单：[results/fsdp_8b_deployment.json](results/fsdp_8b_deployment.json)。可运行 `bash scripts/research_python.sh scripts/collect_fsdp_evidence.py` 重新审计四个 rank、导出完整性和真实权重变化；清单包含相关源文件 SHA256，不包含模型文本、权重或原始语料。
- 实验记录：`artifacts/fsdp_r1_qwen3_8b_pretrained_v1/rank{0..3}.json`，导出/重新加载记录在 `artifacts/fsdp_r1_qwen3_8b_export_job_v1/`；所有 artifacts 和权重仍不进 Git。

### 可复用入口

```bash
# 自动执行：初始 HF 对齐 → 四卡训练 → HF 导出 → 单卡重新加载与生成验证
# 同一 output 不能同时启动多个流程；完成后再次运行请选择新 output。
bash scripts/research_python.sh scripts/deploy_sharded_8b.py \
  --model /data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --seq 512 --steps 3 --output artifacts/fsdp_8b_deployment_v1
```

运行阶段和真实进程 PID 写入 `pipeline_status.json`；`training.log`、`reload.log`、四个 rank 的 JSON、`verification.json` 位于 output。`hf_model/` 是训练后可直接交给 `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16, attn_implementation='sdpa')` 的模型。S512 已验证模型位于 `artifacts/fsdp_8b_deployment_v1/hf_model/`。首次运行需要足够 CPU RAM（本机约 376 GiB）和至少约 17 GiB 导出空间；初始化阶段会读 CPU 权重，不能只根据 GPU util=0 判断卡死。

四卡训练命令的底层入口为 `training/run_sharded_qwen.py`，可单独指定 `--export-hf`、`--data`、`--checkpoint` / `--resume`。BF16 模型导出只包含权重，不包含 FP32 master 和 AdamW 状态；它用于模型部署，不能替代精确训练恢复 checkpoint。

### 已知限制

- 当前是同步、单节点、无通信重叠的自研分片原型，不是生产 FSDP，也不是 Web 推理服务。
- DeepSeek checkpoint 的 YaRN 配置含 `attn_factor`，Transformers 4.51.0 会报告该字段不识别；本实现与同版本 HF 原样配置对齐，没有悄悄改 RoPE。只验证短上下文，未验证其 131K 长上下文语义。
- 需要所有 rank 相同有效 token 数、相同 collective 顺序；不支持随机 Dropout 重算、高阶梯度、弹性 world size、DDP hooks、自动通信重叠或跨节点。
- 分片 optimizer checkpoint 已在 Tiny-Qwen tied/untied 多卡测试验证；8B 全量 FP32 master+AdamW checkpoint 预计约 92 GiB（另需安全空间），当前磁盘不足，未保存或声称验证了实际 8B 完整恢复。BF16 HF 导出约 15.26 GiB，可用于部署但不等价于该 checkpoint。
- 不宣称收敛、质量提升、生产稳定性或胜过官方 FSDP。原版 `Qwen/Qwen3-8B` 下载已因找到完整本地 checkpoint 而停止，部分下载保留；不将它标记为下载完成。

## 后续实现：按 Qwen block 分片

`training/block_fsdp.py` 现在提供独立的 `BlockFSDP` / `shard_qwen`，不再局限于下文旧版的单单元 FP32 实验：

- CPU 加载模型，按 embedding / decoder block / final norm / LM head 分单元；每卡仅将本地参数区间迁移到 GPU。
- 参数 master、梯度通信和 AdamW 状态为 FP32；每个单元 all-gather BF16 参数用于计算，backward 重算后释放完整单元。
- 共享 embedding/head 共用一个 FP32 master shard，两次使用分别转换 BF16；梯度在 FP32 shard 上累加，优化器只更新一次。
- NCCL all-gather / reduce-scatter；Triton 本地 shard 复制/梯度平均。没有通信重叠，没有声称比官方 FSDP 快。
- `training/sharded_checkpoint.py` 支持同 world size 的分片保存/恢复，校验结构、精度和训练参数，拒绝覆盖旧目录。保存前检查磁盘；FP32 master + AdamW checkpoint 约需每个全局参数 12 字节，不含原始权重和额外安全空间。

### 已验证

- Tiny-Qwen 双卡 FP32、四卡 BF16，tied/untied 两种结构，分别检查 logits、loss、所有分片梯度和 3 次 AdamW 更新。
- 四卡 BF16 checkpoint 测试：第 1 步保存，主动破坏参数/清空 optimizer，恢复后继续第 2、3 步，仍与未分片参考对齐。
- 真实预训练 Qwen3-1.7B 四卡 BF16 全参数训练，B1/S128，3 次更新，所有参数有梯度且有限；每卡 allocated 峰值约 9.33 GiB，reserved 约 11.12 GiB。
- 上述 1.7B 同时验证 rank 不同数据分片的本地 WikiText uint32 corpus；这是短程可运行性测试，不是收敛证明。
- 官方 Qwen3-8B 全尺寸随机初始化模型四卡 BF16 全参数 AdamW 训练，B1/S128，3 次更新通过；每卡 allocated 峰值 35.75096 GiB、reserved 39.01953 GiB，后两步约 7.35 秒。仅为容量/链路证据，不是预训练权重、性能优化或收敛结果；完整 CPU 随机初始化约需 6 分钟。

### 命令

```bash
bash scripts/research_python.sh -m torch.distributed.run --standalone \
  --nproc_per_node=4 benchmarks/verify_block_fsdp.py --bf16 --checkpoint-test

bash scripts/research_python.sh -m torch.distributed.run --standalone \
  --nproc_per_node=4 training/run_sharded_qwen.py \
  --model /data1/tzh/models/Qwen/Qwen3-1.7B \
  --data artifacts/convergence_corpus_v1/train.bin \
  --steps 3 --seq 128 --output artifacts/fsdp_1_7b_real_data_v1
```

原版 Qwen3-8B 可选下载工具为 `scripts/download_qwen3_8b.py`，固定 HF revision 并对 LFS 文件校验 SHA256，权重保留在仓库之外。其下载尚未完成；真实部署改用上面已经完整存在的 DeepSeek-R1-0528-Qwen3-8B。`--random-init` 仅用于官方结构容量检查，不能替代预训练模型部署证据。

下载完成后真实 8B 运行命令（必须先确认所有权重校验通过）：

```bash
bash scripts/research_python.sh scripts/download_qwen3_8b.py
bash scripts/research_python.sh -m torch.distributed.run --standalone \
  --nproc_per_node=4 training/run_sharded_qwen.py \
  --model /data1/tzh/models/Qwen/Qwen3-8B \
  --data artifacts/convergence_corpus_v1/train.bin \
  --steps 3 --seq 128 --output artifacts/fsdp_8b_pretrained_v1
```

`scripts/deploy_sharded_8b.py` 默认优先使用本地 DeepSeek checkpoint；显式指定原版 Qwen3-8B 且缺少权重时才下载。状态写入所选 output 的 `pipeline_status.json`，默认 output 为 `artifacts/fsdp_8b_deployment_v1`。只有四个 rank 的真实训练、HF 对齐、导出重载和生成均通过才写入 `verified`；这仍不表示收敛或生产服务。不要在旧进程仍运行时重复启动。

分段下载最多 16 路并发，64 MiB 分片可断点恢复，最终按 HF LFS SHA256 校验；成功后只清理本脚本产生的已合并分片，不清理用户 checkpoint。临时分片与最终文件合并时需要额外磁盘空间。

同一条命令去掉 `--data` 仅测试 rank 不同的固定随机 token。无论哪种数据，3 步都只能证明执行链路，不是收敛或性能胜出证据。当前数据读取/H2D 不包含在 step seconds 内；finite 梯度检查包含在内，不能拿这些时间与此前严格 benchmark 直接比较。

运行器可选 `--checkpoint NEW_DIRECTORY` 保存、`--resume DIRECTORY` 恢复；`--steps` 表示本次追加更新数。不同 rank 使用不重叠语料区间；必须使用同一 world size / batch / seq / 数据文件恢复。不会自动清理用户 checkpoint。

当前支持本机单节点、确定性 Qwen3、use_cache=False。没有生产推理服务、弹性 world size 恢复、随机 Dropout 重算或重叠调度支持。下面保留最初原型说明，其限制不用于描述新增 block 路径。

入口：`training/minimal_fsdp.py::MinimalFSDP`。
这是用于理解和验证 full-shard 生命周期的独立原型，不替换现有 HF 训练默认路径，也不代表已经支持 Qwen3-7B。

## 谁负责什么

- PyTorch distributed / NCCL：broadcast 初始化、all-gather 参数、reduce-scatter SUM 梯度。
- Triton：初始化本地参数 shard 的复制、reduce-scatter 后平均梯度的本地 kernel。
- PyTorch autograd：重算模型以及求完整单元梯度；AdamW 仅接收本地 shard，因此优化器状态也按 rank 分片。

通信不是用 Triton 重写的；性能优化也未得到验证。这是分片训练基础设施原型，而不是新型通信算法。

## 一次训练更新

1. 初始化时所有 rank 暂时持有完整模型；rank 0 广播一致参数，flatten 并补零到 world size 整数倍。
2. 每个 rank 只保留自己的可训练 shard；模板参数迁移到 meta，不保留完整权重存储。
3. Forward：all-gather 得到完整单元参数，无梯度执行模型；返回输出，不保存完整参数或中间激活。
4. Backward：重新 all-gather，重算 forward，用 autograd 求本地数据对应的完整参数梯度。
5. reduce-scatter 将对应参数区间的梯度跨 rank 求和；Triton 除以 world size，得到平均梯度 shard。
6. 本地 AdamW 更新参数 shard，下一次 forward 再聚合更新后的参数。

要求每张卡的 local batch 大小相同、loss 使用相同 mean 约定。输入梯度保留本地 loss 的语义，只有参数梯度跨 rank 平均。

## 运行验证

在已配置的项目研究环境中，从仓库根目录运行：

```bash
bash scripts/research_python.sh -m torch.distributed.run \
  --standalone --nproc_per_node=2 benchmarks/verify_minimal_fsdp.py

bash scripts/research_python.sh -m torch.distributed.run \
  --standalone --nproc_per_node=4 benchmarks/verify_minimal_fsdp.py
```

每个进程使用自己的 LOCAL_RANK 对应 GPU；如需选择设备，在启动命令前设置 CUDA_VISIBLE_DEVICES。
验证模型包含 490 个参数，因此四卡还覆盖了非整除 padding。各 rank 使用不同输入，对比未分片模型的平均梯度，检查输出、loss、输入梯度、参数梯度及连续三次 AdamW 更新。

双卡实测通过：每卡 245 个参数，最终参数最大误差为 0。
四卡实测通过：每卡 123 个参数（总计补 2 个零），最终参数最大误差为 7.45e-9。

## 限制与下一步

- 当前仅支持 CUDA FP32、单 Tensor 输入/输出、全部参数可训练、无 buffer、无共享参数的确定性纯函数模型。
- 不支持 Dropout/随机操作、带副作用模块、参数或输入原地修改、AMP、高阶梯度、DDP hooks、分布式 checkpoint。模板不应在构造后修改。
- 当前未实现通信重叠、多单元调度、混合精度、自动 wrap、CPU 初始化或流式加载；没有与官方 FSDP 的性能比较。
- 一个单元在 backward 仍需要完整参数、完整梯度和重算激活。它降低的是持久状态复制，不能将多卡总显存视为无需调度的统一显存。
- 实际 Qwen 有 tied embedding 等结构，不能直接套入本原型。7B 接入前需按 Transformer block 分单元、支持 BF16、处理共享权重，并验证分片初始化和恢复。
- 本地 benchmark 保留完整 reference 做正确性对照，因此该进程峰值不是分片方案的显存 benchmark。

接口参考：[PyTorch distributed](https://docs.pytorch.org/docs/stable/distributed.html) 与 [官方 FSDP](https://docs.pytorch.org/docs/stable/fsdp.html)。本项目刻意只实现基础同步链路，不复刻官方完整实现。
