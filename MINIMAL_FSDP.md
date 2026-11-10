# 自研分片原型与 HF 导出

本文的部署、恢复、导出原型路径以 `experiments/legacy/` 为根，统一通过根目录 `scripts/legacy_python.sh` 启动。当前自研训练后端位于 `training/fsdp/custom.py`，不是这些旧 CLI 的兼容别名。见 [目录导航](REPOSITORY_LAYOUT.md)。

这个原型用于理解参数分片、重算和collective生命周期，不是当前速度默认。官方FSDP2对照与完整质量结果见 [8B证据](FSDP_8B_RESULTS.md)，不再以初始原型发布状态描述当前仓库。

## 实现

training.block_fsdp.BlockFSDP / shard_qwen按embedding、decoder、final norm和LM head分单元：CPU加载后逐单元构建FP32 master分片，forward all-gather BF16计算参数，backward重算并做FP32 reduce-scatter。NCCL负责通信，Triton负责部分本地cast/复制/平均，不是用Triton替代NCCL。

这是同步单节点原型；异步预取实验未证明稳定端到端收益。所有rank需相同有效token数和collective顺序，不承诺随机dropout重算、高阶梯度、弹性world、跨节点或DDP hooks。training.minimal_fsdp保留为早期单单元基线，不能代替真实8B多单元实现。

## 已验证结果

真实DeepSeek-R1-0528-Qwen3-8B，四卡A6000、S512/B1，完成全参数更新、初始HF parity、BF16 HF导出、普通HF重新加载和生成。初始及导出后logits最大误差为0；参数实际变化，不是原样导出。生成smoke不证明任务质量。

证据：results/fsdp_8b_deployment.json。早期诊断step包含有限性检查，不当严格性能benchmark；自研原型完整作业没有胜过官方FSDP。

## 部署入口

需要本机匹配环境、权重、数据、足够CPU RAM和导出磁盘空间；先核对CLI与资源，使用新输出目录。

~~~bash
bash scripts/legacy_python.sh scripts/deploy_sharded_8b.py \
  --model /path/to/DeepSeek-R1-0528-Qwen3-8B \
  --seq 512 --steps 3 --output artifacts/deployment_new
~~~

流程包含初始HF对齐、多卡更新、导出、重新加载和生成；状态写入该输出目录。底层入口 training/run_sharded_qwen.py 提供data、checkpoint/resume和export-hf选项，以 --help 为准。

BF16 HF导出只含部署权重，不能替代FP32 master、AdamW、数据位置和RNG的训练checkpoint。真实8B完整恢复也已单独验证，见 [恢复指南](CHECKPOINT_RESUME.md)。

本checkpoint的YaRN配置在Transformers4.51.0会有attn_factor未识别警告；实验原样对齐同版本HF，只验证短上下文，不承诺131K上下文语义。
