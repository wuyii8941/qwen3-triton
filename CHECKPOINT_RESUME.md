# 官方 FSDP2 保存与恢复

本文描述已归档的恢复/导出原型，源码相对路径以 `experiments/legacy/` 为根。通过根目录 `scripts/legacy_python.sh` 执行；当前 `training/train.py` 没有 optimizer checkpoint 恢复功能。见 [目录导航](REPOSITORY_LAYOUT.md)。

## 当前验证结论

真实DeepSeek-R1-0528-Qwen3-8B、四卡A6000、S512/B1、K32/every1，BF16计算与FP32 master/AdamW/归约：step2 checkpoint恢复后两步重放的输入/loss、399参数条目/rank的master和optimizer状态、buffers/groups、Torch CPU/CUDA/Python/NumPy RNG精确一致。

证据：results/fsdp_8b_resume_summary_v1.json、results/fsdp_existing_resume_sorted_v1.json。这限定同world、拓扑、版本和协议，不证明弹性、多节点、任意精度恢复或长程质量。保留单份约92GiB状态，没有保存第二份大参考。

原生首次600秒等待超时是失败记录；使用实验性偏移排序reader与更长进程组等待后通过。load阶段约1688–1692秒包含准备/同步，没有matched cold-cache对照，不能称reader加速。

## API

training.fsdp_checkpoint适用于optimizer-step边界、标准FP32 AdamW、官方FSDP2；自研BlockFSDP有自己的格式。

~~~python
from training.fsdp_checkpoint import save_checkpoint, load_checkpoint

# 所有rank在同一边界调用；这些变量由训练循环提供。
save_checkpoint(
    output / "step100", model, optimizer, step=100,
    data_state=data_state, metadata=metadata,
)
restored = load_checkpoint(
    output / "step100", fresh_model, fresh_optimizer, metadata=metadata,
)
step, data_state = restored["step"], restored["data_state"]
~~~

metadata需包含一致的模型、数据和协议身份；data_state需保存准确采样位置，不能只恢复权重后从头读数据。多worker DataLoader的预取队列和内部RNG不自动序列化。

## 安全约束

- 先reshard全部官方FSDP单元，保存FP32 master分片，而不是临时BF16完整参数。
- DCP full_state_dict=false、CPU offload；不在GPU汇总整个模型。
- 保存前检查磁盘/主机内存，拒绝覆盖、错误schema/metadata、缺完成标记和不支持optimizer。
- 每rank trainer/RNG sidecar与SHA校验，所有状态完成后才提交complete.json。
- 8.19B完整状态约91.5GiB，保守保存预算约100.7GiB；可用磁盘必须运行时检查，不写home、不删除用户权重。
- 成功HF导出或checkpoint提交都不是恢复通过证据。

## 复现入口

~~~bash
# CPU检查
CUDA_VISIBLE_DEVICES='' GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1 \
  bash scripts/legacy_python.sh -m pytest -q \
  tests/test_fsdp_checkpoint.py tests/test_dcp_reader_adapter.py tests/test_dcp_sorted_reader.py

# 仅做只读资源估算，不启动CUDA或保存大状态
CUDA_VISIBLE_DEVICES='' bash scripts/legacy_python.sh benchmarks/verify_real_fsdp_resume.py \
  --resource-plan --output artifacts/resume_plan_new
~~~

多卡门禁入口为 benchmarks/verify_fsdp_checkpoint.py，真实重放入口为 benchmarks/verify_real_fsdp_resume.py；自行执行前看 --help、确认GPU空闲和完整状态磁盘预算。benchmarks/dcp_sorted_reader.py是固定版本opt-in，不修改安装的PyTorch。原始checkpoint和重放日志不发布。
