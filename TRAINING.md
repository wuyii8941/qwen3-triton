# 训练

## 入口与范围

[完整训练入口](scripts/train_qwen3_runtime.py)默认启用完整 Triton GPU 运行时，加载本地官方 Qwen3-8B 配置和权重，并检查 36 层及完整参数数量。模型前后向、优化器、GPU 张量通信与辅助数据操作由 Triton 执行；FSDP/autograd 主机调度、数据准备、CUDA 驱动和主机设备传输由基础设施负责。

[模型算子](qwen3_triton/kernels/)包含 Linear、Attention、归一化、激活及对应反向实现；[训练运行时](qwen3_triton/runtime/)包含优化器、分片数据操作和通信。[Tiny 配置](model/config.py)用于快速测试，并非正式训练配置。

## 环境与输入

需要四张支持 BF16 的 GPU、足够的主机内存、本地官方权重及固定训练/验证语料。已测设备为四张 RTX A6000；依赖见[清单](requirements-qwen3-triton.txt)。研究脚本将缓存和临时文件放在项目 `.runtime-research` 下，正式训练入口要求输出位于 `/data1/tzh`。

按[输入校验清单](results/input_recipe.json)获取官方模型及 WikiText 数据；权重和语料不随仓库发布。在项目根目录准备语料和绑定输入：

```bash
TOKENIZERS_PARALLELISM=false bash scripts/research_python.sh \
  scripts/prepare_qwen3_triton_corpus.py \
  --model-dir weights/Qwen3-8B \
  --parquet-source datasets/wikitext-parquet \
  --output artifacts/reproduced_wikitext

bash scripts/research_python.sh scripts/configure_qwen3_triton_inputs.py \
  --model-dir weights/Qwen3-8B \
  --corpus-dir artifacts/reproduced_wikitext \
  --output artifacts/reproduced_inputs
```

完整梯度对照需要先创建匹配同一 input manifest 的 HF 参考，且需要足够磁盘空间：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  bash scripts/research_python.sh -m torch.distributed.run \
  --standalone --nproc_per_node=4 scripts/export_training_reference.py \
  --variant hf --save-reference --input-root "$PWD" \
  --protocol "$PWD/artifacts/reproduced_inputs/protocol.json" \
  --input-manifest "$PWD/artifacts/reproduced_inputs/input_manifest.json" \
  --output "$PWD/artifacts/training_reference" \
  --quality-steps 1 --warmup-steps 1 --measure-steps 2 \
  --keep-full-layers 36 --checkpoint-every 1 \
  --skip-embedding-backward-gather --retain-lm-head
```

使用同一输入和参考启动完整训练对照：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  GLOO_SOCKET_IFNAME=lo CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTORCH_ALLOC_CONF=backend:native,expandable_segments:False \
  bash scripts/research_python.sh -m torch.distributed.run \
  --standalone --nproc_per_node=4 scripts/train_qwen3_runtime.py \
  --input-root "$PWD" \
  --protocol "$PWD/artifacts/reproduced_inputs/protocol.json" \
  --input-manifest "$PWD/artifacts/reproduced_inputs/input_manifest.json" \
  --reference-raw-dir "$PWD/artifacts/training_reference" \
  --output "$PWD/artifacts/training_run" --full-validation --trace
```

以上路径是假设仓库位于 `/data1/tzh` 的示例；按实际路径调整，输出目录必须是新目录。完整协议包含短程质量、预热、计时、trace 和前后验证；运行前应确保四卡空闲。

## 验证与限制

固定四卡、每卡 B1、序列 512 的对照覆盖全部 399 个参数梯度、16 步质量和训练前后 512 块验证。完整 Triton 运行时约 5.927 秒/步，HF＋Flash 约 5.276 秒/步，结果见[实验结果](docs/RESULTS.md)。这些数据绑定验证时的源码；当前代码包含后续兼容修改，尚未重新完成四卡全量验收。短程检查不证明长期训练收敛。
