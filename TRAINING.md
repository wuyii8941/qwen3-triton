# Qwen3-8B 训练

## 范围与依赖

唯一支持的训练入口是 `scripts/train.py`。它加载官方 Qwen3-8B BF16 权重，以四进程 FSDP2 运行完整模型；Triton 路径负责模型前后向、优化器及 GPU 数据操作，PyTorch 负责 autograd、FSDP 主机编排和 CUDA 驱动服务。

推荐环境为 Python 3.11、四张支持 BF16 的 GPU、足够的主机内存和本地模型/语料。安装 `requirements-qwen3-triton.txt`，并始终通过 `scripts/research_python.sh` 调用 Python。仓库不会自动下载或覆盖输入。

## 1. 准备并绑定输入

`configs/input_recipe.json` 固定模型与 WikiText 文件的哈希。先从已缓存 parquet 或已验证文本构建 token 数据，再把本地路径绑定到新的 manifest：

```bash
bash scripts/research_python.sh scripts/prepare_corpus.py \
  --model-dir weights/Qwen3-8B \
  --parquet-source datasets/wikitext-parquet \
  --output artifacts/wikitext

bash scripts/research_python.sh scripts/configure_inputs.py \
  --model-dir weights/Qwen3-8B \
  --corpus-dir artifacts/wikitext \
  --output artifacts/inputs
```

两个输出目录都必须是新目录。命令会核对配置、tokenizer、权重分片和三个语料 split 的哈希。

## 2. 导出同输入的 HF 参考

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  bash scripts/research_python.sh -m torch.distributed.run \
  --standalone --nproc_per_node=4 scripts/export_training_reference.py \
  --variant hf --save-reference \
  --input-root "$PWD" \
  --protocol "$PWD/artifacts/inputs/protocol.json" \
  --input-manifest "$PWD/artifacts/inputs/input_manifest.json" \
  --output "$PWD/artifacts/training_reference" \
  --quality-steps 1 --warmup-steps 1 --measure-steps 2 \
  --keep-full-layers 36 --checkpoint-every 1 \
  --skip-embedding-backward-gather --retain-lm-head
```

## 3. 运行 Triton 训练与验收

`configs/gemm_plan.json` 是最终矩阵乘计划。正式验收使用同一 protocol、manifest 与参考目录：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  GLOO_SOCKET_IFNAME=lo CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTORCH_ALLOC_CONF=backend:native,expandable_segments:False \
  bash scripts/research_python.sh -m torch.distributed.run \
  --standalone --nproc_per_node=4 scripts/train.py \
  --input-root "$PWD" \
  --protocol "$PWD/artifacts/inputs/protocol.json" \
  --input-manifest "$PWD/artifacts/inputs/input_manifest.json" \
  --reference-raw-dir "$PWD/artifacts/training_reference" \
  --gemm-plan "$PWD/configs/gemm_plan.json" \
  --output "$PWD/artifacts/training_run" --full-validation --trace
```

运行前确认四卡空闲且输出目录不存在。短程数值检查只证明固定协议下的实现一致性，不证明长期收敛；硬件、驱动、依赖或源码变化后都应重新做完整验收。
