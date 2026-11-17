# 来源与许可证

本仓库自行编写 Triton 模型计算及 GPU 运行时；PyTorch 负责张量、参数、autograd 和主机 FSDP 调度。权重和数据需自行获取，不包含在源码中。项目自有源码许可证尚未选定；以下第三方许可不替代项目许可。

- 模型结构与参数名称参考 [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)。
- Attention 算法与舍入参考 [FlashAttention 上游源码](https://github.com/Dao-AILab/flash-attention/tree/979702c87a8713a8e0a5e9fee122b90d2ef13be5/csrc/flash_attn)，其许可证保存在 [第三方目录](third_party/flash-attention/LICENSE)。
- 归一化、激活和优化器的算术顺序参考 PyTorch 上游的 [归约](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cuda/Reduce.cuh)、[激活](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cuda/ActivationSiluKernel.cu)和[优化器](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cuda/fused_adam_utils.cuh)源码；其许可证保存在 [第三方目录](third_party/pytorch/LICENSE)。
- GPU 通信与同步的算法参考 [NCCL 上游](https://github.com/NVIDIA/nccl/blob/3ea7eedf3b9b94f1d9f99f4e55536dfcbd23c1ca/src/device/all_reduce.h)和 [CUDA 驱动接口](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)。实际候选路径使用 Triton；CPU 使用 Gloo 协调元数据。
- 验证语料来自 [WikiText](https://huggingface.co/datasets/Salesforce/wikitext)，输入哈希见[结果目录](../results/)。

具体验证环境、源码哈希与有限范围结果见[结果目录](../results/)及[实验结果](https://github.com/wuyii8941/qwen3-triton/blob/main/docs/RESULTS.md)。
