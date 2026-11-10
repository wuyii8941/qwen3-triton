# 历史实验归档：不作为当前运行时依赖

本目录保存提交 `39d4e2e54db3bc6d6ea36f2c48bed2f57970d2a9` 中 428 个原始源码/测试/配置文件。`SOURCE_MANIFEST.json` 保存逐文件 SHA256；不要格式化或修改这里的原始文件来让旧结果“对上”新代码。

包括历次参数筛选、监控、分析、消融、旧训练 job、恢复/导出原型、推理和负结果。它们不都是失败代码，只是不属于当前主训练入口。

从仓库根目录执行：

~~~bash
bash scripts/research_python.sh scripts/verify_legacy.py
bash scripts/legacy_python.sh -m pytest -q tests
bash scripts/legacy_python.sh -m training.fsdp_convergence_job --help
~~~

`legacy_python.sh` 会切换工作目录和 Python 搜索路径，仍使用根目录的项目虚拟环境及缓存。不要使用 `python -m experiments.legacy.…` 代替它，否则原始相对路径和模块导入语义不同。

`results/` 链接到根目录已发布证据；本机 `artifacts/` 链接不发布，克隆后复核原始记录需自行提供。这里的原始 `scripts/research_python.sh` 只是快照，不作为启动命令使用。

原监督器可能校验冻结清单中的绝对路径。归档保留源码身份和只读审计能力，不承诺所有历史长训练监督器在新路径下能原样重启；不得为迁移重写旧证据。当前重新训练用根目录 `training.train` 并生成独立记录。

当前目录说明见 [仓库导航](../../REPOSITORY_LAYOUT.md)。
