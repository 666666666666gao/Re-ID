# N1现有环境执行入口

复用已完成六端的实际远端环境：gaob@172.19.12.138:2026，根目录`/data/gaob/Re-ID/Trifusion`，Python`/data/gaob/Re-ID/conda-envs/tri_reid/bin/python -B`。不安装依赖、不重建环境。旧单卡AGENTS路径属于历史。

尚待fresh代码复核；只有复核无阻断问题后才执行一次以下预检查。先实查GPU健康、显存<500MiB及空间，选择实际空闲卡；此命令假设实查0/1健康空闲：

```bash
CUDA_VISIBLE_DEVICES=0 /data/gaob/Re-ID/conda-envs/tri_reid/bin/python -B tools/prepare_native_detail.py --output-dir /data/gaob/Re-ID/Trifusion/logs/native_detail_preflight_20261002_v1 --gpus 0 1
```

逐张量比较已完成roles与新模型全部非stem状态、高低分辨率全部初始化状态，再运行RGBNT201两端各8个真实训练batch，检查所有实际可训练参数累计非零有限梯度、相机和视觉更新、未训练状态不变、完整保存/严格重载输出一致。M0不采用官方检索分数，也不复用其更新权重做正式训练。失败保存原目录与日志，读原因后改正，禁止无变化重试。

只有预检查实际COMPLETE且两项M0回执验证通过才登记：

```bash
/data/gaob/Re-ID/conda-envs/tri_reid/bin/python -B tools/queue_native_detail.py --campaign /data/gaob/Re-ID/Trifusion/logs/native_detail_20261002_v1 --preflight /data/gaob/Re-ID/Trifusion/logs/native_detail_preflight_20261002_v1/preflight.json
```

六端high/low×三数据集均完整50轮。其余四端各自M0先行，M0失败不开始该端正式训练。复用240秒队列，无抢占/自动重试，失败停止新增调度并等待已启动端结束。每端最高fused mAP的一份权重、并列较晚轮、四指标同行；车辆主表只列mAP/R1。完整真实GT图库与原过滤、干扰身份保留，无rerank。

两条命令不表示已经执行。当前只实现N1信息来源，不执行N2/N3。完整结果收齐后才运行一次配对分析与fresh结果审计。


## 完整CPU报告等待器（§41.724）

前文预检与正式训练命令已实际执行，controller21951运行，不得重复。报告代码经两次实际fresh源码复核PASS；确认下列receipt和output尚不存在后，仅执行一次：

```bash
/data/gaob/Re-ID/conda-envs/tri_reid/bin/python -B tools/wait_native_detail_complete_analysis.py --campaign /data/gaob/Re-ID/Trifusion/logs/native_detail_20261002_v1 --output-dir /data/gaob/Re-ID/Trifusion/results/native_detail_complete_20261002 --receipt /data/gaob/Re-ID/Trifusion/logs/native_detail_analysis_waiter_20261002.json
```

持久进程240秒观察，只在六端完整终态后运行一次已保存GT距离CPU分析；失败记录不自动重试。此处记载命令不表示已启动，后续必须登记真实PID/WAITING与最终退出码。无模型前向/新训练，工程报告完成不等于N1门槛或全Goal达成。
