# 角色 Patch 读取实验执行表

2026-10-01 00:43执行快照；Goal已恢复ACTIVE / UNMET。四卡设备恢复且四张卡均通过新进程真实Mamba正反向。原故障campaign保持FAILED，恢复campaign运行中，正式验收2/6。

| 数据集 | 模式 | 当前状态 | best正式指标 |
|---|---|---|---|
| RGBNT201 | local_memory | 新真实M0通过；GPU2实际trainer1054535，13/50轮 | 未完成 |
| RGBNT100 | local_memory | 新真实M0通过；GPU3实际trainer1054538，3/50轮 | 未完成 |
| MSVR310 | local_memory | 原50轮权重独立严格评价成功，新评价实际退出0；全图库CPU验收通过 | 第10轮；52.3622 mAP / 67.8511 R1 |
| RGBNT201 | full_memory | 原50轮权重独立严格评价成功，新评价实际退出0；全图库CPU验收通过 | 第2轮；72.5335 mAP / 73.9234 R1 / 82.6555 R5 / 87.9187 R10 |
| RGBNT100 | full_memory | 新真实M0通过；GPU0实际trainer1062431，2/50轮 | 未完成 |
| MSVR310 | full_memory | 新真实M0通过；GPU1实际trainer1062597，11/50轮 | 未完成 |

四张卡当次利用率93%/100%/100%/100%，四个训练进程均实际存在且R(running)。原硬件故障中止端L201/L100保留，恢复版从相同初始化重新完整训练，未使用不含optimizer状态的中途best续训。四个新M0各8批/127个参数张量梯度非零/冻结基线不变/重载差0，32批原始记录已经归档。

三数据集local/full初始模型SHA匹配：201=4ea93f5e584ec9f0a1691edd65fa142844bfaf8665f25855e8e633497b1d0ad3；100=7b882dbf9e482586e20e8b0c3da8ca1c6bb0c3cd7d79b79252ef6aa2083a79fe；MSVR=e9c47306249d8f1e835e873d9663859de5aeb8d14a184ae035482f6199da0f0e。210份原运行源文件SHA不变，恢复只改变执行目录和评价进程，没有环境重建/依赖修改/损失/seed/门槛变化。

恢复controller1050794；目录logs/patch_memory_roles_recovery_20261001；脚本和登记计划见RECOVER_PANEL_20261001.py与RECOVERY_PLAN_20261001.md。旧进程均已不存在，不SIGCONT旧PID，不将原失败回执写为COMPLETE。恢复collector复用原verify()，明确区分原完整训练引用与四个fresh_full50目录。

现有两端尚未形成任何完整local/full数据集配对，等待其余四端完成后再判断支持范围的作用。逐轮best仅用于原固定选点，未完成端不填正式表。两个已完成端仍不足以达到Signal或当前SOTA，也不是整体目标完成。

观察按估计里程碑：首次正式恢复回执00:38，四卡训练00:43；下一个单次持久observer1080242已创建，首次2026-10-01 01:03 CST检查201 local与MSVR full，若未完成则240秒检查同一实际子进程，二者完成后收一次全量partial collector。100按当前整轮墙钟粗估02:35—02:55完成，估计不是终态。不要提前重复轮询或重启训练。

证据：logs/patch_memory_recovery_intake_20261001_0038/，logs/patch_memory_recovery_m0_20261001_0043/，logs/patch_memory_recovery_observer_20261001_0043.json。checkpoint/距离/probe仍在远端，所有失败保留。
