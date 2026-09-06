# RGBNT100 Signal 正式基线工程停止

记录时间：2026-09-06T10:22:31.091890+08:00。完整正式基线尚无检索结果。

60a3d0e上的wrapper80418于09:53:07启动，09:53:41.130891退出1，耗时33.77653791196644秒。
第一计划观察于10:09:23发现已结束，现场于10:16:01.901188取得；不把观察时间写成失败发生时间。
直接失败为原runner第201行AMP overflow; fixed run stops。三个fold中仅fold0开始、完整epoch记录0；
fold0目录为空，无checkpoint、training.json或retrieval。原运行成功优化步数未持久化，不能写成0或按耗时估算。

此前T0全量切片/协议和M0三折各8步、24更新均通过；它们不能保证30epoch训练无后续数值异常。
原runner在AMP断言之后才把步骤加到内存列表，整个fold训练返回才写training.json，导致本次早停缺少逐步证据。
此次先保存完整baseline.log、wrapper、launch、terminal和effective config及文件清单；均不覆盖原文件。
原config、runner、模型、优化器、调度器、AMP256和原工程门不变，不盲目重启正式训练。

固定诊断见 evidence/trifusion_rgbnt100_signal_v1_amp_capture_plan_20260906.json。
它直接调用原绑定train_source，通过三个固定行断点记录同seed42 source fold0，
首个overflow前后逐步落盘并保存原异常batch、前向前buffer/RNG/参数；最多一个epoch，第二epoch之前停止。
首先核对前8步与原M0的全部数值和索引；实际捕获后再决定零更新的单batch诊断。
这是工程定位，不是新方法试验、heldout检索或科学失败重跑，不改变任何科学门。

提前准备的完整终态核验器仍NOT_RUN，本失败目录不符合三折30epoch完整终态输入条件。
RGBNT201主目标与既有V23/V24/MSVR310原三角色科学负结果保持原结论。
