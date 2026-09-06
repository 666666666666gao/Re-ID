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

## 固定source捕获完成，根因仍待定位

8b412d0执行一次，wrapper81713，10:25:42.449230退出1，37.470637565秒。
原AMP断言在本诊断第34步复现：33个成功更新/2176源记录前向，step34 loss6.734400272有限、scale256到128且该步跳过。
195份梯度中153份包含非有限值，均在CLIP encoder；未据此认定上游损失或具体算子是根因。
原正式失败运行的步数仍未知；本次诊断消耗独立登记，不冒充正式训练有效步数。
初始状态与M0相同、前8步记录索引全部相同，但只有step1的全部标量逐位相等；
step2起差异保留（step6 loss差+0.722964287），尚未解释，不声称固定seed保证训练轨迹逐位复现。
已保存438600256字节真实异常batch及前向前model/buffers/RNG，SHA3c9b41a70a3e3bfabd317cea8b76f314ba16ded8a8246fd41e0561fc7f8286eb；只在远端保留。
本地完整34步/source/loss重算通过，非有限梯度计数与原记录一致；无本地模型/张量/图像操作。

下一固定诊断一次检查fp16/fp16_anomaly/fp32，各64条相同保存source batch，共192源记录前向、0优化更新、0图像解码/heldout。
异常模式自然保留PyTorch原始traceback；原精度先核对捕获loss，FP32只作数值定位，不改正式训练精度。
保存两次volume_computation3的全部64×64 Gram矩阵/行列式/输入，便于随后对真实触发算子最小化。
计划evidence/trifusion_rgbnt100_signal_v1_amp_batch_probe_plan_20260906.json，当前READY_NOT_RUN。
