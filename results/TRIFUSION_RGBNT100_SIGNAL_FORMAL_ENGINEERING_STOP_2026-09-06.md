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

## 单batch三项定位完成：Gram矩阵半精度量化与零点导数

56f094f、wrapper82470于10:35:40.134040结束48.6883秒；三项退出码fp16=0、anomaly=1、fp32=0，均按实际含义报告。
fp16/anomaly的四个ID/Triplet分量、Gram和Patch全部与捕获现场逐位相同；fp16仍153/195梯度非有限。
两次完整64×64 Gram中各3个行列式恰为0，坐标(3,3)/(4,4)/(5,5)，另各1个负值。
原normalized输入是FP32，但einsum/matmul在autocast下生成FP16 Gram；之后G.float()求det不能恢复此前舍入的信息。
异常模式在Signal utils/volume.py第60行sqrt(abs(gram_det))处报告AbsBackward0产生NaN。
完整FP32模型下两次Gram均0个零值，最小绝对det7.6721002884e-9，195/195梯度有限；总loss6.645667553，与原6.734400272不同。
这个全FP32比较改变了全模型数值精度，尚不足以证明只改Gram即可修复，因此不直接用于正式训练。

下一项局部修复已准备：tools/signal_gram_fp32.py只在原volume函数内关闭autocast并使用FP32输入，
公式、既有abs/sqrt、四组身份目标、Gram/Patch权重、AMPscale256与全部优化配置不改，无epsilon/clamp/fallback。
先用已保存两次完整Gram输入做原AMP/局部FP32算子回归（0模型前向），原非有限必须复现、修复后全梯度有限且零det=0；
通过后才运行完整batch64，比较原/修复3072D推理逐位相同、四身份分量/Patch原值不变和195梯度有限。
最多192条保存source记录前向、0更新/解码/heldout；计划登记于2026-09-06T10:43:29.465425+08:00，当前READY_NOT_RUN。
正式基线仍未恢复，单batch修复不代表长期训练或检索效果。

## 局部FP32提议未通过，零点数值处理有真实证据

8034451、wrapper83185于10:46:25.641635退出1，耗时7.948580876秒。
真实保存输入的原AMP算子复现loss3.970210552、各输入1536NaN；
仅Gram FP32后loss3.971660137，但仍1个零det(4,4)，三个输入各512NaN。
原注册的全梯度有限门失败，完整batch/模型构建未执行；本次模型前向0、优化0、仅2次真实特征算子backward。
全FP32模型之前能通过，不代表“只改Gram FP32”充分；该提议明确保持失败。

现准备tools/signal_gram_stable.py：FP32构造Gram并计算sqrt(abs(det).clamp_min(1e-12))。
实际风险是上述FP32零det产生NaN，不是推测edgecase。1e-12沿用作者layers/triplet_loss.py平方距离开方前的已有数值下限。
这是明确的零点数值定义变更：绝对det低于1e-12时输出1e-6、对该det的梯度为0；不冒充原无保护公式逐位等价。
没有阈值扫描、额外损失、fallback、学习率或scale修改；原Signal源码、原正式runner/config和失败提议保持不变。
新回归保留原AMP红例，对同输入要求稳定化后梯度有限、真实1个rawzero仍被记录且受到保护，
通过后才做完整batch195梯度/四身份分量Patch不变/3072D精确推理；最多192模型记录前向/0更新。
计划登记2026-09-06T10:51:50.831737+08:00，当前READY_NOT_RUN。原FP32提议失败条件不被改写，新数值处理单独注册。
