# V21 SAM M0 完整终态（2026-09-05）

V21原进程执行342.520254秒后按固定M0门自动停止，状态M0_FAIL；
16:49:44实际观测原PID32331已退出、GPU1MiB/0%。Q1 fold/endpoint/epoch
均为0，未生成新checkpoint，没有D1/dev/official检索或模型改进指标。

三fold六模型共48个preflight batch完整配对，初始化、增强和五路输出SHA
均相同；各模型98,800,141总参数、7,841,292训练参数、203训练tensor。
普通/SAM各8容量优化步，前向反传分别8/16对；SAM固定batch100优化步、
200对前向反传。M0项目训练总116步、224对前向反传，另有48个forward-only
preflight batch；不能将这些计成Q1主训练预算。

容量峰值reserved分别6126/6284MiB。两端容量和SAM100步的第一遍与实际
更新梯度都覆盖203/203训练tensor，无缺失、overflow0、冻结state不变。
七BN每实际step计数只+1，SAM参数及第一遍统计按源码逐步精确恢复。
实际扰动范数范围0.04999999329447746–0.050000011920928955，符合rho0.05。

唯一未通过的固定M0条件是100步过拟合超额损失比：
原参数点loss第1/100步更新前为0.6110473871231079/0.5914160013198853，
解析底0.75H=0.57838292104621，比例0.39899872365870204>0.1。
用于更新梯度的扰动点loss为0.627093493938446/0.6106454133987427。
完整100步原始轨迹都保存；中途最小原参数loss0.5813781023025513，
末20步均值0.5843785017728805，都只是轨迹描述，不替代预先固定的第100步。
不能择最小值、均值、更换batch/rho/步数或放宽门槛使本次通过。

该结果只否定本次M0准备资格，尚无SAM的heldout检索结果，不足以断言SAM
检索泛化一定有害。此固定V21运行封存，不进行rho/LR/epoch/种子扫描或重训；
后续优化需要新的证据和新主假设，不能把改名重跑当新实验。

原始M0 summary297310字节，SHA
2ecc322270e4e1b82a77cf76e22ab76e359179fda9abc5ef7f2036db064d3c5d；
完整日志47488字节，SHA
0be3a21d007f1ff125779c13231c672ffaab67e884f02478b4be68e620f85194。
远端30个源/配置/方案/CLIP/V12/六source权重全字节SHA通过，目录新权重0；
本地JSON/math实际复核全6配对、116/224预算、100步全部分量、范数和状态
回执以及熵下界/比值，后两者数值差0。没有额外模型或权重张量加载。
工程数值可运行与M0固定准备门通过是两回事。独立M0终态审计待完成，
执行器的本地复核不替代独立审计。

## 证据索引

- 实际执行源码commit：3c393510f0e0a31bad602af8dd618a8dcdfe6ae6。
- 固定方案：refine-logs/v21/EXPERIMENT_PLAN.md；配置与源码SHA见
  evidence/trifusion_v21_preregistration_20260905.json。
- 原始完整M0：evidence/trifusion_v21_m0_seed42_3c39351.json。
- 全日志：evidence/trifusion_v21_m0_run_snapshot_20260905.log。
- 远端文件核验：evidence/trifusion_v21_m0_terminal_file_verification_20260905.json。
- 本地实际算术复核：evidence/trifusion_v21_m0_array_verification_20260905.json。
- T0：evidence/trifusion_v21_t0_20260905.json，三项远端CUDA数学测试PASS。
- 原始进程确认：evidence/trifusion_v21_launch_observation_20260905.json。
- 管理连接超时单列：evidence/trifusion_v21_launch_transport_20260905.json。
  screen -DmS等待结束引起的管理错误未导致训练中断；本次终态是固定M0门失败。
- 最终进程/状态观测：evidence/trifusion_v21_progress_20260905_1649.json。

## 当前研究范围

这是source身份的训练内工程门，无heldout mAP/R1、dev或官方测试结果；
T0 toy optimizer数学测试不是项目主训练。不能将源损失或容量当作SOTA证据。
V20完整Q1_FAIL保持。当前可部署dev最佳V8 58.4050/59.3939、exact Signal
58.0109/57.4545未改变；dev65及官方85.3/87.9目标未达到，全局任务继续active。
