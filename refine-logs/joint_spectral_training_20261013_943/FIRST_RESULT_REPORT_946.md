# §946：第一端完整结果与原队列后续状态

第一joint_delta RGBNT201/seed42已经完整fresh50并通过首次strict。Same mAP-best E8：mAP/R1/R5/R10 = 74.5331/79.1866/88.2775/91.8660。它是1/3正式完成，不是三条件全完成；没有新车辆结果、seed或调参。

相对封存原raw semantic，数字变化为mAP +0.1582、R1 +0.3588；相对独立shared-adapter global，mAP +0.2364、R1 +0.2392。这是已有接受结果的算术差值，还不是完整query配对报告；相对旧semantic同时改变了共同dense读出/slot池化和增加589836参数，不单归因乘积。全Signal既有本地80.3029/85.1675参照仍未超过；上述净增量不到0.5项目决策线，也不是统计无效证明。

关键同参数的joint_delta−add_delta和joint_delta−joint_absolute尚未收齐，不能凭这一端证明乘积或reference有独立作用。两种已完成prepare的完整初始化model-state SHA相同，第三项实际初始化仍待执行；不预签三项完全通过。

Formal50轮/2649训练batch steps；不把batch记录数冒称逐步核验的有效optimizer更新数（M0才有8次有效更新证据）。Full training receipt 2026-10-13T23:53:22.642952+08:00→2026-10-14T00:33:33.706282+08:00，实际40分11秒，含逐轮评价/保存，不以history.seconds之和冒充完整成本。首次strict命令00:33:35→00:34:20 EXIT0，未重试/更换best。

E50 mAP/R1 71.1558/73.4450；mAP best→末轮下降3.3773点，仍部署同一E8 best，不使用末轮，不声称训练不足或已定位后期退化原因。

原prepare578058、M0 578635、fresh50 580021、firststrict670402均第一次EXIT0。Best SHA `5397835ef2654f5f788a40e2157e264abf4cdec1bc7ebc82165a94fbe8c7dc07`；保存distance SHA `c7b8458419fa4ff528706c8d541071b412ce9ac3c10861ec0a60f47085714b71`，正式receipt SHA `03f3848d1ddddd5e35edcca097349ec4dbec44eccd5fc7cb1fd1b636305fce59`。原supervisor578010/controller578011未重启；第一端完成后按预登记继续另两项，并非看到分数之后临时扩展。

近首full observer23649与原join49172均真实结束，终端42a073/fba831 EXIT0已消费。其唯一远端观测2026-10-14T00:43:10.673065+08:00：原parent/controller仍live、parent RUNNING，第二add_delta真实prepare/M0 command EXIT0后已进入fresh50 PID673850，实际CUDA0,1，看到8/50。这里只用观测时刻的8轮计数，不报告或依据第二项中途分数选择模型；第三项此节点尚未执行。

依据第一full40分11秒/strict45秒及第二原train开始2026-10-14T00:35:48.576899+08:00，已安排唯一整组三端近结束观察native73726/localPID49896，deadline 2026-10-14T02:02:48.576899+08:00（第二start+87分钟，涵盖剩两端及prepare/M0、约3分钟余量）。此前不查询进度，观察超时不能当训练失败，也不能重提NN。仅26物理GPU0/1；无T/P、25/2/3、环境修改或preemption。

三端完整50/firststrict全部通过后，才运行已准备的一次CPU保存距离报告，再按§945 exact seal做每端一次G/C/F与因子/reference/delta诊断。现在这两种报告调用均0。必须据完整匹配对照与实际失败诊断决定下一步，不因第一项小正值扩车辆/seed/gain/rank/margin。

所有best与M0当前验收依赖保留，本次无权重退休；三端strict/report/诊断结束后，先D盘核SHA归档再清理无后续消费者的重复M0。三数据集全面超过SignalGoalACTIVE_UNMET，初始正向数值不改写成主模块或科学机制成功。
