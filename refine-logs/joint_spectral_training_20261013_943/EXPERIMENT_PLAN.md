# §943：跨光谱变化读出的真实接入与有限三条件队列

2026-10-13。用户“继续训练”授权持续有效。这里只执行RGBNT201、seed42、作者配方/raw global-role职责目标、每端fresh50；不增加车辆/seed/rank/gain/margin搜索，目标仍为三数据集全面超过Signal。

条件顺序joint_delta、add_delta、joint_absolute。三者使用同一589836参数、相同初始化；现有16槽位在角色/模态内部池化，沿光谱用加法或乘积，再经相同dense1536读出。差分参考为可微同语义读取，绕开CNN/T/M和桥接，不是冻结教师或identity保底。LMF因子机制、V8匹配参考及V17已有Hadamard先例依§942核读；不声称新颖性已成立。

新增完整模型只编码一次CLIP，保留原C/T/M算子和桥接；processed和reference的CLIP stages/context均detach。global与作者头仍沿原global任务更新，角色沿停止参数梯度的当前作者头/raw Triplet训练；没有新头或损失。新增模块初始化用fork_rng，保持原共享/head/RNG，prepare实际核对原shared/visual/camera/cfg/head初始化，三条件整个model state SHA相同。

每端顺序prepare→真实8步M0→fresh50→第一次strict evaluation。M0检查全部trainable张量非零有限梯度、BN仅8次更新、优化器覆盖恰好一次、参数/visual/camera真实变化及strict重载。新两个因子参数还逐步记录梯度及最终参数变化；首个M0 forward上的role loss对shared global梯度应None，对参考descriptor在delta两条件应有限非零，在absolute控制应None。失败不跳过/篡改旧失败/调阈值，保留原退出和日志；不因观察超时重启。

正式同一mAP-best的mAP/R1/R5/R10；primary joint_delta−add_delta，secondary joint_delta−joint_absolute，不用三条件声称2×2协同。推进线至少0.5mAP且R1不降，与全面超过Signal目标区分。原raw semantic及独立global对照和完整Signal只在各自匹配范围比较；共同dense读出/槽位池化不能全部归因乘积。

仅26物理GPU0、1，一个原始两卡进程顺序运行，无25/2/3、T/P查询、preemption或环境修改。首次memory检查01各18MiB；磁盘free5276340224B。三端预算4831838208B，包含2GiB保留+六个384MiB权重上限+256MiB文字/距离；每阶段沿旧2GiB门检查。预计约2–3小时，按真实M0/首轮校正，近ETA240秒监测。每端只保留mAP-best，M0依赖暂留strict/report；其后先D盘归档核SHA才退休远端重复文件，唯一best/源码/数据集和证据不删除。

首次温环境CPU import/configuration4667a7 EXIT0，确认builder/factory/GlobalTaskRoleHeads/raw loss/M0 observer/schema接线，0构造/0NN/0更新；不能代替真实初始化/M0。代码复核为root依code-review-excellence技能完成，没有独立代理或科学效力结论。当前正式新分数为0，GoalACTIVE_UNMET。
