# §944：第一条件真实M0通过，原三端控制器继续fresh50

用户“继续训练”已落实。原supervisor578010/controller578011只启动一次，26物理GPU0/1。三个条件joint_delta/add_delta/joint_absolute顺序执行；现在只有joint_delta通过prepare和真实8M0，进入fresh50；另外两个条件尚未执行prepare/M0/训练，不写成三端全通过。

注册Git `9bb45ad540a60975234efd5d2fe90b37647c03b1`，488source seal（原482未修改、扩6）。实际prepare与原raw semantic的shared/visual/camera/author/cfg/head等initializer核对通过。新trainable89604566参数、283张量，比原新增589836/2。跨条件完整initializer state相同要求仍待后两条件真实prepare核验；此时不能预签。

M0 2026-10-13T23:52:25.102572+08:00→2026-10-13T23:52:55.177880+08:00：8有效更新、283/283张量至少一次有限非零梯度、作者BN bottleneck计数8，strict reload最大差0；冻结参数不变，visual/camera真实变化，whole-wrapper optimizer恰好覆盖一次。首次实际role loss对shared_global无梯度，reference_descriptor有限非零。新增factor最终Δmax0.0001887530、rank_weights最终Δmax0.0001581907，两张量8次有限梯度和有效变化成立。它是工程路径活动，不是身份证据或增益证明。

M0探针SHA `549bd15f7631c954b39a4280857921f203c305243f14a2654fa2a2d29ecbabbe` 保留至formal strict/report依赖完成；不作为fresh50起点。原joint_delta train PID580021于2026-10-13T23:52:58.920710+08:00启动。单次预计M0节点观测2026-10-13T23:54:44.875862+08:00实际CUDA0,1、parent/child live，看到1/50。该1/50只是观测时刻，不代表当前即时轮次；不公布中途测试分数或据此救分。

唯一下一次近第一full ETA观察：local session23649/PID38852，deadline 2026-10-14T00:42:58.920710+08:00（原train开始后50分钟）；此前不查训练进度。预计三端约2–3小时，按该节点的实际用时校正。只消费原handle/原EXIT，观察超时不能当训练失败，禁止重复提交、preempt、安装或改warm环境；不查询/控制温度功率，不触及25/2/3。

已准备tools/report_joint_spectral_delta.py供三端首次strict全通过后一次CPU保存距离报告；当前调用0次，不是已有report结果。报告将再次核M0/50轮/同best/receipt/checkpoint/distances/source，比较joint_delta−add_delta、joint_delta−joint_absolute和各自相对原raw semantic/global。检查真实训练batch order一致、完整query修复/新增错误、身份AP分布及全轨迹，旧controls只用封存receipt/distances与当前固定batch SHA，不要求旧M0二进制重跑。

CPU报告source独立于正在冻结的488训练来源，不改变活动NN入口。root source review检查复用camera_scores/compare的receipt+distance SHA、query/gallery一致性、匹配指标与固定模型bootstrap边界；尚未实际执行，禁止在报告失败时反复重试挑通过。完整Signal仅作既有不同系统数值参照，原三条件只能证明两个条件效应，不能主张2×2协同或原创性已成立。

当前0/3正式端完成，没有新增完整检索成绩。原§941负结果和所有旧best保留，不扩车辆/seed/rank/gain/margin；本次清理权重0，M0是当前验收依赖。Goal仍为三数据集全面超过Signal，ACTIVE_UNMET。
