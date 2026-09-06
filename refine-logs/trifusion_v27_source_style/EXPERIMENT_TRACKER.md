# V27 tracker

IMPLEMENTED_REGISTERED_NOT_RUN。T0/M0/Q1均未运行；完整计划已固定。

2026-09-07T03:14:22.617591+08:00 R1 STARTUP_CONFIG_TYPE_FAIL：yaml将JSON指数值读为string，模型/更新0。R2仅按JSON读取，原科学合同不变，REGISTERED_NOT_RUN。

2026-09-07T03:28:24.943088+08:00 R2 c225652完整M0 PASS：48预检/116updates/ratio0.0133480123，原Signal与冻结状态不变，容量6418MiB。原126981进入完整Q1；03:25 fold0 control230steps，无终态。不改原合同。详见results/TRIFUSION_V27_M0_AND_Q1_LAUNCH_2026-09-07.md。

2026-09-07T03:31:23.537075+08:00 TERMINAL_VERIFY_QUEUED：128288存活，180秒等待原126981结束；03:30原Q1 fold0 control469持久步，overflow0。Q1仍RUNNING，科学判断待全部六端。

2026-09-07T03:44:09.621876+08:00 原126981与队列128288/129047均真实存活；03:42 fold0 control580/candidate460更新，overflow0。完整报告链已部署，要求六端及全数组核验终态；训练源/科学合同未变，无完整Q1结论。


2026-09-07T04:43:25.103022+08:00 COMPLETE_Q1_FAIL_4_OF_5: 六端3360更新/120epoch，训练/全数组核验/报告均exit0；fused+1.338944、三折及角色平均全正、bootstrap下界+0.170768，唯一失败为CNN高0.070498。31原始文本SHA与本地全量复算通过。保留正增益和原FAIL，不晋级或调参救援。下一步source-only完整关系支持诊断尚未登记/启动。详见完整终态分析与master41.85。
