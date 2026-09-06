# RGBNT100 全50身份角色主训练终态（2026-09-06）

核验时间：2026-09-06T20:38:12.983144+08:00。固定20epoch完整结束，状态COMPLETE_FULL50_ROLES_MAIN_EXECUTOR_VERIFIED_FIXED_EPOCH20。
实际执行2800c88875736a679d3713dbdd6ee4dcefec73a0，19:42:58.070783至20:31:30.501081；
训练及远端CPU核验均exit0，总2912.427695秒。官方检索尚未运行，训练完成不等于检索有效。

- 2625次有效更新，168000记录曝光；全部8675训练记录及50身份都已覆盖。
- 203/203可训练张量有梯度，全部逐步梯度有限，AMP下降0。
- 20轮完整学习率、PK索引、FP32损失重组都通过全量核对，末epoch均值0.531187150063。
- 总参数97519117，可训练6692364；峰值allocated5709.758789/reserved5992MiB。
- fresh初态与M0逐项一致；没有使用M0已训练权重或旧OOF角色权重。
- 冻结参数及内嵌Signal全状态保持，最终权重严格重载后的五输出一致。
- 16个clean角色source前向、8个独立Signal source前向完成原一致性检查。
- 官方模型前向0，RGBNT201 dev前向0；当前未产生本次官方检索分数。

唯一保留角色权重：/root/autodl-tmp/trifusion-v2/artifacts/rgbnt100_trifusion_main_v1_seed42_20260906/roles_main/roles_epoch20.pth，390494154字节，
SHA256 3946c4ede79e6f622819b10c183906e25526796165b0b82560a7d4ebed6605e1。
summary SHA256 908397c508e1052735153d4dfa5e2bc96091ebc947cc33eb7981661797736ab9；
远端完整核验 SHA256 a367158e7571ef073714b15f9a761d176343c19a444627ea52e959fc129d2f66。
所有12个文本/JSON/日志共103421043字节已完整取回并逐文件SHA核对；
本地仅使用JSON/标准库/NumPy核对全部2625标量更新，没有加载权重或图像。
独立审计服务不可用，此项是完整执行者核验，不写成独立审计通过。

同身份正样本对588000，其中cross-camera506518（86.14251701%）；
这是本次实际采样描述，不构成泛化或收益的因果证据。

官方比较已绑定真实Signal epoch30与本次角色epoch20的summary/核验/权重。
配置configs/RGBNT100/Official-main-v1.json，SHA256 96baf923bc5f92b56feb490342d193031a522cfa50a626156bd2eb2e0d732985；
39个源码文件绑定，所有1715 query和8575 gallery、50测试身份、五输出。
固定增益条件沿原计划：fused至少+1pp、三完整分支均不低于Signal、
50身份bootstrap下界>0、fused mAP严格最高；不含三fold门。
本登记状态READY_NOT_RUN，待发布并同步后只启动一次固定完整比较。
不按官方结果选择epoch、seed或融合权重；SOTA与同源Signal增益分开判断。

20:39:40磁盘复查：数据卷free25914597376字节（24.135GiB）；
系统卷free11129262080（10.365GiB）。
原24个删除路径都不存在，12个独立模型存在且大小一致，
当前CLIP、B0、M0与epoch20权重保留。无额外删除。
清理时全部保留模型前后完整SHA见原回执，本轮磁盘复查仅验证存在和大小。
