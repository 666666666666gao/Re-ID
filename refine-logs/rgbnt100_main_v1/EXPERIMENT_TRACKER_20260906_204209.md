# RGBNT100 全训练集主比较跟踪

更新时间：2026-09-06T20:42:09.976719+08:00。两端固定主训练均已完整核验；官方配置已绑定真实终点，尚未启动。

| 运行 | 固定范围 | 最新状态 |
|---|---|---|
| T0 | 全18965文件、原作者标签、1715完整掩码 | COMPLETE |
| Signal M0 | 全50类完整首epoch | COMPLETE PASS；130更新 |
| Signal B0 | fresh固定30epoch | COMPLETE PASS；3936更新，195梯度，AMP0 |
| 角色M0 | fresh8容量+另一fresh100固定batch | COMPLETE PASS；108更新，excess ratio0.0011232312 |
| 角色主训练 | fresh同M0初态，固定20epoch | COMPLETE PASS；20:31:30结束，2625更新，203梯度，AMP0 |
| 官方完整比较 | 固定两端，1715query/8575gallery，五输出 | READY_NOT_RUN；已绑定真实epoch30/epoch20 |
| 原内部OOF | 5195更新/8675query | COMPLETE SUPPORT PASS；fused相对Signal+1.792365pp |

主训练及全权重、全部2625 FP32/PK/梯度/LR行核验完成，12原始文本103421043字节已归档；
完整说明见results/TRIFUSION_RGBNT100_FULL50_ROLES_MAIN_2026-09-06.md。
fresh初态不变；Signal和冻结state不变；M0/OOF已训练权重未复用；本机模型/张量/图片调用0。
官方配置SHA 96baf923bc5f92b56feb490342d193031a522cfa50a626156bd2eb2e0d732985，原注册主计划和39源码绑定不变。
下一步发布同步后启动一次完整官方比较，预计10–20分钟，按固定阶段观察，不反复轮询或缩小图库。
官方四个Signal增益条件及SOTA资源边界见固定计划；现在官方模型前向/检索分数仍0。
20:39磁盘free24.14GiB/系统10.36GiB，24个重复快照清理保持，12个旧独立模型及当前权重保留。
独立审计服务不可用，执行者核验不能替代独立审计；RGBNT201和MSVR310主目标仍未达。
