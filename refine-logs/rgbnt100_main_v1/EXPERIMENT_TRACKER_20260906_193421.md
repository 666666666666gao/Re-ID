# RGBNT100 全训练集主比较跟踪

更新时间：2026-09-06T19:34:21.964629+08:00；Signal B0完整PASS，角色M0真实绑定就绪，尚未运行。

| 运行 | 固定范围 | 最新状态 |
|---|---|---|
| T0 | 全18965文件/原作者标签/1715完整掩码 | COMPLETE；官方文件头已读，官方模型评估0 |
| Signal M0 | 全50类首epoch/严格重载 | COMPLETE PASS；130有效更新，195梯度/AMP0 |
| Signal B0 | fresh同初态固定30epoch | COMPLETE PASS；18:42:15→19:16:19，3936有效更新/251904曝光，全8675记录/50身份覆盖 |
| 角色M0 | 固定full50 Signal上的fresh8容量+另一fresh100过拟合 | READY_NOT_RUN；配置68b091dd，真实B0权重/summary/核验已绑定，远端全部36源文件SHA通过 |
| 角色主训练 | M0通过后fresh同初态20epoch | NOT_RUN；不加载M0已训练权重，不使用旧OOF训练权重 |
| 官方完整比较 | 两固定终点，1715query/8575gallery/五输出 | NOT_RUN；配置等待真实角色epoch20完整核验 |
| 原内部OOF | 5195更新/8675query，fused相对Signal+1.792365pp | COMPLETE PASS；不再训练 |

Signal B0执行545186daf4edd3bf3aabf1a905adbc5a3359d03f，全部30作者LR/3936 FP32 loss和PK索引核验。
195梯度完整，0 AMP下降；全部8675独立source记录和50身份均见。
初始state9eb41bcd8ab10438ed6f74997f9821f19406d6cbbccabf084367f4ea8400d584与M0相同；
final state3989c1332e8cb566b11cf0350462e4a42b1ff0b74466b3f7b90e45cd93fe1e1a。
checkpoint f173efd1eb43193b4012b6165be451161b31163684a65759bfdaa7085b240bee；
summary7c82f15a1227e36b04b41091bcf57ebc8a23375682448d62cd263a58120ccc13；
远端核验5d1697e37035dfe17c0ad02268b0b8da5ee23801e9f071664ba7c8db39669a94。
12个原始文本/JSON共14283116字节完整取回逐文件SHA一致，权重/张量/原图仅远端。
首130步与原M0整个step JSON逐条相同，是描述性检查，不增加科学门。

角色配置configs/RGBNT100/TriFusion-main-v1.json已冻结原模型/损失/优化设置。
七个50类head、203梯度、固定100步过拟合excess ratio<=0.1在实际M0检验；不以准备就绪代替通过。
官方评估仍0模型前向/0指标，固定主计划SHA57f9cb601b1a5697acaff0e8bed3a737eb499d1f1a752ad6a29691c9afb503f3不变。
19:31现场数据盘free26829139968字节约24.99GiB，系统盘free11129262080约10.36GiB；
已删24个重复续训权重仍不存在，旧12个独立模型和当前M0/B0均保留。
独立审计服务不可用；本轮完整核验来自执行者，不能写作独立审计通过。
