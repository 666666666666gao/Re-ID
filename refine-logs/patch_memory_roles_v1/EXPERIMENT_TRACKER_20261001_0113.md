# 角色 Patch 读取实验执行表

2026-10-01 01:13 实测。Goal ACTIVE / UNMET。恢复 campaign 运行中，正式验收 4/6；原硬件故障 campaign 保持 FAILED。

| 数据集 | local_memory | full_memory |
|---|---|---|
| RGBNT201 | 完整50轮，严格重载通过；best E2：72.5442 mAP / 73.9234 R1 / 83.1340 R5 / 88.0383 R10 | 完整50轮，严格重载通过；best E2：72.5335 / 73.9234 / 82.6555 / 87.9187 |
| MSVR310 | 完整50轮，严格重载通过；best E10：52.3622 mAP / 67.8511 R1 | 完整50轮，严格重载通过；best E10：52.2919 / 67.8511 |
| RGBNT100 | GPU3 实际 trainer1054538，15/50，R(running)；未有正式终态 | GPU0 实际 trainer1062431，16/50，R(running)；未有正式终态 |

四端通过原 collect_patch_memory_roles.verify，48项 fused/global/joint_local 完整图库CPU复算的最大误差为2.7378210063489e-6个百分点。三数据集两条件的初始模型SHA和可训练参数相同；210个冻结源文件SHA未变。角色阶段每端50轮，逐轮官方fused mAP选同一best并严格重载，无M3或局部辅助身份头。

full−local mAP：201 −0.0106882；MSVR −0.0703241；两数据集R1差均为0。当前两对没有支持扩大读取范围带来收益。RGBNT100两端继续跑完，统一六端门判断仍待完整面板；不改阈值或挑种子挽救。

GPU1/2已完成本面板对应训练并释放，暂时空闲；GPU0/3仍100%利用率。空闲不表示设备再次故障。没有重启正常训练或加入未登记的填充任务。恢复controller1050794继续等待剩余两端。

原observer1080242按01:03、01:07、01:11检查，01:11 partial collector实际exit0并接受4/6。原字节证据17件+INTAKE保存于logs/patch_memory_recovery_intake_20261001_0111/，权重/距离数组留远端。

结束observer1119184于01:10:53实际启动，首查2026-10-01 02:35 CST，未结束时240秒检查同一campaign，完成后核对原accepted_matrix的6/6；只观察，不训练或自动重试。脚本OBSERVE_RECOVERY_FINAL_20261001.py SHA6003ce476260ef6be513b4097bb4f31ce04ca73e9cd72082464bc4bc9b6f4652。02:35仅为估计里程碑，不是完成承诺。

partial结果见results/PATCH_MEMORY_PARTIAL_2026-10-01.md；下一步先完成剩余两端和全六端机制分析，再按预登记门决定后继。研究目标尚未达到。
