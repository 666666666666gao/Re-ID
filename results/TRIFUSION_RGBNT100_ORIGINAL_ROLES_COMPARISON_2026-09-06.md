# RGBNT100 原完整三角色主比较

更新2026-09-06T14:15:52.731518+08:00。**RUNNING；没有完整三fold检索终态。**
实际启动2026-09-06T14:13:43.910354+08:00，执行commit bbe49e1c24956e891afb8ec3e83df01acd75c579，
wrapper93313/child93317。启动观察2026-09-06T14:13:45.758912+08:00确认两个进程存在；
这只是启动检查，未报告完成epoch或检索数字。

## 固定比较

[实验合同](../refine-logs/rgbnt100_trifusion_v1/EXPERIMENT_PLAN.md)和
[配置](../configs/RGBNT100/TriFusion-source-oof-v1.json)保持M0原SHA：
8d3ce84068a584ca732e62ac4dac2dca366119863570ca6f8d4af6fb3d3f2ee2。
三fold各20epoch、seed42、B64/K8，fresh角色与M0初始state SHA相同，
只加载对应fold的已完成epoch30 Signal，不加载任何M0训练权重。
没有额外模块、新损失、消融或已封存科学失败重训。

[Signal R2内部基线](TRIFUSION_RGBNT100_SIGNAL_R2_SOURCE_BASELINE_2026-09-06.md)为
89.5241750420 mAP/96.8299711816 Rank-1；全部3权重/数组/8675query/7794步已核验。
[M0](TRIFUSION_RGBNT100_ORIGINAL_ROLES_M0_2026-09-06.md)全部124更新及文件/标量已通过，
excess ratio0.00155176796144928；它只满足工程前置，不代表新身份检索有效。

| 输出 | 维度 | 本次完整终态mAP/Rank-1 |
|---|---|---|
| Signal baseline_only | 3072 | 待全部features/distance与B0逐元素复现；既有B0见上 |
| full CNN | 4608 | 尚未产生 |
| full Transformer | 4608 | 尚未产生 |
| full Mamba | 4608 | 尚未产生 |
| fused | 7680 | 尚未产生 |

每fold最终一次完整gallery前向，合计8675内部query/gallery、50身份、43375 query-output；
保存3权重/15特征距离数组/全部排序、AP/CMC/源采样与203项梯度逐步记录。
只删同身份且同camera，其他同camera身份仍为负例；不跨fold计算距离。
同一固定checkpoint五输出比较，保持原五科学门：
fused gain>=1pp、每fold非负、每fullbranch>=Signal、50身份bootstrap下界>0、fused严格最好。
bootstrap固定seed42/10000/2.5%linear、身份重抽且query加权；不选中间checkpoint。

## 时间与核验安排

实测M0容量1.2083–1.3325秒/更新，正式约5190更新（实际以完整sampler为准），
估计总105–125分钟，当前预计完成2026-09-06T15:58:43.910354+08:00至2026-09-06T16:18:43.910354+08:00。
首个阶段观察计划2026-09-06T14:43:43.910354+08:00，接近首折预计35分钟终点；
启动后另按>=180秒确认真实训练活动，后续按预估里程碑检查，不逐epoch轮询。
预计时间不是完成承诺；任何停止均保留原日志、exit及中间产物。

终态沿已登记的远端文件/CPU数组核验器及本地全量JSON/NumPy重放，
不以局部排名、少量身份或训练loss判断主方法成立。权重/张量/图像留远端。
独立审计当前因服务额度限制不可用，没有本实验独立verdict。

[启动回执](../evidence/trifusion_rgbnt100_original_roles_comparison_launch_20260906.json)；输出根 /root/trifusion-storage/artifacts/rgbnt100_trifusion_source_oof_v1_seed42_20260906/comparison。
官方测试/RGBNT201 dev为0。RGBNT201保留58.4050/59.3939、主目标未达；
MSVR310原完整比较、V23/V24负结果保持，不能用本内部结果取代官方成绩。

## 真实启动活动核对

2026-09-06T14:17:33.866656+08:00（启动后229.956302秒）：fold0完成epoch1/2，81+83=164更新；
第3epoch进行中，已完整写出189条更新，全部有效且203项梯度有限，AMP下降0。
GPU6260MiB/100%，两个进程存活；完整fold receipt和terminal仍未出现，未读取heldout指标。
[原始启动活动证据](../evidence/trifusion_rgbnt100_original_roles_comparison_startup_check_20260906.json)。训练loss不作为检索增益证据，阶段观察计划仍14:43:43。
