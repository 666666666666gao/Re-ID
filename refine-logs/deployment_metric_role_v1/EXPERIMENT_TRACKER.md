# 当前执行登记

更新：2026-10-05 §41.839。依据02:58:48实际收取：正式1/6；M0终态2/6。339项scientific source与187项封存控制不变。

| 数据集 | 条件 | 实际阶段 | 状态/结果 |
|---|---|---|---|
| RGBNT201 | semantic | prepare、真实8步M0、fresh50/2649、首次strict完成 | E8：74.9363 mAP / 78.7081 R1；两项配对推进均未通过 |
| RGBNT201 | native | manifest M0 complete/exit0；fresh50已开始 | train3741795从02:58:05运行；内部M0诊断尚未单独收取 |
| MSVR310 | semantic | 原顺序排队 | 待执行 |
| MSVR310 | native | 原顺序排队 | 待执行 |
| RGBNT100 | semantic | 原顺序排队 | 待执行 |
| RGBNT100 | native | 原顺序排队 | 待执行 |

首semantic相对匹配raw-role控制ΔmAP +0.5614、ΔR1 −0.1196；相对独立global-only为+0.6396/−0.2392。预登记+0.5mAP且R1不下降，两项均未通过。完整四项、50轮原始曲线和所有回执见主交接§41.839与 logs/deployment_metric_role_first_full839_20261005。

首端M0探针在自身完整50轮/第一次strict和SHA接受证书完成后退役358,375,480B；正式best及必要控制/初始化/作者权重保留。其他端不提前清理。

原supervisor3606472/startticks39300650；原66644观察器正常退出0。唯一新观察器session37914于03:01:01启动，首次03:36:44.855310、之后240秒。节点前不重复查询或启动。原唯一15对CPU报告仍在全部六端闭合后执行一次。

仅26物理GPU0/1，不设置、查询或监控功率/温度；不使用2025或GPU2/3。2025原I/O pending。单seed42、已消费官方基准；Goal ACTIVE / UNMET。
