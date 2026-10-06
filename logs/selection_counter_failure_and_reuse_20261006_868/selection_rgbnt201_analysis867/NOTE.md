# RGBNT201：已闭合三端轨迹与成本

| Arm | Best epoch | mAP | R1 | R5 | R10 | Last mAP | Best–last |
|---|---:|---:|---:|---:|---:|---:|---:|
| global_only | 27 | 73.4728 | 77.1531 | 85.8852 | 89.9522 | 71.8170 | 1.6558 |
| masked | 29 | 72.7854 | 75.4785 | 84.8086 | 90.6699 | 71.9256 | 0.8598 |
| all_patch | 7 | 72.5484 | 76.6746 | 86.0048 | 90.0718 | 69.9037 | 2.6447 |

1. 主要同容量配对masked−all_patch为+0.2370mAP、−1.1962R1，未达原推进门。两SIM臂mAP均低于无SIM global；后者还混有头/维度/容量/训练目标差，不是选择专属效应。
2. best到末轮分别回落1.6558、0.8598、2.6447mAP；记录提示后期回落，尚未定位唯一原因。图中两类指标的点都沿用mAP-best，未另选R1峰值。
3. 所有曲线是同一训练内的50轮，不是50个种子。无置信区间或跨数据集成功主张。时间分项来自实际日志，未混算首次构造、M0、旧失败和推理成本。

后续：继续既定MSVR310/RGBNT100六端与唯一九端全query报告；不据本三端调整topk、LR、seed或推进门。

All3closed50-epoch records,150epochs. No NN/SSH/scoring/report rerun; not newseed/stability/significance/SOTA. mAP-bestCMC unchanged. Mainmask-allpatch matched3072; globalcomparison mixescapacity/head/dimension/objectives. Loss totals have different headcounts. Timefields distinguishloop,loop+epoch-eval/save,trainCLI andfirststrict; no M0/prepare/oldfailedwork orinferencebenchmark included.
