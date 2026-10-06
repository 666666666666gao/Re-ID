# MSVR310：完整三端曲线与成本

| 条件 | 最佳轮次 | mAP | R1 | R5 | R10 | 末轮mAP | best−末轮 |
|---|---:|---:|---:|---:|---:|---:|---:|
| global_only | 38 | 50.8388 | 68.6971 | 81.3875 | 85.9560 | 50.7016 | 0.1373 |
| masked | 47 | 53.2453 | 70.3892 | 82.0643 | 86.8020 | 53.1508 | 0.0944 |
| all_patch | 48 | 53.2706 | 70.7276 | 81.3875 | 87.3096 | 53.2279 | 0.0427 |

匹配的masked−all-patch为-0.0254 mAP、-0.3384 R1，本轮未达到原推进条件。两条SIM路径相对plain-global的改善同时包含交互层、分类头、维度和目标变化，不能计成选择机制的独立增量。

masked/all-patch训练CLI时间比为1.0927，差113.10秒。该次顺序运行中masked耗时更长；这不是重复隔离测速或一般硬件效率结论。表中时间分别记录loss-loop、loop与逐轮评价保存、完整训练CLI、首次严格重载。初始化、M0及历史失败成本另列。

三条完整轨迹共150轮，来自三个模型各一次训练，不是150个独立种子。两个SIM的mAP-best位于第47、48轮，best到末轮变化不足0.1 mAP；当前记录不呈现此前RGBNT100 joint-L2那种大幅后期退化，不能用旧退化概括这两个新端。

保留作者mask实现：未选token置零但不移除，模态内与模态间mask取并集，top-k不是最终保留总数。当前未记录真实mask覆盖率，不能宣称MSVR实际全选或据此定位负结果的唯一原因。

Threeclosed50epoch/706update arms,150epochrows. Originalfirststrict andsame mAP-best CMC only. No NN/SSH/scoring/report replay. Finalregisteredall9queryCPUreport pending. SIMpair matchedhead/capacity/3072D; globalcomparison mixeshead/capacity/width/objectives. Time single sequentialsharedserverrun, not isolatedrepeatedbenchmark; excludesinitial/M0/oldfailcost. No newseed, significance/equivalence/stability/SOTA orthree-datasetcompletionclaim.
