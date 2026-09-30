# 角色 Patch 读取实验执行表

2026-10-01 03:18。Goal ACTIVE / UNMET。恢复面板6/6完整验收，原硬件故障campaign保持FAILED。

| 数据集 | local9：best轮 / mAP / R1 | full128：best轮 / mAP / R1 | full−local mAP |
|---|---|---|---:|
| RGBNT201 | E2 / 72.544235 / 73.923445 | E2 / 72.533547 / 73.923445 | −0.010688 |
| RGBNT100 | E1 / 85.103030 / 95.102042 | E1 / 85.104798 / 95.102042 | +0.001768 |
| MSVR310 | E10 / 52.362244 / 67.851102 | E10 / 52.291919 / 67.851102 | −0.070324 |

每端完整50角色轮，逐轮官方fused mAP选同一best、严格重载全图库评价。300epoch / 20,416接受训练步 / 48M0步。原门FAIL；不启动门条件下的独立global-only或多seed，不改门或选点规则。

恢复controller1050794、结束observer1119184均已结束；observer最后02:43:00实收6/6。原两项已完成训练另作评价，四项缺失端fresh full50，不伪造旧训练OS退出码。旧01:13快照保存在EXPERIMENT_TRACKER_20261001_0113.md。

RGBNT100全query/gallery CPU/GPU诊断02:47:45完成，实际wait退出0；其余201/MSVR诊断原COMPLETE与缺失wait边界保持。完整表与六条轨迹见results/PATCH_MEMORY_COMPLETE_2026-10-01.md。完整源210/210远端原字节一致；权重、probe、距离数组留远端。

Fresh gpt-6-astra/max、same-family/provisional审计总WARN，确定性工程/计分检查通过；原科学门失败。审计及全部trace见INTEGRITY_AUDIT_678_20261001.md/.json。本地27份源仅换行符不同，107份远端依赖未本地检出；没有因此改动任何已登记运行源。

后继槽位竞争干预尚未登记/实现/训练，不以诊断相似性推断唯一因果或保证提高。整体性能目标未达到。
