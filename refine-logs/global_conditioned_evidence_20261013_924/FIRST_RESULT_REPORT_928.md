# 第一项：完整50轮与首次严格评价

| 条件 | best轮 | mAP | R1 | R5 | R10 |
|---|---:|---:|---:|---:|---:|
| gradient | 8 | 74.3645 | 79.3062 | 88.3971 | 91.8660 |

相对raw_semantic：ΔmAP=-0.010402；ΔR1=+0.478469；首位修复7，新增3；身份宏AP变化-0.002518。

| 首位组 | query数 | 对总mAP变化的贡献 |
|---|---:|---:|
| repair | 7 | +0.053834 |
| new_error | 3 | -0.017256 |
| kept_correct | 656 | -0.035138 |
| kept_wrong | 170 | -0.011842 |

相对independent_global：ΔmAP=+0.067796；ΔR1=+0.358852；首位修复5，新增2；身份宏AP变化+0.064679。

| 首位组 | query数 | 对总mAP变化的贡献 |
|---|---:|---:|
| repair | 5 | +0.041317 |
| new_error | 2 | -0.012149 |
| kept_correct | 658 | +0.030897 |
| kept_wrong | 171 | +0.007731 |

50轮2649次更新。末轮mAP=71.372674，best→末轮下降2.991786。原训练producer墙钟2689.464858秒；history.seconds之和1929.708647秒仅为训练循环。

One original seed42 full50/first-strict endpoint, saved full legal camera-filtered ranking. Activity is model-derived sensitivity/support, not ground-truth region importance. Same-epoch independent history is a text metric comparison, not same-model global or feature equality. Final gradient/random comparison requires both endpoints. No model inference, test-time adaptation, causal proof, seed uncertainty or Signal superiority attested here.

原始12:37观察：random已完成1/50且原worker存活；下一唯一observer native51083/PID38216，due=2026-10-13T13:18:13.877004+08:00。未查询之后的进度，不能把这一快照写成当前轮数。

gradient单端结果不支持超过原raw semantic的mAP，也未达到完整Signal；同数量random仍待完成，尚无两端主配对。推理仍为单1536维/无mask/无测试标签。两类原best和必要来源/距离全部保留。
