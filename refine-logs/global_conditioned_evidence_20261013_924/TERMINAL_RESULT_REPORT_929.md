# 两端终态与身份比较净用途

仅RGBNT201/seed42；每端完整50轮，同一E8 mAP-best首次严格评价。

| 条件 | mAP | R1 | R5 | R10 |
|---|---:|---:|---:|---:|
| gradient | 74.364459 | 79.306221 | 88.397127 | 91.866028 |
| random | 74.469545 | 79.784691 | 88.157892 | 91.866028 |
| gradient−random | -0.105086 | -0.478470 | +0.239235 | +0.000000 |

主配对：修复5、新增错误9；query AP提高207、下降268；身份宏AP变化-0.074209点。

| 首位组（random→gradient） | query数 | 对总mAP变化的贡献 |
|---|---:|---:|
| repair | 5 | +0.036773 |
| new_error | 9 | -0.070844 |
| kept_correct | 658 | -0.039909 |
| kept_wrong | 164 | -0.031106 |

最差5% query AP变化均值-5.152224点，仅为固定best事后描述，不是训练种子不确定性。

gradient：2649记录更新，末轮mAP=71.372674；best→末轮下降2.991786。train producer墙钟2689.464858s，history.seconds之和1929.708647s仅训练循环。
random：2649记录更新，末轮mAP=71.367367；best→末轮下降3.102177。train producer墙钟2711.559642s，history.seconds之和1941.034883s仅训练循环。

两端batch顺序及支持/有效模态/排除计数一致；记录teacher标量最大绝对差：{'global_loss': 0.0, 'target': 0.0, 'gradient_norm_min': 0.0, 'gradient_norm_max': 0.0, 'gradient_norm_mean': 0.0}。这些标量不证明同模型global向量一致。

两个CPU报告各调用一次，对原保存距离及两个既有控制完成合法全query分析；主配对只组合已接受query AP/first-rank，无新排序或模型前向。源码审查与结果复核为同家族provisional，实际模型/effort未attest，不声称审查者重新加载了数据集或tensor排名。

当前机制没有胜过同数量随机排除，也未到完整Signal；不能作为已成立的有效主模块。不按官方分数搜索层号/比例/gain/margin/seed，不扩展车辆，不启动新训练。主Goal仍ACTIVE_UNMET。

计划中的同模型global/correction/fused分解、区域实际内容及隔离推理时间没有新增测量，作为明确缺项保留。不能把本次CPU行政闭合写成科学主张成立、泛化失败根因已定位或整份原计划所有诊断已完成。

论文主线：不追求每个分支更强，检验每条新增证据在最终身份比较中的净用途。梯度敏感性是模型代理信号，不是部件真值或因果重要性。
