# MSVR310 来源角色关系覆盖完整分析

本报告复用原 V8 initial/control_final/style_final 的已封存来源数组，覆盖三折、三种 query 视图及两种来源关系定义。共54条件、37152条 query-condition-protocol 成员；不同条件与折间存在重复，不能作为独立样本量。没有新模型前向、参数更新、权重或官方成绩。

以下合并三折，仍完整列出18个状态/视图/协议组合。identity 表示同身份且排除自身记录；cross_scene 的正例额外要求不同 scene，负例仍为所有不同身份。无合法正例 query 保留记录和图库作用。

| 状态 | query 视图 | 协议 | 合法成员 | 具有额外有效负关系 | 其中存在额外反序 | 子集 hinge 相同 | 子集 hinge 更低 |
|---|---|---|---:|---:|---:|---:|---:|
| initial | clean | identity_exclude_record | 2064 | 1260 | 609 | 1979 | 85 |
| initial | clean | cross_scene | 1200 | 743 | 492 | 1146 | 54 |
| initial | augmented | identity_exclude_record | 2064 | 1330 | 615 | 1969 | 95 |
| initial | augmented | cross_scene | 1200 | 770 | 491 | 1140 | 60 |
| initial | coupled_style | identity_exclude_record | 2064 | 1371 | 636 | 1943 | 121 |
| initial | coupled_style | cross_scene | 1200 | 800 | 498 | 1121 | 79 |
| control_final | clean | identity_exclude_record | 2064 | 1483 | 138 | 1917 | 147 |
| control_final | clean | cross_scene | 1200 | 893 | 130 | 1121 | 79 |
| control_final | augmented | identity_exclude_record | 2064 | 1462 | 177 | 1884 | 180 |
| control_final | augmented | cross_scene | 1200 | 886 | 163 | 1088 | 112 |
| control_final | coupled_style | identity_exclude_record | 2064 | 1546 | 257 | 1871 | 193 |
| control_final | coupled_style | cross_scene | 1200 | 918 | 217 | 1086 | 114 |
| style_final | clean | identity_exclude_record | 2064 | 1418 | 198 | 1926 | 138 |
| style_final | clean | cross_scene | 1200 | 851 | 187 | 1119 | 81 |
| style_final | augmented | identity_exclude_record | 2064 | 1442 | 208 | 1921 | 143 |
| style_final | augmented | cross_scene | 1200 | 877 | 193 | 1123 | 77 |
| style_final | coupled_style | identity_exclude_record | 2064 | 1489 | 267 | 1904 | 160 |
| style_final | coupled_style | cross_scene | 1200 | 895 | 239 | 1109 | 91 |

“额外”是相对 fused 当前单一最近负例而言，并非原完整候选池之外的新样本。“有效”表示该负例与同一 fused 最难正例形成0.3欧氏 hinge 违约；“反序”另指非正距离间隔。它们都是表示关系统计，不是参数梯度或泛化收益。

在 control_final/augmented/cross_scene 的1200个合法成员中，886个存在额外有效负关系，其中163个存在额外反序。角色提出的1324条额外负例曝光里，1320条 hinge 激活、200条反序。三角色独有最近负例曝光分别为 CNN 508、Transformer 588、Mamba 488；这些集合不能直接相加解释成独立身份数。

但是，角色子集在1088/1200成员上保留相同 fused hard hinge，在112/1200上降低，没有任何提高。所有54条件也都没有子集 hinge 高于全集。原因是对同一候选全集的子集，最远正例不会更远、最近负例不会更近；若继续只取 fused 极值，额外的非极值关系不会自动参与目标。相同标量也不自动证明并列极值的梯度相同。

结论是角色确实可提出不同且在 fused 空间仍未满足间隔的关系，但“只换成角色并集、保留原 hard 目标”不足以宣称新增监督。下一步若让这些关系进入学习，必须明确改变采样权重或目标聚合，并测量其实际参数梯度；不能把列表差异包装成模型改进。

这里的固定诊断视图需区分：coupled_style 在外层 eval 提取下强制启用已登记的 style 分支，不能称为普通无扰动 eval。

这些固定诊断视图与完整来源图库并不是实际训练历史队列；模型也不是刚结束的历史反传 Q1 终点。不得用本结果推断新目标在训练期必然活跃、提高未知身份检索或已满足下一阶段条件。

## 最接近的方法与边界

Multi-Similarity 将关系挖掘与软权重分开，并通过相似度梯度分析哪些样本对实际进入优化。这直接支持检查关系权重，而非只统计候选数量；直接引入其 loss 属于已有方法适配。[作者论文](https://arxiv.org/html/1904.06627)

Sampling Matters 分析采样对优化目标的影响，并提出按距离分布加权的负例采样。论文对极近负例的讨论不证明本项目标签含噪或 hardest mining 是失败原因；本项目已用未平方欧氏距离，也不能把改用该距离作为新修复。[作者论文](https://arxiv.org/html/1706.07567)

尚未登记或启动下一项训练。下一假设草案仅考虑让角色提议的额外负关系进入 fused 均值 hinge；保留原 fused 最难正例和其他训练条件，先完成数学/真实梯度检查。草案不是运行合同或收益结论。

聚合输入 SHA256：`ec83100a9c6ed6a27207a4ffbec0324308407809438ed7513fd6ea3a805ec790`。完整原始证据：`evidence/msvr310_role_relation_coverage_complete_20260908/`。


## 独立复核与边界

独立审计终态 WARN，same-family/provisional；请求路由gpt-6-astra/max，不声称后端身份独立认证。审计自行复算54条件、37152行、108份原数组，确定性全字段检查通过；完整语义结论和限制以[原始审计](../refine-logs/msvr310_role_relation_coverage_v1/EXPERIMENT_AUDIT.md)为准。审计时tracker仍保留RUNNING当前措辞，本次改为计算/审计均完成，历史时间观察保留。

Root另对全部18个合并组、324个计数与审计独立聚合比较，完全相同。原始输入和全量独立脚本/输出均保留在[证据目录](../evidence/msvr310_role_relation_coverage_complete_20260908/)。下一项[设计草案](../refine-logs/msvr310_role_relation_coverage_v1/NEXT_HYPOTHESIS_DRAFT.md)不构成M0/Q1启动。
