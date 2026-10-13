# 真实结构读出的增量用途：有限原始机制核查

2026-10-13。结论：两个已有机制可支持“干预实际读出、评价最终排序”这一环，尚不足以支持新的训练方案或预期提升。本次只读本地源码、论文及作者仓库，未执行作者程序、SSH、GPU推理或训练，未修改生产源码。

## 已有证据边界

指定的 `modeling/trifusion/roles.py` 不存在，实际路径为 `GlobalTokenRoles`→`FP32SlotCompetitionRoles`，继承的结构读出在 `correspondence_evidence_readout.py:65`。`region_use_objective.py:6` 将实际区域分成九个角色/模态组，在各组16槽位上做对称MaxSim；辅助目标绕过readout，普通fused任务仍监督readout。实际部署先把16槽位按固定四个2×2组平均，拼接三个角色，再用既有权重生成3模态×4组×128维修正。集合目标可忽略槽位排列，部署保留组归属；这是源码差异，尚非失败原因。[实际读出](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/correspondence_evidence_readout.py:65)；[现有目标边界](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/evidence_learning_gap_20261013_920/CURRENT_OBJECTIVE_BOUNDARY.md)。

已保存报告中，R/J未胜同模型G；梯度排除比随机排除低0.105086个百分点mAP。不能因此确认压缩、错配或重复global；也不能把correction独立低分当作无用途。[路径核查](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/evidence_information_paths_20261013_921/README.md)；[原结果](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/global_conditioned_evidence_best930/RESULT_REPORT.md)。

## 至多两个已有机制

**CAL，ICCV2021**：固定feature map，替换attention，经同一池化/分类器计算真实与反事实输出差，再监督差值。作者ReID源码训练时用U(0,2)假attention，实际损失为CE(real)+triplet(real feature)+CE(real−fake logits)，没有切断fake路径梯度。原任务使用身份标签，不用区域GT；ResNet50、8头、batch80、160轮；额外池化/head计算，依赖PyTorch1.4/CUDA10.1。作者仓库当前固定提交为2023年版本，并非论文时快照。[论文§3.3、§4.2](https://arxiv.org/html/2108.08728v2)；[BAP与输出，0ba9d5084f2532eeb21c9ef051c23f8b339595ff](https://github.com/raoyongming/CAL/blob/0ba9d5084f2532eeb21c9ef051c23f8b339595ff/reid/modeling/baseline.py#L76)；[实际loss](https://github.com/raoyongming/CAL/blob/0ba9d5084f2532eeb21c9ef051c23f8b339595ff/reid/layers/__init__.py#L18)；[配置](https://github.com/raoyongming/CAL/blob/0ba9d5084f2532eeb21c9ef051c23f8b339595ff/reid/configs/softmax_triplet.yml)。

可借用受控干预原则；直接复制差值CE应排除：它没有监督本项目1536余弦排序，差值变大还可能来自fake预测变坏。该提交eval返回分类logits，亦不能称为原样保留本项目部署特征。

**FastAP，CVPR2019**：对单位embedding的`2−2·dot`建立软直方图，优化batch内所有正例的近似AP。原任务为SOP/In-Shop/VehicleID的身份检索，ImageNet预训练ResNet；除身份标签，还利用高层类别信息组织难batch；作者主要使用单张12GB Titan X Pascal。它只提供最终比较目标的对齐，不能保证证据非冗余。[论文§3–5](https://people.bu.edu/bkulis/pubs/CVPR2019FastAP.pdf)；[具体loss，ca85b2ab3f22460795d4306c8fa01c655d17027e](https://github.com/kunhe/FastAP-metric-learning/blob/ca85b2ab3f22460795d4306c8fa01c655d17027e/pytorch/FastAP_loss.py#L36)；[作者复现/资源说明](https://github.com/kunhe/FastAP-metric-learning/blob/ca85b2ab3f22460795d4306c8fa01c655d17027e/matlab/README.md)。

PyTorch移植未由作者复现论文结果；该pin的MATLAB文件还存在未定义`N_neg`引用，不能当作已验证可运行模块。[作者声明](https://github.com/kunhe/FastAP-metric-learning/blob/ca85b2ab3f22460795d4306c8fa01c655d17027e/README.md#L43)；[具体引用](https://github.com/kunhe/FastAP-metric-learning/blob/ca85b2ab3f22460795d4306c8fa01c655d17027e/matlab/FastAP.m#L61)。

## 最小可证伪检验；均未执行

1. 只复用固定best的g、实际区域、readout权重及gain，先重现原f分数。预先固定组内交换与跨组交换：两者MaxSim应不变；组内交换readout应不变，跨组交换可变。记录真实余弦/AP改变，不能把跨组扰动损伤称为自然错配证据。
2. 在每个角色/模态组把16槽位替换为该组均值，走同一readout；将对照修正逐样本匹配真实修正范数，保留gain。比较真实F、该对照F、G的全正例AP、身份宏AP、修复和损伤。真实排列未胜均值对照且未胜G，即不支持本读出需要这些槽位细节。只检验最终读出，不能代表上游桥接或完整分支删除。
3. 若讨论后续训练，FastAP必须落在实际`normalize(g+gain·c)`及原合法关系上；单用它仍是已有排序学习。`L_AP(f)−stopgrad(L_AP(g))`梯度完全等于前项，不能改名增量目标。CAL式差值监督亦须独立证明自然F净收益，不能只证明对照更坏。

该检验使用已有身份关系作为评价标签，不把身份、相机或区域GT输入部署读出。固定global及gain，维持1536维；若实际区域缓存不在本地，本次只保留检验定义，不能写成已完成测量。即使检验阳性，也只说明该固定模型的读出需要区域排列，不能证明新增信息独立于global，更不能代替未见身份上的净收益。

以上检验不同于重复MaxSim/拼接、普通局部ID、repair/keep、自由anchor重建、末端router或调大残差：唯一新问题是相同集合均值与修正能量下，固定结构读出是否利用槽位细节。两来源均未解决“学习global缺少的身份证据”全部问题；当前排除直接移植训练，不增seed、容量、batch或参数搜索，不保证提升。
