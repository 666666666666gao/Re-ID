# Smooth-AP固定终态来源候选覆盖

状态：原提取/分析COMPLETE，执行器全量数值核验PASS；新上下文独立审计待进行。0模型更新、0heldout/official图像读取，不是Q1或官方结果。实际执行f7a0590，核验代码efb1d04，固定六个已封存Q1的epoch20模型。

完整12条件/8256来源记录前向；原提取04:18:37退出0、分析04:21:13退出0、逐行核验04:33:51完成。核验全部82560完整query记录、1996800批次anchor曝光、120输出/过滤汇总，AP重算最大误差0；特征直接差平方和与矩阵代数距离误差在1e-12内，排序使用已固定矩阵距离及稳定全局record序。全部159份文本13715377B接收并SHA核对。大候选明细/NPY留远端。

## 全部来源成员上的fused

| 视图 / 正例规则 | Control mAP | Smooth-AP mAP | 每端合法成员 |
|---|---:|---:|---:|
| clean / all-identity |99.105917|99.419141|2064|
| clean / cross-scene |97.185653|98.224923|1200|
| augmented / all-identity |98.591449|98.934077|2064|
| augmented / cross-scene |95.444495|96.672922|1200|

source成员在不同fold重复，不是独立新图像；每fold只在自身坐标计算距离，汇总AP而不混合特征。增强cross-scene来源fused增益+1.228427pp，Rank1 96.583333→97.916667；1200成员中251改善/70下降，其余不变；被负例严格超越的正例位置1214→1002。三个角色该来源条件的mAP也都改善：CNN+1.758941、Transformer+1.429036、Mamba+1.607211。全部40汇总和20配对输出保留，未挑角色/视图。

## 共同合法anchor上的候选/完整图库

| 端 / 视图 / 正例规则 | 候选mAP | 完整来源图库mAP | 共同合法曝光 |
|---|---:|---:|---:|
| control / augmented / all-identity |99.604749|99.198965|49872|
| smooth_ap / augmented / all-identity |99.682293|99.318497|49872|
| control / augmented / cross-scene |98.150400|96.402936|20496|
| smooth_ap / augmented / cross-scene |98.665998|97.384815|20496|

同一模型的候选和完整比较使用同一表示表，故该差异来自候选集合及其关系组成，而不是再次随机提取特征。候选池按record_index去重并排除query自身；cross-scene只去掉same-ID/same-scene，不改为负例。全部异身份干扰保留。每端49920原batch anchor曝光，cross-scene共同合法20496；168次曝光在全来源有跨scene正例、当前池无正例。all-identity池缺正例48次源于唯一记录定义，不能误说原训练K8没有正例：原训练保留同记录不同随机view位置。

本表按原采样的重复anchor曝光加权，前表按完整来源成员加权，所以同一端的两个完整图库mAP也不同。禁止跨表直接相减当成候选效应。clean和其他角色的全部条件见evidence/smooth_ap_source_coverage_analysis_20260909。

## 解释与下一步

Smooth-AP确实改善了这些固定来源视图的完整排序，而原身份外Q1仅+0.343455且低于Signal。来源收益不能替代泛化晋级。候选mAP较高也不能单独证明分数校准错误：正负数量、困难视角及权重随集合变化。当前结果不包含动态训练dropout历史重放或fused与其余13项的参数梯度分解。

梯度分解已经单独登记并启动六端8batch零更新预检，执行a2dec7f；04:38:12已5/6端完成，原63156/63157仍在运行。只有全部工程条件通过，才运行1560批完整来源测量。没有启动新的性能训练、扫描温度/权重或消费官方结果。

## 证据

原始文本与完整query行：evidence/smooth_ap_source_coverage_complete_20260909；全部条件/汇总/配对：evidence/smooth_ap_source_coverage_analysis_20260909。执行器验证脚本、固定计划及输入口径均保留。文本接收首尝试有生成代码括号SyntaxError，修正后仅重做接收，未重跑模型/分析；失败源码及记录在completion_support目录。
