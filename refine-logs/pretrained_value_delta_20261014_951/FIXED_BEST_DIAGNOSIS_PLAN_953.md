# 两端正式终态后的实际value用途诊断

当前训练496 source不可热改；本工具独立于训练scope。只有原semantic/uniform两端各自fresh50、首次strict、原supervisor EXIT0，以及一次已保存距离的CPU配对报告全部完成后，才封输入并各执行一次固定best完整query/gallery编码。现在未执行诊断，没有新诊断分数。

保持原best/初始化/condition，匹配完整50历史的mAP-best和同权重CMC。真实模型state前后SHA一致、grad空、0optimizer；重新提取fused距离与首次official≤1e-5，同一作者camera评价和完整gallery。不更换epoch/seed，不用未验收分数。

保留G/C/H/F以及实际attention_change和region值产生的每角色1536维贡献。验证原c恰为三角色贡献均值，h=g+gain*c、f=L2(h)。读取变化统计记录L1、最大绝对值、行和绝对值、实际区域范数；按三角色/三模态分别报告。没有从attention差值编造实际权重熵，不用“图好看”证明选择有效。

同模型global/fused与独立训练global_only分别比较，诊断best选择是否改变了全局能力。C和三角色贡献的独立完整检索仅说明身份关系，不能代替算子必要性重训消融。记录修正/G能量比、G→F夹角、均值/中心化能量及修复/新增错误。描述数据集均值不用于test中心化或教师监督。

不保存大128×512 patch缓存：只保存G/C/H/F、每角色1536向量、按区域的读取统计和六个距离矩阵。预计每端约70MB特征与17MB距离，加少量JSON，两端预算256MiB。实际大小需由输出记录核对；入场需至少2GiB保留空间+256MiB预算+256MiB余量。M0二进制在原first strict/CPU报告关闭依赖前保留，未来原verifier要使用退休依赖则精确恢复D归档。

解释规则：读取变化弱且c弱，支持检查实际选择职责；读取变化明显或C独立强但F不涨，支持检查有用性/损伤；ownG退化则先分离全局和增量表现。以上均不是唯一因果定位，不自动触发gain、温度、margin、seed或熵尖锐度搜索。超越Signal完整主目标不缩小。

只有26物理0/1、原warm环境；任何观察超时不重启NN，原读取失败保留。单独记录两次固定best读取的PID/命令/输入SHA/原EXIT。先按已登记near-ETA原observer看训练结果，不因本计划再查询当前epoch。
