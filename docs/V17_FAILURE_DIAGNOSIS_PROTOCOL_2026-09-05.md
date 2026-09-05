# V17 完整错误诊断：执行前固定范围

本诊断接续已失败的V17，读取全部三折、两端的最终checkpoint，不更新任何参数。
使用上一轮完整gallery的全部3126记录、571合法query，不访问30-dev或官方测试。
它是已消费fit数据上的开发诊断，不是新独立验证，也不改变原Q1失败判定。

执行前固定采集：

- 五路输出的全部query AP、first-hit rank、前五个合法gallery、最近正例/负例、
  最近距离差，以及全部正例/负例的平均距离。
- 三专家每条记录的原始correction向量范数、teacher与corrected cosine、三模态
  能量占比。不得用相关性直接声称因果，也不得据此扫描V17参数。
- CNN实际head的LayerNorm输出，按现有四个水平分区平均成3x4x768特征；这与
  当前CNN head一致，不尝试其他分区、分辨率、层或token选择。
- 全部六端均须与上一轮完整gallery逐query AP和rank精确一致，checkpoint和
  整模型state SHA与原训练终态一致，前后状态不变。

固定汇总与图像选择：

1. 全部query、每折和每身份均纳入比较。分别统计最近正例、最近负例和margin
   的变化，以及Rank1修复/破坏；同时报告CNN、Transformer、Mamba和fused。
2. 图像检查每折CNN AP损失最大、CNN AP增益最大、fused AP损失最大的query，
   并列取gallery原顺序首项；保留重复病例，不手工换成视觉上更有说服力的样本。
   每个病例显示query、weight0最近正/负例和DTRED最近正/负例的RGB/NI/TI三模态。
3. 对全部query记录CNN四分区的正负相似度及最佳分区对应是否偏离对角线；
   使用相同模态中的cosine矩阵，不据此部署GT选分区、rerank或测试时更新。
4. 检查结果后才能确定新的结构假设。新主实验另行固定完整训练终点、完整gallery
   和全分支比较，不把V17的只读诊断或原始对照当成后继主实验。

代码：`tools/diagnose_v17_failure_geometry.py`。大特征cache仅保存在远端artifact目录；
轻量汇总与运行来源进入GitHub和交接文档。
