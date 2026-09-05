# V18 PVNP：source配对视角变化投影，固定主实验

状态：执行前冻结。V17失败保持不变；本实验不使用其失败包络损失，不扫描原模型。

## 证据与单一假设

完整V17诊断覆盖3126 gallery记录、571 query、全部六个终态。CNN新增14个Rank1
错误而修复6个；9/14新增错误的最近负例与query同相机，fused为4/5。全部CNN
query最近负例同相机比例从62.17%升到64.62%，但这只是相关证据，不证明相机是
唯一原因。DTRED的CNN correction范数更小(0.2885 vs0.3351)，却使最近负例距离
缩短0.00455，排除“只要修正更小就更安全”的解释。

固定四分区正例最佳对应97.62%为同一区域；直接保留四分区原特征的完整gallery
分支mAP72.7453，低于冻结CNN投影头79.3199，因此不直接换分区头或做区域搜索。
图像显示跨相机的亮度、视角变化及部分模态遮挡，需要训练内证据验证新表征。

新假设：同身份跨相机差分的主要方向包含可去除的视角/成像变化；从三专家各自
特征中移除该source估计方向，再学习同一个低秩修正头，可以改善未见身份排序。
该方向也可能包含有用身份线索，这正是完整对照实验要检验的风险。这里不是已
证明的camera因果分解，也不声称发明了通用新度量学习理论。

## 网络与source拟合

- 继续使用V12三个完全身份隔离的Signal+V8 checkpoint；完整3072D Signal前缀、
  camera SIE、CNN/Transformer/Mamba冻结来源不变。
- 每折只遍历94个source训练身份，按身份和相机分别平均各专家1536D残差。
  对同一身份的不同相机均值作差，将差分矩阵的第一右奇异向量作为该专家唯一
  投影方向。固定rank1，不用heldout身份拟合，不扫描秩、阈值或相机组合。
- fold0的source有14个cam0/1跨相机身份、没有cam2/3配对；其他折为13个cam0/1
  加1个cam2/3配对。因而不采用需要每个相机均有source配对的相机专属偏置表。
  本投影不读取推理相机来选方向，所有相机应用同一个source方向，无缺相机fallback。
- 每专家执行`normalize(x - dot(x,n)*n)`，在低秩修正前、后各一次。最终维度和
  三专家等能量拼接不变，Signal-only仍可从同一checkpoint输出。
- 对照`uncentered`保留同一结构/参数/方向buffer，但显式关闭投影；实验端`projected`
  开启投影。两端无DTRED envelope，仅原ID/triplet和Signal保护项；新模块无额外
  可训练参数。它改变的是source拟合的检索子空间，不是V17的loss/width扫描。

## 固定训练与完整评价

1. 所有source拟合回执先落盘；训练/heldout身份集合严格不相交。
2. M0：真实B64/K8、8步容量、100步固定batch过拟合；22/22训练tensor有梯度、
   overflow0、冻结来源/投影buffer不变；去除label-smoothing下界后的loss ratio<=0.1。
   两端初始权重和前8个增强batch逐字节配对、Signal前缀精确一致。
3. M0通过后完整三折x两端x20epoch，seed42、AdamW lr0.00035/wd0.0001、AMP256、
   无中间检索/epoch选择。每端只有最终checkpoint；模型重载必须与最终state SHA相等。
4. 每折所有47个heldout身份留在gallery，共3126条；全部571个合法query参与；
   无跨相机正例的记录仅不进入query分母。统一报告五路mAP/Rank1/5/10、逐query
   AP/rank、每折和总汇总，不能选择有利子集。
5. Q1晋级条件在结果之前固定：fused aggregate相对uncentered >=+1.0mAP；三个fold
   fused增益均>=0；三专家各自aggregate增益均>=0；identity-cluster bootstrap10000次
   (seed42)fused增益95%下界>0；projected fused优于同checkpoint baseline及三个分支。
6. 任一科学条件失败，完整结果封存，不改rank/epoch/LR/方向估计后重跑。不进入D1。
   通过后执行一次all141-fit重拟合和完整20epoch主训练，最终30-dev仅访问一次，
   要求>=65mAP且严格超过Signal、V8 Phase-B和三个分支，再冻结官方全训练/比较。

没有测试时训练、rerank、query/gallery相互拟合、额外mask模型或GT推理选择。
已消费的21个fit留出身份仅作开发资格，不称独立新验证。SOTA总目标仍未达到。
