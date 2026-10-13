# 角色分类入口的最小独立核查

2026-10-13，§41.936。当前训练停止；本项只登记源码与CPU合成head检查。真实初始化、M0、正式50轮与车辆扩展均未获本次启动确认，不自动执行。

## 实验问题及边界

§934发现的训练BN分类／raw Euclidean Triplet与L2检索的几何差异，实际上在2026-10-05的deployment_metric_role_v1计划中已经提出。本项不是新发现或论文创新。§935确认末端Mamba LN bias可产生一个条件task-flat参数子空间，但现有bias的直接修正极小，不支持通过删除bias解释或修复失分。

封存raw职责模型中，角色分类和Triplet均使用h=sg(g)+gain*c；joint-L2旧对照只改变Triplet输入。本候选只改角色分类：L2(h)先进入现有训练模式BN，再使用现有classifier的停止参数梯度值；角色Triplet仍读取raw h。global作者任务继续使用raw g。BN运行统计仍仅由global更新一次，角色调用沿用detached_head的克隆buffer；分类头仍不是独立角色头。推理保持L2(g+gain*c)，无新参数、头、损失、采样、角色、选择、原生CNN、文本、路由或检索表示。

BN位于L2之后，因此这不是直接余弦分类，也不是NormFace完整移植，不能称为训练／推理完全同几何或全部盲区已消除。外部BNNeck、NormFace只解释训练空间分工的一环；相关来源及适用边界见§934。训练入口改变仍会改变梯度与优化尺度，不能归因于纯粹“消除平移”。

旧F2/F3使用无角色的current基础配方；joint-L2使用raw分类、L2度量；独立角色头三集均未达到推进线。本项不把这些旧路线重新包装为未尝试，也不重新训练旧控制。新增实现仅支持RGBNT201 direct/semantic，车辆输入尚未设计或验证。

## 本次唯一允许执行的检查

- warm已有conda CPU，CUDA_VISIBLE_DEVICES为空，2线程；无安装或环境重建。
- 实际调用原GlobalTaskRoleHeads与新增NormalizedRoleClassification，但evidence是明确的合成stub：4行1536维、2身份、确定性head权重；不加载CLIP/Mamba、图像、真实权重或检索缓存。
- 两种wrapper各对固定shift=0与shift=[0,2,0,...]运行一次训练head forward与一次eval forward。标签[0,0,1,1]、label smoothing=0.1；原作者TripletLoss只forward核查raw特征与困难距离。
- 角色CE和global CE分别计算向量—雅可比积：角色只达correction/gain，global只达shared/head；不调用optimizer，不更新参数，不保存权重。
- 对比state键／形状、参数量、raw metric、部署向量、global logits与Triplet；验证原global BN计数为1，角色克隆buffer不持久更新，eval不更新buffer。
- 固定FP32判据：raw共同偏移logits／CE／Triplet差<1e-5；候选logits差>0.01且CE差>0.001；raw共同偏移CE梯度<1e-5、候选>0.001。这是预设的有限反例检查，不证明生产精度、全部方向、未来收益或梯度稳定性。
- 记录实际执行、退出、源码哈希及所有失败；不原样重跑碰运气，不事后按输出调整判据。组件失败先保留实际证据再判断原因。

## 后续决策

合成检查通过，只允许称接口／选定数值见证通过，不是生产源码审查、真实初始化、8步M0或检索成功。训练范围问题已单独向用户提出，未回复不启动依赖工作。

若明确恢复一项训练，先对当前新增入口作源码与真实初始化/M0资格检查，再做一个fresh公共CLIP、匹配作者配方的201 semantic/seed42完整50轮；所有旧控制保持。主要比较原raw semantic（74.3749 mAP/78.8278 R1）、独立global-only（74.2967/78.9474），同时列完整Signal（80.3029/85.1675）。原项目推进线仍为相对匹配控制mAP至少+0.5点且R1不降，不是显著性证明或Signal目标完成。远未接近前不多seed、调margin/gain/层数或车辆展开。

若后续无增益，否定此具体角色分类入口的效用，不泛称L2分类、全部空间对齐或所有证据学习无效；若只有局部判别性增加，仍不算互补收益。必须报告同模型G/C/F、query修复/新增错误、身份AP及完整50轮轨迹。三数据集超过Signal与原创机制仍未证明，Goal保持ACTIVE_UNMET。
