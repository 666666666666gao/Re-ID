# ROADMAP 与当前 Smooth-AP：条件式候选的原文/代码核查

本次为2026-09-20只读方法准备，未登记新训练，未改变当前Smooth-AP/完整梯度诊断。论文事实、作者代码和本项目推断分开。固定作者提交 fed37d75f475f636542b2ccd1ffc0c3918498a57；sources.json记录所有原始代码URL/字节SHA，LICENSE保留MIT声明。

ROADMAP原文§3.1–3.2分别提出SupAP排名近似与batch间分数校准；前者对正例内部排序使用阶跃、对严重负例反序使用线性尾部，后者约束正负绝对分数。二者是不同干预。[原文](https://arxiv.org/html/2110.01445)

Smooth-AP原文通过sigmoid近似排序，温度控制有效梯度区间；本项目已完成标准tau=0.01配置，不应再把它作为尚未尝试的方案。[原文](https://arxiv.org/html/2007.12163v2)

作者实现核查：

- 固定配置roadmap/config/loss/roadmap.yaml把CalibrationLoss与SupAP各设weight=1；SupAP的tau=.01/rho=100/delta=.05/offset=1.44。论文等比例组合与代码两项相加具有整体尺度差异；本项目另有13项监督，不能忽略这会改变相对梯度规模。此判断是项目化推断，不是作者报告的ReID结论。
- smooth_rank_ap.py的step_rank用t>0选择正向分支，t=0落入sigmoid分支；它与用>=0定义的阶跃在精确tie处不同。实际作者函数CPU前向：t=0给负例排名增量0.5；一正一负精确tie的SupAP loss为1/3，而>=0阶跃loss为1/2。因此不能把该公开实现无条件称为对所有tie输入的严格上界。这是合成数学检查，不说明本项目实际有多少tie。
- 同一实际函数在t=.05输出1.4933071491，在t=.050001输出1.4401，来自配置offset=1.44；不能宣称该固定配置在分段点连续。没有修改作者实现来隐藏差异。
- CalibrationLoss继承pytorch-metric-learning ContrastiveLoss，并在有reference时构造当前到reference的索引。作者requirements锁定0.9.99；该版本ContrastiveLoss默认AvgNonZeroReducer，ThresholdReducer只对严格大于0的loss取均值。因此默认实现的分母不同于论文每个query全部正/负pair均值。这是固定源码核查，不是安装原依赖后的整套复现。[固定ContrastiveLoss](https://raw.githubusercontent.com/KevinMusgrave/pytorch-metric-learning/v0.9.99/src/pytorch_metric_learning/losses/contrastive_loss.py)、[固定归约器](https://raw.githubusercontent.com/KevinMusgrave/pytorch-metric-learning/v0.9.99/src/pytorch_metric_learning/reducers/threshold_reducer.py)。

上列代码来源：[排名实现](https://github.com/elias-ramzi/ROADMAP/blob/fed37d75f475f636542b2ccd1ffc0c3918498a57/roadmap/losses/smooth_rank_ap.py)、[校准实现](https://github.com/elias-ramzi/ROADMAP/blob/fed37d75f475f636542b2ccd1ffc0c3918498a57/roadmap/losses/calibration_loss.py)、[配置](https://github.com/elias-ramzi/ROADMAP/blob/fed37d75f475f636542b2ccd1ffc0c3918498a57/roadmap/config/loss/roadmap.yaml)。作者README也披露公开代码与论文结果存在差异，不能把下载源码当作等价复现。

与当前项目的联系（尚未验证的机制判断）：候选池/完整来源AP之差同时受正负组成、去重和权重影响，不能仅由这个差值诊断batch校准故障；轻微源梯度符号变化也不够证明需要SupAP。应先完成当前1560批同参数块目标梯度与完整来源覆盖审计，再选择协议关系、排名近似或校准中的一个明确干预。任何后继均需保留同身份同环境排除与负例定义、无合法正例anchor仍保留其他监督和干扰资格等协议边界；这些是项目要求，不是ROADMAP原文主张。

本次只新增作者来源快照和CPU数学检查，无数据集图像/模型前向、无参数更新、无新mAP。它不是SOTA综述，也不宣称ROADMAP已在TriFusion或三光谱任务有效。
