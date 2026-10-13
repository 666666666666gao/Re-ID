# §947：保留实际区域，预登记固定证据比较

原三条件队列继续，训练488源码和初始化/损失/seed42/50轮均不改。最新实际训练观察仍是§946的00:43:10：joint_delta完成首次strict，E8 74.5331/79.1866/88.2775/91.8660；add_delta当时8/50，第三项当时未启动。本文不冒称当前实时epoch，不读取中途分数。唯一整组观察器73726及原join55981到02:02:48.576899；没有新观察器、NN或训练重启。

现有固定best诊断只保存slot池化后的因子，无法回答现有局部证据是否被比较方式用漏。因此仅改训练后诊断：同一次query/gallery提取时，readonly pre-hook保存实际处理后和matched-reference的B×3角色×3模态×16槽位×128区域；不替换forward输出，不增加编码或更新。新版诊断不属于当前488训练seal。§945旧导入证据保留，新版单独核查，不追认旧证据覆盖新代码。

实际warm-conda CPU导入EXIT0，无模型/NN；实际readout组件6次forward，其中3次注册真实hook，三模式输出/state完全一致；捕获次序/shape与输入相同。独立两基向量fixture：均值[.5,.5]，对称MaxSim[1,.74999994]，并检查槽位置换及相同token。三个fixture比较/0更新/0真实图像/0完整CLIP-Mamba调用。Root源码审查通过，不称独立agent或生产模型/提取/排名验收。

全部三端50/首次strict和一次保存距离CPU报告成功后，才以exact source/artifact seal运行每个固定best一次完整提取；需验证state/梯度不变、官方fused距离/指标复现及metadata次序。缺少任何一项不运行后续CPU比较，也不放宽容差或更换best。

后续纯CPU在同一份region arrays上做6项预登记局部比较（三模式×处理后/reference）。每个role/modality组独立token L2；比较所有16×16成对cos均值与双向row/column maxima均值，再固定平均九组；完整合法gallery和原camera过滤。没有global融合倍率、拟合、teacher、种子/方案选优或重排子库。逐query AP/首位修复新增错误、身份宏平均和CPU时间均保存。该mean规则不是原factor/dense读出；因此MaxSim较好也不能单独证明原压缩是掉分原因。

晚交互启发来自[ColBERT原论文](https://arxiv.org/abs/2004.12832)和[作者v1源码](https://github.com/stanford-futuredata/ColBERT/blob/706a7265b06c6b8de1e3236294394e5ada92134e/colbert/modeling/colbert.py)，cached commit/source SHA见COLBERT_PRIMARY_947.json。项目使用对称均值MaxSim，仅借局部比较原则，没有安装/训练完整ColBERT或加载其权重。新增多向量存储和成对计算属于诊断协议，结果不替换原1536维正式主表、不宣布新主模块或全部晚交互成立。

存储算术：新增区域每条件246546432B，原features/factors/descriptors110432256B，三距离8386752B，三条件共1096096320B，另有序列化/JSON开销。此为投影，不是当前磁盘空闲实测。真实终态必须先核磁盘预算。

因该有证据的容量需求，明确修订§946清理次序：三端首次strict及原CPU报告通过后（报告仍需要M0 binaries），先D盘逐份size/SHA核验归档3份M0临时probe，然后只退休服务器重复probe，再准入完整region诊断。诊断不读取M0 binary；其seal保存M0 receipt/accepted/CPU报告。未来要运行原M0-dependent verifier/report时必须恢复exact probe，不能改旧verifier消除依赖。所有best、恢复依赖、数据集、唯一记录保持。现在实际退休数仍0。

三端/区域诊断/CPU比较尚未实际完成，full三数据集超过Signal Goal ACTIVE_UNMET。当前更新只是可执行诊断准备及预登记，非新检索成绩；之后根据真实结果决定新增证据机制，不能转成gain/rank/margin/seed搜分。
