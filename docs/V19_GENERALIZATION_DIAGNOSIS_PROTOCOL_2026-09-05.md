# V19终态表征诊断固定范围（2026-09-05，执行前记录）

V19完整Q1_FAIL保持不变。本诊断解释已完成的全三折两端结果，不修改权重，不作
新模型选择，不把读取同一OOF身份称新验证，不运行dev/official或消融。

实际依据：完整Q1六端末轮平均训练loss为0.6072–0.6109，标签平滑加权下界为
0.5783829210；全部梯度、更新、冻结和重载检查已通过，额外私有尾部容量的融合
收益仍仅+0.256035，跨折/专家/身份不稳定。冻结141-fit中只有21跨camera身份，
各fold94-source包含14个跨camera身份，47-heldout包含7个，其余为单camera身份。

只读执行 `tools/diagnose_v19_generalization_geometry.py`，范围固定如下：

1. 严格验证原Q1汇总、全部依赖源码、配置、Signal来源、六个最终checkpoint和
   训练终态state SHA。从相同来源重新构建并strict reload全部六端。
2. 每端先读取该fold全部heldout gallery，以原batch128、256×128、FP32 eval、
   无数据增强重算原五路完整AP/Rank，须与终态数组逐项精确相等；保留所有图库
   身份和记录。记录全部query最近正/负样本、同camera负例与距离，不挑样本。
3. 同一最终模型读取全部94-source身份的干净训练记录；对原五路输出报告训练
   内retrieval和全部query几何，并对既有七个分类头报告所有来源样本分类准确率、
   label-smoothed CE和逐样本原始值。source结果是训练拟合诊断，不是泛化证据。
4. 对source与heldout全部合法cross-camera query，记录每个专家RGB/NI/TI的全部
   3×3模态对：同一实例cosine、同身份跨camera均值/最近正例cosine、最近异身份
   cosine、负例索引及其是否同camera。这里只分析已有表征，不构建模态子集检索
   输出，不调整任何融合权重，不进行模态/层/头扫描或择优。
5. 两端全部来源和留出记录共18756次triplet前向。各scope独立计算距离，不跨fold
   或source/heldout混合图库。source标签用既有fold-local映射，另保存原registry
   映射和文件名，避免将其与heldout全局编码混同。

预算预计5–12分钟，先按首个端点修正时间，常规查询间隔180–300秒或预测里程碑。
仅远端RTX3090，启动前至少22000MiB空闲；optimizer0、checkpoint writes0，
不保存图像或大feature缓存到本地。完整JSON、源码/权重hash及全部结果归档，
允许读写新的诊断产物；原Q1/方案/权重不变，错误须按具体报错修复并保留失败日志。

解释限度：同一终态的相关性不能唯一证明camera、模态或容量的因果作用；真实图像
标签用于诊断不充当可部署监督预测。新假设必须等待全部诊断结果后再另行固定，
不能用此结果补写V19预注册或改变其失败状态。当前dev65及官方SOTA目标未实现。
