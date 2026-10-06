# 独立角色头固定best分解诊断

状态：方案和源码准备；尚未执行新NN诊断。

三端独立头fresh50/first-strict及唯一六配对报告已闭合（0779be3d/§41.874）。本轮只检验：同一best的global与fused之间净修复多少；独立global-only与该模型global的差距是多少；correction单独检索、实际幅度与旋转是否与净效用一致。它是解释性诊断，不是新主模块或性能提升实验。

实际CPU检查表明三份official_distances.pt只含fused及身份/环境元数据，不能CPU推算缺失g/c。需要对三份封存mAP-best各执行一次原完整query/gallery前向：201 best8，MSVR best38，100 best26。原权重、epoch、seed、初始化与正式报告不变，无optimizer、更新、增广、测试时学习或新选点。FP32 eval，复用已执行过的模型构造/严格加载、extract/scorer/paired/describe。独立头仍不用于部署前向。

对每个原query/gallery记录提取g/c/h/f，完整数量分别836/836、591/1055、1715/8575，合计13,608个记录曝光。按原camera/scene过滤、保留全部图库。评分global、correction、fused与已保存独立global-only；完整逐query/身份修复和新增错误。检查g/c/h/f公式、有限值、model/buffer SHA及无grad/无更新，复现原fused和独立global指标到既有1e-5容差；不更改既有容差或把旧记录改判。

保存全部三类诊断距离及metadata，query/gallery逐样本g/c/h范数、实际gain*c/g比例和g→f夹角，以及DIAGNOSIS.json。完整四套特征向量不另存；它们的省略不省略任何前向、评分或样本统计。距离逻辑tensor载荷192,342,312 B，统计和序列化预算合并上界256MiB；启动要求2GiB+256MiB=2,415,919,104 B，保留既有2GiB预留。原通用诊断的四套向量缓存约334MB额外开销对本问题无必要。

现有F2 raw201已退役，资格读取在任何删除前失败并保留；已确认的F3 metric-raw MSVR是有效强参照，不删除。本轮没有删除新权重、转移到25、降低预留或更改模型。公开CLIP、RAW187、三份当前正式best及初始化/receipt/protocol被封入输入seal。

顺序201→MSVR→100，只26物理GPU0/1、max1模型，复用已有tri_reid环境。启动时仅查0/1显存，不查询GPU2/3、功率或温度。不得在NN活跃时同步源码/Git；按实际预估里程碑或180–300秒观察同一进程。预计5–10分钟，具体依据实际运行修正；失败保留，不盲重启或按分数调gain/margin/seed。

三端完成后再做一次完整接收/汇总，原训练六配对报告不重跑。结果为官方已选best的描述性分析；同模型global不是独立消融，单独c能力不证明互补，角度/幅度与收益关联不证明因果。来源支持、V26/R2/CIRC已有机制边界继续适用；宽目标ACTIVE/UNMET。
