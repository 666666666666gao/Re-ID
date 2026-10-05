# 选择性跨光谱区域传输：方法候选，尚无实现或训练结果

## Problem Anchor

底线问题：在RGBNT201、RGBNT100、MSVR310上，从匹配公开CLIP/作者配方与RAW职责训练起点形成对未见身份有实际净增益的角色证据，最终超过同协议baseline并向注明资源和协议的强参照/SOTA推进。
必须解决的瓶颈：分支活动、判别性或某些首位修复不能替代完整排序的净增益；已完成独立global与同容量控制显示多种额外读取/重建未获三集一致效用。新方案必须隔离具体作用路径，证明角色贡献而不是配方/容量/继承收益。
本轮边界：不用旧失败重试、LR/gain/margin/seed/Triplet尺度搜索、N1失败重建、外部文本/掩码/DINO或测试更新；这些是本轮控制边界，不缩小总体SOTA目标。无功率温度动作，只26物理GPU0/1，25原I/O pending保持。
成功条件：新两臂共享初始化、参数、作者batch/采样/优化/50轮/RAW职责/1536L2部署与全合法gallery过滤；预登记候选相对直接活动控制和原RAWsemantic ΔmAP≥0.5且Rank-1不下降，并单列对独立global结果。三集未全部成立不得写通用机制；项目推进线不是统计显著性，固定模型bootstrap不是训练seed。最终论文仍需强基线、机制必要性、完整流程多种子与资源合格SOTA证据。

## 当前证据与两条路线

§859六端全部50轮、首次strict与15配对完成。patch−mean mAP在201/100/MSVR分别−0.338730/−0.059234/+0.000801，0/15项目门；不重开该候选重建。源码GlobalTokenRoles.forward的Transformer按B×3独立运行；Mamba输入permute后48-token把同编号RGB/NIR/TIR交错，正反扫描共享参数，未建立对应或允许拒绝。读取处输入detach、全局/角色职责分离已做，继续保持。

路线A：只改变角色跨模态传播权限。以原RAWsemantic为底座，不带原生/容量MLP/候选重建。Mamba先按每模态16区域独立扫描，跨模态交流由软部分分配完成，比较保留未匹配质量与把真实候选强制归一化的直接活动控制。选择A。
路线B：额外DINOv3或文本先验改善证据形成。RoDI/DEEP等提供资源有区别的强报告，但新资源/主干与配方对照尚缺、不是本次传播权限的可识别干预；留为不同资源路线，不同时加入。CLIP本身已是预训练视觉模型，不为“frontier”额外堆组件。

## 方法与精确接口

保持原role sampler输出C0,T0,M0均B×3×16×128。CNN和Transformer路径及其直接读出不变。Mamba输入仍M0+anchor+旧Transformer→Mamba桥接，但将其reshape为B*3×16×128，经旧mamba_norm、同一旧Mamba正反扫描、均值，恢复H=B×3×16×128。这样跨模态状态不再经一条48序列隐式传递；它是与旧RAWsemantic明确不同的作用路径，不宣称初始化fused等于旧48序列。只要求新两臂完整state以及初始化raw/L2/global/head逐项匹配；两臂旧参数/公开视觉/camera/head起点与登记源一致。

新增单个Transport模块，仅3个bias-free128×128矩阵：matching_projection、message_value、message_output，共49152活动参数/3张量。matching与value普通初始化，message_output为唯一零出口。无新增MLP/FFN/归一化参数/分类头/auxloss。Transformer输出经functional LayerNorm再同一个matching_projection得到Z。每对m<n的分数Cmn=Zm Zn^T /sqrt(128)，reverse使用同一软分配转置；不分别引入方向预测器。

将16×16分数扩展为17×17，最后行/列及角的固定null logit=1（公开SuperGlue默认初值启发；本版本不学习标量，避免forced控制中的闲置标量）。真实区域每行/列目标质量1，null行/列目标16，总质量32。log空间Sinkhorn固定100迭代，输出logP+log32；真实块Pmn的每行和≤1，缺少的质量属于未匹配项。没有硬阈值、部件标签或同编号位置真值，不调用SuperGlue权重/GNN/原几何监督。这是借鉴分配数学，不是完整SuperGlue复现。

partial臂：Wmn=exp(logPmn真实块)，直接保留其不满1的质量。
forced臂：Wmn=softmax(logPmn真实块,dim=-1)，把真实候选条件化为满1；reverse按转置的log块再对接收方行softmax。两臂均实际执行完整相同分配及全部三矩阵，无闲置参数。forced刻意去掉拒绝的幅度作用，是直接作用路径控制，不冒称真实对应。

peer消息Jm=1/2 *sum_{n≠m} Wmn message_value(Hn)。各模态自身Hm始终保留，最终M=旧output_norms[2](Hm+message_output(Jm))。CNN/T直接旁路照旧，原1536维structured readout、原learnable gain和f=Normalize(g+gain*c)不变。私有保留指这条不通过peer消息的作用路径，不宣称可辨识语义解耦。global仍有旧三模态上下文，这里只隔离区域peer传输，不宣称整个系统模态独立。

## 训练、推理、来源与局限

继续原RAW职责分离：global自己的作者raw ID/Triplet更新共享与头；fused用sg(g)、当前作者head的detached参数/clone BN buffer，角色只接受原主任务，不增加对应/循环/重建/排序辅助loss。仍seed42，201 B64/K8、100 B128/K16、MSVR B64/K4，三集各50轮，一份mAP-best的全部CMC。保留原FP32/AMP边界、第一6视觉块GPU1/后6及head GPU0；没有旧parity/kernel修复。

推理在一个对象三光谱内部计算，不对query–gallery联合推理、不更新测试模型或库统计，仍一次1536维离线检索向量。49152参数不计作已有语义证据改善的保证。

仅身份监督不能保证P是几何/语义真匹配；本轮主张限定“是否保留未匹配质量带来可用传播权限”，不称已校准可靠性。slots原读法可能重复，可能全部拒绝/全部强制匹配/向同一区域集中；这是已有读取趋同风险的新可检验后果，不预先加额外loss救。每步记录real匹配质量mean/min/max、行熵、最大列份额、peer/self范数；只诊断，不能依据官方分数改方案。CPU组件须在构造有明确对应/无对应的分数上验证分配边界、拒绝确实不被再归一化、转置方向、所有3矩阵8更新累计活动以及strict组件重载。8M0是真实完整author batch；未通过就记录并停该端，不用缩batch/改门/延长M0救。

零出口只保证新两臂初始化相同，不能保底训练收益。对应质量无真值，只能报告软分配统计与身份检索效用；若后续需要真实部件/可靠性主张，另须对应验证数据，不能用循环或身份loss代替。省掉auxloss与外部先验减少归因混合，但相应科学结论严格限定。

## 最小验证

唯一新训练块：三数据集×partial/forced共六端，各真实初始化匹配→8M0→fresh50→首次strict→一次保存距离CPU报告。原sealed RAWsemantic与独立global六控制直接复用，不重训。主要两个配对每数据集：partial−forced、partial−RAWsemantic，均ΔmAP≥0.5/R1不降；同时展示partial/forced对global/RAWsemantic及全部query/身份、首位修复/新增错误、完整50轮与成本。参数/计算/初始state相同的新两臂隔离是否保留拒绝质量；对旧RAWsemantic比较同时包含Mamba从48到3×16的改变与额外模块，不能归因全部为null项。

若两臂均不超过RAWsemantic/global，传播重写不晋级；若forced更好，不声称拒绝有效；若只有单集正，不推广三集。没有评分后的LR/gain/margin/null logit/Sinkhorn迭代/seed搜索。若主结果成立，再另登记必要的self-only、普通cross-attention等结构消融与完整流程多种子，不本轮一并堆。

## 创新与时间预算

可争取的主张是“显式私有路径与部分共享传输比无条件真实候选归一化更有用”，不是首次Sinkhorn、首次共享/私有或三个算子创新。与SuperGlue、MDReID、Signal的差别是多光谱角色的实际传播权限及最终离线检索效用，当前未知是否有独特价值。已有SuperGlue有几何监督，本方案没有，不迁移其指标或名称。

历史同等author batch六端训练+逐轮评价约6.0小时（旧2350*2+1396*2+7123*2秒），新增100次小矩阵Sinkhorn成本须实测，安排6.5–8小时范围估计，按实测起点/端耗时结束前观察，不提前轮询NN。存储需六best+一M0 probe+2GiB余量；现实际free2.654879744GB不足，只有核实完整旧回执/weightSHA/保留winner、排除所有原187/354及后继控制依赖后清理闭合冗余权重才启动。该数值来自23:22实际CPU/允许两卡memory-only检查，不是容量保证。GPU0/1当时仅18MiB各，未见TriFusion工具进程；其他卡和项目不动。

该提案仅待方法复核，未实现、未M0、未训练、无性能结论。研究-refine最多两轮；分数是同家族provisional方法判断，不能代替检索证据或为了9分增加组件。整体Goal仍ACTIVE/UNMET。
