# 按行未匹配质量：六端活动对照，正式训练前登记

## Problem Anchor

底线问题：在RGBNT201、RGBNT100、MSVR310上，从匹配公开CLIP/作者配方与RAW职责训练起点形成对未见身份有实际净增益的角色证据，最终超过同协议baseline并向注明资源和协议的强参照/SOTA推进。
必须解决的瓶颈：分支活动、判别性或某些首位修复不能替代完整排序的净增益；已完成独立global与同容量控制显示多种额外读取/重建未获三集一致效用。新方案必须隔离具体作用路径，证明角色贡献而不是配方/容量/继承收益。
本轮边界：不用旧失败重试、LR/gain/margin/seed/Triplet尺度搜索、N1失败重建、外部文本/掩码/DINO或测试更新；这些是本轮控制边界，不缩小总体SOTA目标。无功率温度动作，只26物理GPU0/1，25原I/O pending保持。
成功条件：新两臂共享初始化、参数、作者batch/采样/优化/50轮/RAW职责/1536L2部署与全合法gallery过滤；预登记候选相对直接活动控制和原RAWsemantic ΔmAP≥0.5且Rank-1不下降，并单列对独立global结果。三集未全部成立不得写通用机制；项目推进线不是统计显著性，固定模型bootstrap不是训练seed。最终论文仍需强基线、机制必要性、完整流程多种子与资源合格SOTA证据。

## 科学问题与旧版本边界

本次检验接收槽位之间的跨光谱传输质量分配，是否比同一模态对内统一衰减更有最终检索效用。原RAWsemantic底座、公开CLIP/new camera/作者head、RAW职责训练、seed42、50轮、作者batch与1536维L2部署不改。没有原生/容量MLP/候选重建、外部资源、新辅助loss或测试更新。

原Sinkhorn方案方法复核两轮：6.55→7.25，论文REVISE、same-family/provisional、CALIBRATION:none，实际后端未独立核实；不是论文READY，也不是模型结果。其首次CPU数学组件失败：全低分−8条件100次更新最大边际残差0.0058288574>预设0.001。原源码/失败/诊断保持，不增迭代、改null、放宽原容差或追认为通过，未启动M0/NN。以下是新的直接按行归一化方案，非原OT完整复现；旧评分不冒充新算法独立复核。

## 唯一新机制与控制

保持CNN/T读取、直接读出和原role state；Mamba按每模态16槽独立正反扫描同一原参数，输出H=B×3×16×128。与旧RAW48交错扫描不同，因此不宣称初始化fused等于旧RAW48。global原三光谱context保留，不称整个系统模态独立。

两个bias-free128×128矩阵：matching_projection普通nn.Linear默认初始化，message_output唯一零出口，共32768参数/2张量。新增构造置于CPU torch.random.fork_rng(devices=[])，原参数按完整旧role state复制，保持旧视觉/camera/head/RNG起点；新两臂state相同。没有另一个value投影、FFN、分类头或loss。

Z=matching_projection(functional LayerNorm(T))；每个无向模态对计算Cmn=Zm Zn^T/sqrt128。接收方向分别对C及C^T追加一个固定logit1的null列并FP32 log_softmax；得到16×17按行分配。无Sinkhorn/迭代、列容量约束或双向转置互易性主张。每槽真实质量r=sum真实16列，条件Q=softmax(真实log块)，不使用hard threshold或部件真值。

slot_mass候选W=diag(r)Q；uniform_mass控制W=mean_k(r_k)Q，rbar保留梯度。对同一score两者总真实矩阵质量相同、条件Q相同，区别为质量在接收槽位之间的分配。实际H向量能量不严格匹配；独立训练后的P、总质量及特征可以不同。它是任务驱动latent slot gate，不是已校准几何/语义可靠对应。

Jm=0.5·sumpeer(Wmn Hn)，M=旧output_norms[2](Hm+message_output(Jm))；自有H旁路保留，CNN/T直接读出不变。1536structured readout、原learnable gain、h=g+gain*c不改。原槽位采样忽略positions，不能命名物理部件。

## 固定训练与六端

内部semantic标签表示slot_mass，native标签表示uniform_mass，两者都没有原生图像分支。三数据集×两臂按201→MSVR→100，各真实prepare及完整batch初始化对照→8M0→独立fresh50→首次strict→一次保存距离的15配对报告。角色输入detach/global作者任务与角色任务所有权分离保持；正常作者head训练global，角色用detached当前head参数/clone BN buffer；不增加持久第二套head。201 B64/K8、100 B128/K16、MSVR B64/K4；Adam作者分组及原日程/增强，前6视觉块GPU1/后6及headGPU0，原AMP/FP32边界保持。

复用原RAW9控制/187件依赖，不重训练旧控制。每数据集主要比较slot_mass−uniform_mass和slot_mass−RAWsemantic，mAP≥+0.5且R1不下降，同时报告两臂对独立global、原semantic；共15配对，不跨epoch拼CMC。与旧RAW比较同时包含3×16和额外容量，不能把全部作用归因局部门控。三集不一致不推广。项目门不是显著性，官方集已消费，单seed/固定模型bootstrap不是训练稳定性。

M0固定8次真实完整batch，累计所有参数有限非零梯度/实际更新、优化器恰好一次、作者BN8次、完整state严格重载。失败端封存，不缩batch/延长M0/改门。成功端自己的fresh50才启动；不添加全六端M0前置屏障或旧反向parity修复。只保留一个mAP-best；该端首次strict接受后sealed文本/数组/receipt并退役M0二进制。

## 数学、诊断与成本

新的按行组件在同一预先有限强对角/全低分/随机分数检查所有17列行质量1（atol1e-5）、真实质量[0,1]、两臂同score总质量相同（atol/rtol1e-5）、反向独立接收行、两矩阵toy8累计活动/更新、零出口及strict重载。该检查不改变原Sinkhorn失败的边际定义或容差。

逐step保存真实质量mean/min/max、条件Q行熵、Q列集中度=sum行Q/16的最大列份额、peer/self/新增出口范数，以及既有实际修正/global范数。诊断计算detach，控制rbar本身不detach。不因官方分数搜索null/gain/margin/LR/seed或增加loss。若近于统一质量/所有拒绝/槽位重复，将作为原结果解释，不救分。

仅26物理GPU0/1、单进程一对，其他项目/卡不动；无功率温度动作，25原I/O pending保持。沿用既有conda，不安装/重建环境、无W&B/消息发送。六best+最多一个M0 probe+2GiB预算=4,966,055,936B；每执行阶段2GiB余量不变。预计原作者batch六端NN+逐轮评价约6小时，安排6.5–8小时估计，实际时长另报。持久队列按预计端结束前/240秒查看，观察超时不重启。

当前仅组件通过、尚无真实初始化/M0/正式成绩；新方案没有独立论文READY评分。成功后仍需self-only/普通cross-attention等必要性消融、完整流程多种子与资源合格强参照/SOTA；整体Goal ACTIVE/UNMET。
