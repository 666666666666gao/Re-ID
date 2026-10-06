# 实验计划：新增证据训练职责的机制对照

状态：源码实现与环境元数据绑定见证；真实初始化、M0、正式训练均未运行。

独立头三端0/3及§877完整g/c/f分解已经闭合。201/MSVR同模型g与独立global评分一致，100仍有0.568336mAP的global差距、加修正又下降0.683912；c单独34.50mAP不等于互补。新实验只改变新增证据的监督内容，不改原semantic角色、读出、1536维部署、公共CLIP/new camera初始化、borrowed detached global heads及作者raw任务。

## 要回答的有限问题

相对global的合法实例纠错/保留目标，是否比MDReID式batch距离比目标及当前raw职责目标更有净收益？这是诊断性监督比较，不是已经完成的P1/P2/P3方法，也不声称公式或“上游责任”原创。V26已有九专家×模态责任反传，R2已有合法跨环境多正例排序，CIRC已有删除效用与relay分析；若本轮有效，后续仍须与它们直接比较，并验证增量信号是否真的改变区域证据形成。

## 两个新条件

共同：`L = L_global_author_raw + L_role_author_raw + L_increment`，最后一项只添加一次，不向车辆三个头重复求和。无新增参数、头、原生CNN、文本、mask、teacher、采样或推理变化。global与角色原梯度分工保留，不宣称三个数据集严格数值保护global。

1. `md_batch_ratio`：独立按发表机制表达融合/global/c三类raw欧氏距离。global和c比较支停止梯度；融合距离保持梯度。对batch最难正例最大值和最难负例最小值构造比例，`L = Dp_f/(Dp_f+Dp_g+Dp_c) + 1 - Dn_f/(Dn_f+Dn_g+Dn_c)`。正例包括作者原有self位置，默认不L2归一化，距离平方使用既有1e-12下界。它是当前g/c组件上的机制适配，不是完整MDReID共享/私有模型或作者复现。缺少代码许可，不复制作者实现。
2. `repair_keep`：g停止梯度，f为当前联合1536维归一化部署向量。每个query使用全部合法同ID异环境正例和全部异ID负例，`m=s(i,p)-s(i,n)`。g的margin≤0为repair，目标f margin≥0.1；g margin>0为keep，目标f margin≥g的原margin。先对每query同cell关系取均值，再对有支持query取均值；两cell各0.5。单一loss系数1、margin0.1事前固定，不按官方分数搜索。

MSVR环境从原protocol的scene读取，201/100从camera读取；不使用MSVR日志的view代理camera。原camera embedding和数据增强不改。CPU元数据见证显示三集全protocol basename无环境冲突、第一来源batch路径可解析，MSVR view/scene确实不同。查询/图库仅用元数据做绑定，不参与训练标签或目标选择。

旧真实来源批次201有92/2649没有合法正例；空query/cell的新增loss数学定义为0，普通作者身份任务继续。不是填补检索缺分，不能把库存支持当新增loss实际活动。训练步记录合法正例/多个正例query、repair/keep关系与活动数。

## 分阶段执行

M0：只26物理GPU0/1、max1分段模型、暖tri_reid。按201→MSVR→100，每集两个条件，真实相同公共起点、初始化参数/state匹配原raw semantic；各8更新、严格M0 reload及作者BN计数8。记录隔离增量loss的g梯度必须为None，c及六个Q/K参数累计非零有限梯度；没有活动的条件停止，不伪造formal分数。每个M0完成并收齐实际receipt/hash/reload后退役其仅工程用途probe，fresh正式不加载M0权重。六个M0的批次/初始化必须配对，全部通过才允许正式阶段。

正式：六端fresh50/seed42，共300epoch/预计12968步。201 B64/K8，100 B128/K16，MSVR B64/K4；原作者优化器/日程/增强/heads/raw度量不变，新增联合loss只求一次。复用既有独立global-only和raw semantic三集控制，不重跑旧对照。每端同一mAP-best报告全部CMC，完整末轮/轨迹、修复/新增错误、身份AP和真实成本同时报告。

当前磁盘需要独立容量核验：M0序列启动至少2GiB+512MiB，正式六端保留best/最佳与正式距离需要更大预算。仅退役当前阶段不再使用、已核回执及SHA的自训冗余文件；作者/公开/赢家/当前依赖保留。不得为启动下调2GiB预留。正式容量未核验前不自动启动full；用户已有实验授权，容量达成和六M0通过后继续，不新增批准流程。

主要比较repair_keep−md_batch_ratio，并分别对比raw semantic和独立global。沿用事前mAP≥+0.5/R1不降的研究推进线，非显著性阈值、非SOTA完成。三个数据集都完整，不以201正值覆盖车辆负值；0/3或活动失败封存，拒绝margin/gain/LR/seed救分。本轮不能据一次成功自动采用所有候选模块或宣称新颖性。

## 近邻与证据边界

[MDReID论文](https://arxiv.org/html/2510.23301v2)和[固定作者loss源码](https://github.com/stone96123/MDReID/blob/3525ac2da1a2a90a5a160c930fac674b4f226f6c/layers/triplet_loss.py)支持机制参照，源码只核读不搬运。两种目标改变距离空间、监督关系与归约方式，胜出只能支持这个目标组合，不能唯一归因某一个子因素。官方测试已被开发/选点消费，完整流程种子稳定性仍待架构固定后补足。三个新主模块和三集SOTA目标仍ACTIVE/UNMET。
