# V24 source-only环境身份原型：单一完整主比较

状态：代码实现及AST检查完成，方案执行前冻结并随本提交登记SHA；T0、M0和Q1尚未执行。日期：2026-09-06。

## 主假设与证据

在保持原三角色推理结构与真实全局身份标签的条件下，按摄像头均衡的source身份原型，
以及普通视图更新、强增强视图学习的环境内身份竞争，能改善未知身份完整图库排序。
这是一个训练监督覆盖假设，不增加推理模块，不重跑已封存V22/V23干预。

当前每fold94个source身份中80个单摄像头、14个双摄像头；一共108个真实身份/摄像头组合。
原1680batch重放的跨摄像头正对仅8.070791%，1580batch仅一组跨摄像头身份。
V23已完整Q1失败：fused增益-0.25270450090745555pp，五项门全部失败。
这些证据支持研究监督覆盖，不证明相机偏差是唯一失败原因。

IICI作者源码commit d60e09bad6637b076a3c1347dfe59745b4cd76b3提供
普通视图更新原型、强增强做环境内身份竞争的思路，但其每ID单摄像头断言不适用本项目。
XBM作者源码commit223ecdc25f71ef1721a58bc87cc567025a32bc92提供batch外负例动机；
本版采用身份原型，不采用FIFO样本队列，不声称复现IICI或XBM，也不引入MCNL。
实际源码边界见docs/SOURCE_PROTOTYPE_MEMORY_RESEARCH_2026-09-05.md。

## 固定推理路径与初始化

两端均恢复各fold合法V12 source-only Signal与原V8角色/头最终权重。
恢复完整原始模型，不加载V23或其他后继训练权重，不保留V23新增MLP。
冻结Signal和CLIP共享尾部，训练原CNN/Transformer/Mamba角色与七组分类头。
预期总参数98,800,141、可训练7,841,292、203个tensor，M0以真实构造核验。
原3072D Signal、三个1536D残差、4608D残差银行和7680D融合保持。
原型仅是训练状态，不加入model，不参与检索或终点评分；新增推理参数0。

## 原型定义与更新合同

用g(x)=normalize(fused_embedding(x))表示最终检索表示，7680维。
只用当前fold source、当前端点的当前模型，通过固定eval resize完整遍历全部source记录，
先按真实(id,camera)求特征均值并归一化，初始化108个P[y,c]。
不以零值占位，不从heldout/dev/gallery读取训练特征。
每端独立初始化；M0各fresh阶段及Q1各端都重新建立，禁止跨fold或端点共享。

全局身份原型 G[y]=normalize(mean_{c属于身份y} P[y,c])。
双摄像头身份的两路各占一半，避免样本较多的摄像头淹没另一真实正环境。
全局竞争为全部94身份，每anchor有93个真实不同ID原型负例。
环境竞争仅用当前摄像头真实出现的身份原型，目标仍是原全局ID。
不同摄像头的同ID不会变为负例；不生成伪身份或伪相机。

每次更新使用普通视图、更新前参数提取的detach特征。
同batch同(id,camera)的多次出现先取均值，然后只做一次EMA：
P[y,c] <- normalize(0.2*P[y,c] + 0.8*mean(normalize(g_weak)))。
重复采样/不同擦除视图按batch实际出现次数进入该均值，避免顺序逐样本EMA。
只在两视图反传和一次优化器更新完成后修改原型；强视图不更新原型。
原型在当前loss计算期间不变，不让当前强视图成为自己的新教师。

每个原型保存last_update/update_count，记录逐epoch年龄分布、未刷新组合、实际负例
覆盖和真实跨摄像头正原型覆盖；完整初始/最终状态及SHA随训练工件保存。
EMA原型会滞后于当前模型，此设计不声称特征slow drift已证实，也不事后按指标选择缓存年龄。

## 两视图与固定目标

每条三模态记录只采样一次flip/crop，六份模态/视图共享几何。
256x128、padding10、flip0.5；普通视图保持原erase0.5。
强视图在同一几何上逐模态施加uniform[0.8,1.2]亮度缩放和erase0.6；
不加hue/saturation/contrast，不把普通视图已擦除内容再当成强视图底图。
归一化与RandomErasing的其余参数保持原项目设置。

L_original保持原七组ID/Triplet，14项及原权重不变，smoothing0.1、margin0.3。
L_base=(L_original(weak)+L_original(strong))/2。
L_global=CE(g_strong * G^T / 0.05, true_id)。
L_environment=CE(g_strong * P_camera^T / 0.05, true_id_in_camera)。
L_prototype=(L_global+L_environment)/2。

控制endpoint ordinary_two_view：L=L_base。
候选endpoint environment_identity_prototype：L=L_base+L_prototype。
两端都构造、计算、更新原型，控制以0系数进入总loss，候选系数1；不扫权重/温度/momentum。
原型直接约束最终融合表示，不要求三种模态向量互相余弦对齐。
两端视图、采样次序、起点、模型参数量、原型读写与前后向数量配对。
原CrossCameraIdentitySampler保持，batch内正对比例不宣称提高；改善来自额外原型关系覆盖。

## 完整数据与训练预算

3fold各94-source/47-heldout；source记录2126/2075/2051。
source各摄像头身份数为fold0:72/31/2/3，fold1:71/31/3/3，fold2:69/30/3/6。
全部真实source身份保留。每端seed42、B64/K8、workers4、AdamW0.00035、wd0.0001，
20epoch、warmup5/cosine20、AMP初始scale256、原gradient checkpointing。
每批先普通视图反传0.5 L_original，再强视图反传0.5 L_original+系数L_prototype，
随后仅做一次optimizer step；不跨batch积累梯度。两端每更新均为两个前反传对。
各端580/560/540更新；Q1总3360更新、6720前反传对、120条epoch汇总。
全部28个普通/强原loss分量、两个原型项、加权总loss、关系数及时间均完整记录。

每个Q1端开始前完整初始化source原型，合计12504条source只读特征前向；
不进行source检索或用source AP选点。只保存epoch20最终checkpoint及最终记忆。
新模型strict reload最终权重后，独立用原评价器提取五路特征，记忆不参与评价。
完整图库1000/1051/1075，合法query190/179/202，共3126gallery/571query/21身份。
2555条仅从query分母排除，保留图库干扰。无reranking或跨fold特征距离。
当前OOF已反复开发，明确复用资格协议；30-dev和官方测试访问0。

## T0和M0执行门

T0仅远端CUDA/合成数据：六个测试覆盖摄像头均衡均值、真实正负loss、
强视图梯度/记忆只读、重复组一次EMA/未见组保留、label0与state往返、六视图共享几何。
无真实checkpoint、数据集图像、项目optimizer或检索。

M0预检三fold两端，每端初始化全部source原型，再取8个真实双视图batch，
核对完整模型初始SHA、记忆SHA、全部双视图路径/增强SHA和五输出严格配对。
6模型x8batchx2views=96只读batch前向；不改变model或memory状态。
另fresh fold0两端各8不同batch更新，以及fresh候选固定第一双视图batch100步。
共116项目优化步、232前反传对；M0权重/原型不用于Q1。
九个fresh M0模型的source初始化合计18882条只读source特征前向。
所有203训练tensor须在各容量/过拟合完整阶段获得有限非零梯度；冻结state不变、
overflow0、峰值reserved<24576MiB。两次容量分别记录真实峰值。

原ID目标下界F=0.75*H_94(0.1)=0.57838292104621；新hard-target CE及triplet下界0。
固定第1和第100次更新前总loss，要求(L100-F)/(L1-F)<=0.1。
保留100步全部分量，不用中间最小值、换batch、均值或调辅助系数救回。
任何M0固定门失败则保留M0_FAIL并停止Q1；通过后原单进程完成所有六端。
M0仅为工程资格，不作为未知身份有效证据。

## 科学门、失败解释与后继

沿用五项科学门，必须全部满足：
1. aggregate fused比本轮matched control至少+1.0mAP；
2. 三fold各自fused增益非负；
3. CNN/Transformer/Mamba各aggregate mAP增益非负；
4. 全21身份、10000次seed42 cluster bootstrap的95%下界严格>0；
5. 候选fused严格超过同checkpoint baseline及三个完整专家输出。

完整报告六端五输出mAP/R1/R5/R10、全部21身份与逐query变化、最终权重和记忆绑定。
先负的fold也不能早停，所有六端完成才写终态。失败封存本版本，不扫温度/momentum/
权重/增强/记忆刷新/相机子集/epoch/LR/seed，不运行D1/dev/official/消融。
结果只能支持或否定此完整原型目标的固定比较，不单独归因于其中某项。
参数量相等不等于跨论文同算量；实际记录两端时间、显存、初始化和6720前反传预算。

通过后首次固定dev访问前另定D1；主结果未成功前不做消融或多种子。
核心三数据集保持RGBNT201/MSVR310/RGBNT100；下一跨数据集验证优先MSVR310，
仍按其scene/time规则。车辆数据已安装不算模型成绩。
预计T0几十秒，M0约10-20分钟，完整Q1约2.3-3小时；以真实阶段耗时修正ETA。
长任务按预计里程碑、180-300秒间隔观察，不因观察超时重启原过程。
