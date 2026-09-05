# V23 模态专属语义尾部适配主实验（执行前冻结，2026-09-05）

## 主张与范围

主假设：在同一合法V12 source初始化和完整路径身份隔离下，
冻结CLIP尾部之后的RGB/NI/TI专属残差MLP能提升三专家及其固定融合的heldout检索。
支持性要求：同checkpoint精确保留3072D Signal baseline-only。
问题证据、ICPL实际入口和移植差异见
docs/SPECTRAL_ADAPTER_SOURCE_NOTES_2026-09-05.md。
该方案没有已测科学指标；源码共享结构和V22初始化诊断不证明唯一失败原因。
不得把多出的可训练容量解释为已经隔离的模态机制，参数匹配消融留待主目标成功之后。

## 固定模型

继承原V8合法source专家和Signal，CLIP block8分叉、共享冻结尾部9/10/11；
每个尾部block的完整输出之后、原角色处理之前，增加三个按已知模态分派的MLP：

    y[e,s,m] = CLIP_tail[s](x[e,s,m])
    z[e,s,m] = y[e,s,m] + U[s,m] ReLU(D[s,m] y[e,s,m])
    x[e,s+1] = original_role_operator[e,s](z[e,s])

s为三个尾部阶段，m固定RGB/NI/TI，e为CNN/Transformer/Mamba。
每组s,m的适配器在三个专家之间共享；保留CNN局部patch算子、T全局token算子、
Mamba空间/跨模态算子，保留三个残差头及最终减去frozen reference的定义。
D:768→128、U:128→768，均带bias；U及两个bias初始化0、D权重Kaiming；
scale固定1，无新增dropout、LayerNorm、loss、router或HFER。
128来自原MODEL.ADAPTER_WIDTH，不做宽度/层数/插点/scale扫描。

新增9个MLP、1777536个参数、36个tensor。
两端总参数预期100577677；控制可训练7841292/203tensor，
候选9618828/239tensor，以真实模型绑定和M0核验。
两端同结构同完整初始state，控制的适配器冻结为零，候选的适配器训练；
冻结Signal/尾部不被复制训练，没有V19私有tail微调。
没有把V18投影、V20跨模态loss、V21 SAM或V22 MCNL并入本次结构。
旧版本封存状态不改。

每专家三个512D模态归一化后拼接为1536D残差；branch为3072+1536D；
三专家等能量4608D bank与3072D精确Signal等能量拼接成7680D fused。
baseline保留direct+SIM、camera SIE和所有冻结状态，五输出同checkpoint提供。

## 真实数据与完整比较

固定141-fit registry的3126 triplet；每fold94-source/47-heldout。
恢复对应fold的V12 source-only Signal与expert最终权重，strict load和完整SHA校验，
不重训Signal，不加载其他版本或上游RGBNT201已训练权重。
三个heldout完整图库为1000/1051/1075，共3126条；query190/179/202，共571，
21个跨相机query身份；2555条仅从query分母排除，仍保留为图库干扰。
禁止跨fold特征距离。该OOF反复用于开发，明确是reused development qualification，
不是新独立验证。30-dev和官方test读取次数0。

| 条件 | 控制 | 候选 |
|---|---|---|
| endpoint | frozen_zero_spectral_adapter | trained_spectral_adapter |
| 新模态MLP | 冻结零输出 | 学习残差 |
| 原三专家和分类头 | 训练 | 训练 |
| epoch/warmup | 20/5 | 20/5 |
| fold0/1/2步数 | 580/560/540 | 580/560/540 |
| 总优化步 | 1680 | 1680 |

共3fold×2端×20epoch、120行epoch、3360优化步/前反传对。
真实远端RTX3090、seed42、B64/K8，无累积；原采样/共享几何增强、workers4，
256×128，AdamW LR0.00035、wd0.0001，warmup5/cosine20，AMP/scale256，
CuDNN deterministic=true/benchmark=false，原gradient checkpointing，eval batch128。
两端采用相同14项原始ID/Triplet分量：fused ID0.25/Triplet1，
每branch ID1/12/Triplet0.25，每residual ID1/12/Triplet0.25；
smoothing0.1，triplet margin0.3。记录全部分量，不引入相机或跨模态辅助loss。
两端每次构造和每个训练入口seed42；全sample order SHA及前8增强/索引/路径必须配对。
额外可训练参数带来的反向算量/显存/时间不同，不能声称等计算预算或参数机制已隔离。

只存最终checkpoint，重建原source后strict reload，完整model state SHA等于训练final，
执行同一完整图库的baseline/fused/CNN/T/M全部mAP/R1/R5/R10。
不能用旧版本control结果、只挑fold、挑专家或挑epoch。即使先完成一折负值也完成全部Q1。

## T0与M0

T0全部在远端CUDA上执行合成模型契约：两个条件的零适配器全输出与原模型相同、
strict reload不变；单模态dispatch/梯度隔离；零up初始化首步down梯度为零的正确性；
两个条件各3步合成优化仅候选改变适配器、原Signal输出不变。
5个pytest用例，6步toy optimizer；无真实数据、source checkpoint或项目训练步。
新增代码只做AST静态检查于本地，禁止本地模型和tensor执行。

M0：全部三fold两端各8真实source batch前向（48次），每模型额外用第一batch
临时原encoder只读核对零适配器与legacy全五输出完全相同（额外6次前向）。
两端完整initial state、source state绑定、sample/augmentation/output SHA严格匹配；
参数差恰为1777536、trainable tensors为203/239。
随后fold0两端各8不同batch实际优化，再从fresh source候选固定首batch100步。
共116项目优化步和116前反传对，另54次forward-only；M0权重不用作Q1初始权重。
所有训练tensor在整个各自容量/过拟合阶段有有限非零梯度，overflow0，
冻结完整state不变，显存peak<24576MiB。首个零up步骤down梯度允许为0，
但累积8/100步未到达的tensor必须导致M0失败，不放松该门。

总loss下界F=0.75*H_94(smoothing0.1)=0.57838292104621，triplet下界0。
固定(L100-F)/(L1-F)<=0.1，以第1/第100次更新前loss计算；
保留全部100步，禁止中间最低值、末段均值、换batch/seed/epoch或调scale救回。
M0失败保留完整M0_FAIL并停止Q1；通过后原单进程自动完成全部6端Q1。

## 五项固定科学门及后续

1. candidate aggregate fused mAP比matched control至少+1.0pp；
2. 三折各自fused增益非负；
3. CNN/T/M各aggregate mAP增益非负；
4. 全571query、21身份cluster、10000次seed42 Bootstrap的95%下界严格>0；
5. candidate fused mAP严格高于同checkpoint baseline和三个固定expert。

必须全满足才有D1资格。报告全部三折五输出、全部21身份/逐query数组及变化，
全部六训练/重载/冻结/数据次序/权重SHA，不把工程M0当作泛化。
失败封存本次V23，不扫描width/stage/scale/模态/专家/epoch/LR/seed或改loss重跑，
不进行D1/dev/official/消融。通过后首次dev前另行固定D1执行细节。
当前best dev58.4050/59.3939、开发门65mAP及官方目标85.3/87.9未达状态不变。

## 运行与交付

启动前真实空闲>=22000MiB，核对代码/配置/计划/CLIP/全部source SHA。
T0预计10–20秒，M0预计4–7分钟，完整Q1预计75–100分钟（比V22增加适配器计算）。
按M0和完整paired-fold预计结束窗口查询，长任务查询间隔180–300秒；
用实际epoch时间修正ETA，SSH观察超时不重启原进程。
保留完整原始JSON/log/六checkpoint/receipt，终态完整数组复算及独立审计。
代码及时推送GitHub，master按新增阶段继续写并同步服务器和桌面。
