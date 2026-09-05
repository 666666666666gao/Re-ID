# V22 相机负例残差度量主实验（2026-09-05，执行前冻结）

## 主假设与可声称范围

在每fold仅14/94身份有跨相机图像、原采样正对仅8.070791%跨相机的条件下，
将三个残差专家的batch-hard triplet替换为相机感知MCNL，能否改善完整路径
身份隔离的heldout检索。只改变残差的度量目标；七路ID、融合/单分支triplet、
表示结构、初始化、采样/增强、优化器和预算保持固定。
支持性要求是完整3072D Signal baseline-only从同checkpoint精确保留。

已有source满分和heldout落差是问题证据，不是该方法有效或camera唯一因果的证明。
MCNL原始论文和移植差异见 docs/CAMERA_SUPERVISION_AND_MCNL_SOURCE_NOTES_2026-09-05.md。
这是新度量假设，不是V18投影方向、V20温度/权重、V21rho或epoch变体。
所有旧版本的固定负结果及未执行阶段保持封存。
新控制端与实验端从相同V12原始source权重独立训练，不能复用旧版本有利指标作对照。

## 固定模型与完整数据路径

两端沿用原V8：frozen Signal/CLIP block8 token，共享frozen CLIP尾部9/10/11，
CNN局部、Transformer全局、Mamba空间及跨模态三个角色残差。
每角色每模态512D，归一化后拼接为1536D残差；三个角色等能量拼接为4608D。
单分支输出3072+1536D，fused3072+4608D；baseline-only保留原Signal direct+SIM、
camera SIE和所有冻结状态。推理不运行MCNL或分类器，无新增推理参数。
预期总参数98800141、训练参数7841292、203个训练tensor，以实际绑定核对。

每fold严格恢复V12对应的94-source Signal及expert checkpoint；不得读取heldout
身份训练出的权重。配置固定六source权重、V12summary和CLIP完整SHA。
141-fit registry固定3126triplet；每fold94-source/47-heldout，heldout全gallery
为1000/1051/1075，共3126，合法跨相机query190/179/202，共571（21身份）。
2555记录仅从query分母排除，全部保留为gallery干扰项。禁止跨fold距离。
OOF已被跨版本开发复用，明确标为reused development qualification；
不是新的独立验证集。D1/dev/official访问为0。

## 单一目标替换

对专家e的1536D残差 z 做L2归一化，计算同batch欧氏距离d。
对anchor i定义真实ID正例P_i（排除自身、保留全部已知跨相机同ID），
同相机不同ID负例N_same和异相机不同ID负例N_other。
V为三个集合都非空的行；固定margin m1=m2=0.1：

    d_pos(i)   = max_{j in P_i} d(i,j)
    d_same(i)  = min_{j in N_same(i)} d(i,j)
    d_other(i) = min_{j in N_other(i)} d(i,j)
    MCNL_e = mean_{i in V} [
        relu(0.1 + d_pos(i) - d_other(i))
        + relu(0.1 + d_other(i) - d_same(i))
    ]

只将三项0.25 * residual_triplet_e替换为0.25 * MCNL_e；没有新增混合权重。
原fused ID0.25/Triplet1，三个branch ID各1/12/Triplet各0.25，
三个residual ID各1/12不变；身份系数合计0.75，smoothing0.1，
保留的triplet margin0.3。MCNL距离/归一化/hinge均FP32，其梯度返回AMP模型。
控制端使用原残差triplet；两端均记录两种残差度量供全轨迹诊断，
只有注册端点选择的那一种进入总loss。该选择是固定实验条件。

源标签支持已完整重放1680批且每批42–64行有效，8724/107520行缺少一类负例。
这些无效行不进入MCNL平均值，仍由全部ID、fused/branch triplet监督；
没有补零min、伪造负例或回退到另一损失。若整个批没有有效行，执行assert失败。
两端每个正式训练batch的四项支持数都必须逐批等于已冻结元数据；
不能在新loader/RNG上默默改变有效行域。保留每步M0和每epoch Q1的两段loss、
active rows、合法行/缺失行统计。原始label-replay SHA:
5a42be65a512534bb87f52a5f3f4385042157511803774579e65d96d94662d31。

MCNL不直接对齐RGB/NI/TI方向，也不改变三模态拼接距离。
camera样本量不均匀、对负例相对次序施加的先验不一定有利，这是本次需要检验的
科学假设；不得以T0/M0通过宣布去除了camera信息或解决泛化。

## 完整比较预算与选择

| 条件 | 控制端 | 实验端 |
|---|---|---|
| endpoint | batch_hard_residual | camera_negative_residual |
| 三个residual metric | 原batch-hard triplet | MCNL |
| epoch / warmup | 20 / 5 | 20 / 5 |
| 每fold优化步 | 580 / 560 / 540 | 580 / 560 / 540 |
| 三fold优化步 | 1680 | 1680 |

共3fold×2end×20epoch，120条epoch记录、3360优化步和3360次模型前向/反传。
单卡远端RTX3090，seed42，B64/K8，无梯度累积；workers4；
原CrossCameraIdentitySampler和SharedGeometryTripletTransform；
AdamW LR0.00035、weight_decay0.0001；warmup5后cosine，总20epoch；
AMP FP16/GradScaler256、原gradient checkpointing、CuDNN deterministic=true，
benchmark=false；eval batch128、256×128、无reranking。
两端初始state、全部sample-order SHA、前8增强/索引/文件名严格配对。
每端只评最终checkpoint，保存后从原source重建并strict reload验证完整state SHA，
再评baseline/fused/CNN/Transformer/Mamba五输出。不中途评价择epoch或只取部分fold。
控制loss代数等于原V8目标，但浮点加法分组改变，不宣称与V20历史控制逐步相同。

## T0与M0

T0仅在远端CUDA运行3个数学契约测试：与独立逐行穷举公式一致，真实同ID跨相机
正例不混入负例且自身排除；观测到的缺失负例行域处理正确；
梯度有限非零、样本重排/尺度/正交旋转不变、FP16输入转FP32计算及梯度有限；
完整无效batch显式失败。无模型/图像/权重/项目优化器步。
T0不能替代真实B64训练资格。

M0顺序：三fold两端各8source batch进行真实forward-only配对（48批）；
fold0两端各8不同batch实际优化；fresh source实验端固定第一真实batch100步。
M0实际预算116项目optimizer steps/116前反传对，另48个forward-only。
所有203训练tensor在容量/过拟合阶段累计有限非零梯度、overflow0、
完整冻结state不变、capacity peak<24576MiB。
解析总loss下界F=0.75*H_94(smoothing0.1)=0.57838292104621，
MCNL非负下界0。固定门为第100次更新前与第1次更新前
(L100-F)/(L1-F)<=0.1，并通过上述全部工程检查。
记录全部100步，禁止改为中间最小值/末段均值或更换batch/steps/margin/seed救回。
M0权重不进入Q1；M0失败则保留完整M0_FAIL并停止Q1。
M0成功时原进程自动进入固定全部6个Q1端点；科学结论等终态完整结果和独立审计。

## Q1科学门与后续范围

以下五门全部满足才有下一阶段资格：
1. candidate aggregate fused mAP相对matched control至少+1.0pp。
2. 三个fold fused增益分别非负。
3. CNN/Transformer/Mamba三个aggregate mAP增益均非负。
4. 完整571query、21身份cluster、10000次seed42 bootstrap的95%下界严格>0。
5. candidate fused mAP严格超过同checkpoint baseline及三个固定expert输出。

即使先完成的fold已为负，仍执行剩余完整预算并保留全部结果，除非真实执行错误。
M0工程通过、metadata支持比例、source拟合或一折正值均不是科学门通过。
报告全部5输出×3fold的mAP/R1/R5/R10、全部21身份和571query变化、
6次完整训练/重载/冻结/数据次序及权重SHA绑定；不对相机或查询择优报告。

Q1失败封存该固定V22，不扫描margin、权重、epoch、LR、camera采样、种子或分支；
不进行D1/dev/official/消融。Q1成功后在首次dev访问前另行记录固定D1执行细节，
保持该完整目标与单seed/最终checkpoint规则，不再在OOF选择变体。
整体目标仍要求固定开发门65mAP和官方同协议85.3mAP/87.9R1目标；
当前best dev58.4050/59.3939不因本计划改变，没有SOTA完成声明。

## 时间与交付

GPU空闲>=22000MiB并核对代码/配置/方案/原始权重后执行。
T0约5–10秒，M0约4–7分钟；Q1参考V20完整70.6分钟，初估75–95分钟。
按M0终点、每个完整paired-fold窗口查看，后台诊断如需轮询间隔180–300秒，
由实际epoch速度修正ETA；不因SSH观察超时重启原进程。
保存原始JSON/log、六final checkpoint及standalone receipt、数组重算和独立审计；
代码及时推送GitHub，repo/远端/桌面master逐字节核验。
