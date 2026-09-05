# V21 SAM 主实验方案（2026-09-05，执行前冻结）

## 单一主假设及来源

V19 全六端 source/heldout 诊断显示 source 七路分类、五路检索均100%，
heldout 存在明显落差。V20 完整三折跨模态身份监督主比较及独立审计已结束：
fused -1.010871 mAP、CNN -1.009739、Transformer -4.779791、Mamba +1.306367，
四个固定科学门失败。V20 封存，不扫描温度/权重/分支/epoch或重训。

新假设：保留原V8表示和原七路身份/Triplet目标，采用参数邻域中的梯度更新，
能否在规定前向/反向计算预算下改善未参与该fold完整训练路径的身份检索。
这检验优化目标对泛化的影响，不声称已证明“尖锐极小值”是唯一失败原因。
不将SAM本身描述为新算法，不用source拟合准确率代替heldout检索。

原始作者论文与固定源码来源见
docs/SAM_SOURCE_AND_DESIGN_NOTES_2026-09-05.md。
作者仓库commit dae9904c4cf3a57a304f7b04cecffe371679c702，Apache2.0。
本PyTorch代码独立按公式编写，使用当前AdamW为基础优化器；不同于作者
SGD/RMSProp和损失内L2形式，weight decay继续由AdamW在实际更新时施加。

## 固定模型、数据与推理

两端均从对应V12 fold的94-source Signal与V8 expert checkpoint严格恢复；
配置固定V12汇总、六source权重、CLIP权重SHA，不使用V19/V20最终训练权重。
完整3072D Signal direct+SIM、camera SIE和CLIP共享尾部冻结；索引8后的
原CNN局部、Transformer全局、Mamba空间/跨模态角色模块保持。
每专家每模态512D；fused为3072D Signal + 4608D等能量残差银行，
单专家为3072+1536D。模型expected98,800,141总参数、7,841,292训练参数、
203训练tensor，实际builder必须给出两端精确一致的绑定。

无新推理参数、私有尾部、Router/HFER、投影、跨模态辅助损失、外部backbone、
文本、mask或混合数据集。baseline-only/fused/CNN/Transformer/Mamba五路
全部保留。推理无需SAM或optimizer状态，checkpoint仍保存原非baseline参数
和BN buffer，完整冻结Signal通过原始权重SHA绑定。

数据、增强、sampler均沿用固定141-fit registry的3126 triplet，每fold94-source/
47-heldout。保留3126 gallery全部141身份和单camera干扰项；合法query571条
来自21跨camera身份，2555条只从query排除。三折gallery/query为
1000/190、1051/179、1075/202，不能混跨fold距离。OOF已跨版本开发复用，
本次不是新的独立验证；D1/dev/official访问为0。

## 优化公式与状态语义

原数据损失L为七路smoothed ID + batch-hard Triplet：
fused ID0.25/Triplet1；三branch各ID1/12/Triplet0.25；
三residual各ID1/12/Triplet0.25，身份总系数0.75。
margin0.3、label smoothing0.1。SAM不加入额外对齐项或正则损失权重。

普通对照rho=0，每个batch只做一次前向/反传，直接AdamW更新。
实验端rho=0.05，所有原requires_grad=true参数共同使用一个全局L2范数：

    g = grad L(w)
    epsilon = 0.05 * g / ||g||_2
    g_sam = grad L(w + epsilon)
    w_new = AdamW_update(original w, g_sam)

两遍复用同一已经增强的真实batch，不新采样。第一遍的共同AMP scale在
单位梯度方向中消去；第二遍反传之后只调用一次unscale_，然后step/update。
不访问GradScaler内部状态，不保留第二阶梯度。两遍所有训练参数的梯度均须
存在、有限；跨完整容量/正式训练预算累计覆盖全部203个训练tensor。

原网络七个训练BatchNorm1d在两遍前向都会更新状态，所以保存第一遍后的
running_mean/running_var/num_batches_tracked；第二遍反传后精确恢复。
每实际优化步七个计数器各增加1，SAM端不能增加2。训练仍使用batch统计。
参数扰动前保存训练参数副本，第二遍反传后copy_恢复并核对精确相等，
再以第二遍梯度更新原参数。普通端恢复次数0；SAM端每步恢复一次。
冻结Signal及共享尾部全state必须保持；不对冻结参数施加扰动。

## 完整主比较预算

| 项目 | ordinary AdamW 对照 | SAM 实验端 |
|---|---|---|
| 端点名 | adamw_40 | sam_20 |
| epoch | 40 | 20 |
| 每batch前向/反传对 | 1 | 2 |
| 每fold优化步 | 1160/1120/1080 | 580/560/540 |
| 三fold优化步合计 | 3360 | 1680 |
| 每fold前向/反传对 | 1160/1120/1080 | 1160/1120/1080 |
| 三fold前向/反传对合计 | 3360 | 3360 |
| warmup epoch | 10 | 5 |
| rho | 0 | 0.05 |

总计5040个Q1优化步、6720对前向/反传、180条epoch记录。
匹配的是前向/反传次数，不是wall time、epoch、优化步数或数据暴露次数。
普通端遍历数据两倍；不人为让普通端执行无用的第一次反传。
这比较固定计算预算下的完整训练方案，不唯一分离SAM与不同更新/数据暴露的
因果贡献。不能把40epoch对照解释成V20重训或从多个epoch中择优的扫描。

两端seed42，单卡远端RTX3090；B64/K8，无梯度累积，workers4；
CrossCameraIdentitySampler、SharedGeometryTripletTransform不变；
AdamW LR0.00035、weight_decay0.0001；warmup后按各自完整预算cosine；
AMP FP16、GradScaler初始256、原gradient checkpointing；
CuDNN deterministic=true、benchmark=false；eval batch128、256×128原预处理，
无reranking。两端初始state/前8增强精确配对；SAM全部20epoch的sample-order
必须等于对照前20epoch，逐epoch SHA与prefix SHA都核对。
不能声称40/20的完整采样SHA相同。各端只评价最终checkpoint，对照epoch20
不做检索、不中途挑选。所有三fold、五输出、21身份和query变化都报告。

## T0 / M0 工程门

T0仅在远端CUDA执行三项数学测试，不读取数据集：
1. 二维已知二次型的SAM更新与解析扰动点梯度一致，实际扰动范数匹配0.05。
2. rho=0等于一次真实AdamW更新；共同AMP scale128/256不改变SAM结果。
3. 同一AMP前向下BN最终统计精确等于仅第一遍的统计，计数只增1，
   冻结参数不改变、无梯度。

M0顺序固定：
1. 三fold两端各8个source真实batch，初始state、五路输出SHA、增强回执、
   source绑定精确配对；B64/K8真实同身份正例、cross-camera正例存在，
   baseline前缀精确，eval state不变。
2. fold0两端各8个不同batch实际优化步，记录一/两遍次数及两遍非零梯度覆盖。
   所有203训练tensor在8步内均须有有限非零梯度；overflow0、
   七BN每步只更新一次、冻结state不变；各端peak reserved<24576MiB。
3. 从对应source重新构建SAM实验端，固定同一真实batch，base LR下100步，
   记录每步原参数点loss、用于更新梯度点loss、梯度范数、实际扰动范数。
   解析下界F=0.75H，其中H为94类、smoothing0.1的交叉熵熵下界。
   固定门为第100步更新前与第1步更新前
   (L100(w)-F)/(L1(w)-F)<=0.1，同时两遍全训练tensor覆盖、overflow0、
   BN计数/恢复、参数恢复及冻结state检查全部通过。
   100步包含200对前向/反传；不将其计入Q1的5040/6720预算。

M0只做train-source工程资格。负结果不通过换rho/LR/batch/RNG、放宽阈值
或重新运行救回；真实实现错误按原始错误回执处理。
M0失败则保存M0_FAIL并停止Q1。M0权重不进入正式训练；
M0全通过后原进程才自动进入三fold两端完整主比较。

## Q1 固定科学条件与终态证据

六端训练完毕各自保存最终checkpoint，新建模型严格重载，
完整state SHA与训练终态一致，再全量评价对应heldout。
source权重与最终权重不能被评价改写，原日志/六receipt/完整summary保留。
每fold两端baseline输出对象、初始state、前8增强及前20epoch采样配对精确。
三fold每端前向/反传次数相同；对照优化步恰为SAM两倍。

全部五个条件必须同时通过：
- SAM相对本次实际ordinary AdamW对照fused aggregate mAP增益>=1.0个百分点；
- 三个fold各自fused mAP增益均>=0；
- CNN、Transformer、Mamba各自aggregate mAP增益均>=0；
- 以真实身份聚类，paired fused AP差10000次seed42/float64 bootstrap，
  95%区间下界>0；
- SAM fused aggregate mAP严格高于同checkpoint baseline及三专家。

无论中途某折正负，都完成全部六端后作Q1终态结论。不删负身份、不换分支，
不扫描rho/epoch/LR或新增种子。失败封存、不执行D1/dev/official。
若全通过，另行冻结141-fit refit及一次30-dev完整终态合同，本runner不执行。
终态后核验全部源/终态权重SHA、六receipt与summary、完整query/gallery masks、
30组fold-output指标、全21身份结果和bootstrap；再做独立实验完整性审计。

## 执行、成本与目标状态

运行代码tools/train_signal_preserving_v21.py；
数学步实现modeling/trifusion/sam_training_v21.py；
配置configs/RGBNT201/TriFusion-signal-preserving-v21-sam-rtx3090.yml。
执行前发布源码、方案及配置，外部传入方案/配置SHA并校验，
记录实际HEAD、runner与依赖源码SHA。复用V20的source-only builder和AdamW
构造函数，不调用其cross-modal loss、训练或旧checkpoint。

启动前远端GPU须>=22000MiB空闲。六个原V8非baseline权重预计约0.2GB，
在当前约8GB剩余磁盘内；不删除旧证据。T0约数秒；
M0预计4–8分钟；按V20实际70.6分钟与双倍训练计算量，Q1预计130–160分钟。
取得稳定epoch时长后修正ETA，按完整端点里程碑或180–300秒间隔查询，
不中途做heldout检索诊断。全部模型操作限远端，本地仅文本/AST/JSON/算术。

当前V20独立终态工程PASS、完整性WARN、科学FAIL，不能以WARN替代科学失败。
可部署dev最佳仍V8 Phase-B58.4050mAP/59.3939R1，exact Signal58.0109/57.4545；
dev65和官方85.3/87.9目标未达。RGBNT100/MSVR310只完成数据/协议安装核对。
此方案尚无T0/M0/Q1结果，用户总体目标继续active。
