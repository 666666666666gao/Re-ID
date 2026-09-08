# MSVR310 原三角色架构独立训练比较合同 v1

登记时间：2026-09-06T06:03:04.455453+08:00。状态 PREPARED_NOT_RUN。
Signal B0三折基线已经完整训练、封存和独立审计：WARN / engineering PASS，trace run17两轮闭合。
本合同是最新跨数据集研究路线中的一个主模型比较，不是结构消融、RGBNT201失败版本重跑或官方评估。
RGBNT201 dev65与官方85.3/87.9主目标继续保留；本合同的任何结果不解除其现有晋级限制。

## 主张、对照及边界

唯一主张：相对各自source训练的完整Signal，原CNN/Transformer/Mamba残差银行能否在
MSVR310的新身份和完整干扰图库中带来跨身份分布的检索改善。
现有RGBNT201反复开发的21个query身份不能独立支持跨数据集机制；MSVR310固定协议有60个query身份。
本实验从每折各自车辆Signal出发，新初始化三角色和七个分类头；
不把RGBNT201训练权重直接部署到车辆，因此不是零样本域迁移。

最小主比较表保留Signal / full CNN / full Transformer / full Mamba / fused五个输出，
都来自对应最终同一checkpoint；完整分支是Signal与该角色残差组合，不是残差单独检索。
基线直接使用B0的固定epoch50，不重训、不选择其它epoch；推理时必须逐元素复现其原3072D特征及距离。
不读取V8 Phase-B Router、V23/V24或其他fold角色checkpoint；不继承其失败干预。
本比较能判断整个三角色系统是否在新数据集有增益，不能排除额外参数/计算的解释。
容量、单/双角色、融合、效率与多seed消融保持主结果后再做，不在此运行。
暂不声称新结构、frontier primitive优势、官方复现或多seed稳定性。

## 固定输入与模型

基线固定配置configs/MSVR310/Signal-source-oof-v1.json及原源代码保持原字节，
SHA ebf767f8c704c9886c80335014e571838400e733ce40665847cd1a9ea6783278。
原B0总JSON SHA22a4f3642e88088a8dfcb4acddb610d6566c12d6b765cb7b344b28acdbcea6eb，
三个Signal checkpoint及原始retrieval_arrays由该JSON的完整SHA分别绑定并在入口核对。
原Signal实际17份源码、git commit/diff、CLIP整个权重SHA由B0 configure原封核对。
新入口另绑定全部直接使用的项目源码和本合同SHA。

使用原signal_preserving_collaborative_v8_expert_formation构造，不修改模型实现：
semantic768、feature512、adapter128、每角色每模态512；
block8后分叉，三条路径分别执行共享冻结tail9/10/11，每阶段加入原角色算子；
scale_init0.05，gradient checkpointing开启，三条路径全部执行。
CNN四个水平分区只称空间汇聚；车辆输入高128/宽256，Patch grid8×16。
全部Signal参数、camera SIE及预训练tail冻结；只有原角色模块、BN可训练项和分类头训练。
输出baseline3072D、各完整分支4608D、残差银行4608D、fused7680D，
固定等能量拼接：最终相似度一半来自Signal、一半来自三角色残差。
这没有融合不下降的数学保证；模型仍保留独立baseline输出。
不启用Router、HFER、V23模态MLP、投影、原型记忆、风格混合、蒸馏或新损失。

每折从seed42新初始化角色与分类头，加载且只加载本fold B0 Signal。
构造后保存整个初始化状态SHA；正式比较必须与该fold M0初始化SHA一致，
但不加载M0训练后权重。

## 数据和评价合同

继续使用protocols/msvr310_train_oof_v1.json的原字节：
SHA4ff4c60bca3d019929add5788212c526387d93d535a2c52aa7b1c3acfd387cb4。
只读官方训练目录bounding_box_train；source103/103/104身份、672/683/709记录，
每fold40跨scene source身份；heldout完整gallery360/349/323，query210/207/183。
总600query、60query身份、1032gallery、155heldout身份；
95个单scene身份的432条记录作为干扰保留，不按query身份裁剪gallery。
source与heldout完整身份隔离，训练连续标签来自原fold source_label_map。
原camera0–7用于SIE；scene使用原真实值，仅移除同身份且同scene的gallery。
不同身份同scene样本仍是合法负例；不套用RGBNT201 same-camera过滤。
不跨fold计算特征距离，不混入RGBNT201固定dev或任何官方测试数据。

训练沿用B0 loader、原RandomIdentitySampler、三模态同步几何和独立擦除，
真实B64/K8、workers4，记录每一步原始global record indices，以实际batch数计成本。
clean评估B64，原resize/normalize不变；无增强、reranking、缺失模态或test-time adaptation。
每fold在固定epoch20严格重载后一次性前向全部gallery，query从同次特征按原位置读取。
先逐元素比较baseline与B0的原始features以及重算distance，再评价全部五输出。
对每个输出保留完整特征、平方欧氏距离、gallery排序索引、逐query AP/首正例rank、
原作者eval_func_msrv交叉核对及独立目录中的re.txt。
每fold模型各自L2归一化后平方欧氏、NumPy默认argsort、原scene掩码。
总指标把600query合并加权，不简单平均三fold的mAP。

## 优化与成本

复用原V8 ExpertFormationV8Criterion、weighted_training_loss和learning_rate_multiplier；
总计七组ID/Triplet目标：
fused ID0.25/Triplet1；各完整branch ID1/12/Triplet0.25；
各residual ID1/12/Triplet0.25，margin0.3，label smoothing0.1。
不改变目标的求和实现或损失权重。
AdamW LR0.00035、weight_decay0.0001，AMP初始scale256，seed42。
正式每fold完整20epoch；原5epoch warmup/cosine实现保持（epoch<=5用epoch/5，
其余progress=(epoch-5-1)/(20-5)），不把最后epoch改成额外零LR端点。
容量和固定过拟合使用已有M0惯例的恒定基础LR。

预计每epoch13真实batch，即三折60epoch/780角色更新；以实际记录为准，
不以sampler的估计长度冒充真正批次数。
B0前置成本1950主干训练更新另计；这不是与Signal总训练预算匹配的比较。
M0三折各8步，加新fold0固定100步，共124更新/7936训练记录曝光。
clean source核验每fold24个三角色记录前向+8个独立Signal记录前向，
总72个三角色记录前向+24个独立Signal记录前向；M0 heldout/dev/official前向0。
以上按三模态记录计，一条记录含三幅模态图像；共享tail的多路径和checkpoint重算成本不省略叙述。
初步估算M03–8分钟，正式三折角色训练20–40分钟，依据实际M0计时再更新预计结束时间。
每次启动前free GPU>=22000MiB、无其它训练进程，远端RTX3090执行；不引入新服务或依赖。
长任务按180–300秒以上阶段检查，优先在预计结束前几分钟核对，不按epoch频繁轮询。

## T0、M0及正式比较执行顺序

1. T0只复用B0已通过的两个scene回归及1032真实标签掩码核验，
   因为调用的scene_scores、协议和测试文件均保持原SHA；不重复已有无变化的测试。
   本地仅AST与源码/JSON/hash检查，模型、张量、图像运行全在远端。
2. M0在三fold分别新初始化，先用8条clean source核对独立Signal与baseline前缀。
   三fold各8步真实B64/K8容量；记录所有可训练参数/非零梯度、原损失分量、索引、AMP及峰值显存。
   训练记录先落盘再判断工程门，保留先前B0 R1失败未保存逐步记录的教训。
   完整Signal及所有冻结参数state SHA不能改变，可训练状态必须发生更新；
   所有声明可训练张量在该阶段有有限非零梯度，overflow0，峰值reserved<24GiB。
   保存独立M0 checkpoint后严格重载，8条source的五输出逐元素相同。
3. 三fold容量均通过后，重新构造fold0初始模型，对一个固定增强B64/K8 source batch训练恰好100步。
   采用原七头label-smoothing解析熵下界，(final-floor)/(initial-floor)<=0.1；
   最后一步为唯一终点，不选择中间最小loss、不延长步数。相同冻结/梯度/AMP门也须通过。
4. M0完整回执PASS且配置/runner SHA一致，执行侧检查完整原数组与成本后，
   正式三fold分别重新初始化，训练固定20epoch/最终严格重载/一次完整gallery评估。
   工程失败先封存，不自动修改门槛、宽度、LR、seed、步数或重新抽batch；
   科学条件失败正常保留全部终态，无后续调参重跑。
5. 全部终态的整个checkpoint/数组/hash与全量排序、loss、采样算术核验后做独立experiment-audit。
   模型权重、特征、距离和图像始终留远端；本地保存JSON/文本、离散完整排名及SHA。
   这类审计边界不支持独立图像/二进制复现，不以同家族审阅宣称跨家族证明。

## 固定的跨数据集支持条件

以下五项须全部满足才写本内部协议SUPPORT_PASS，不能改变已封存RGBNT201版本的任何判定：

1. 全600query加权fused相对同checkpoint baseline mAP增益至少1个百分点。
2. 三fold各自fused增益均非负。
3. CNN/Transformer/Mamba三个完整分支的加权mAP均不低于baseline。
4. 用60真实query身份为cluster，保留每个身份query权重，seed42/10000次有放回bootstrap，
   2.5%线性分位的增益下界严格大于0。
5. fused的加权mAP严格超过baseline和全部三个完整分支。

不把上述门当作因果分解或SOTA条件。无论通过与否，报告全部fold/五输出/60身份、
AP改善/下降/不变、Rank-1修复/新增错误，以及参数、训练更新、记录曝光、显存与实际计时。
若只有少数身份贡献收益，同样呈现，不隐藏负贡献；不选单fold、epoch、seed或新的query子集。
本合同满足也仅支持单seed的内部跨数据集比较；官方主表、RGBNT100独立复现、
多seed与任何消融仍需相应后续固定合同及原主目标边界。

## 当前状态与优先级

Must-run：当前仅本合同M0，随后以其PASS为必要条件的三折固定终点比较和终态审计。
Nice-to-have：RGBNT100、容量/结构消融、效率与模态鲁棒性均不属于本入口。
最高风险是原三角色残差在未知车辆身份中仍不能提供稳定排序增益；
RGBNT201多轮负证据与Signal原有监督饱和说明这是真实科学不确定性，不能以梯度通过消除。
先保留固定Signal、完整干扰图库和全部负结果，直接测量该不确定性；不为它加fallback。

## 启动前源码字节绑定修订 R2（2026-09-06T06:08:14.672010+08:00）

148f5a7首次启动尝试在创建run目录、wrapper或训练进程前被严格源码SHA检查阻止；
原异常AssertionError: modeling/trifusion/criterion.py，模型/张量/图像调用及优化更新均0。
全量19输入核对查得5个历史文件本地CRLF、远端LF：criterion.py、state.py、builder.py、
experts/mamba.py、experts/semantic_residual.py。五份远端字节均等于Git原blob，
本地只转换CRLF到LF后字节相同，AST也一致；不是运行源码改变。

R2仅将这5项配置绑定改为逐文件核得的实际远端SHA，并保存原配置和远端文本证据。
入口继续严格逐文件比较实际SHA，不加入行尾自动归一化、fallback或宽松比较分支。
runner、模型实现、所有训练/数据/评估/门槛参数不变；首次计划和注册记录不覆盖。
新输出目录在修订提交之后才建立，M0及正式比较仍NOT_RUN。
诊断evidence/trifusion_msvr310_trifusion_v1_prelaunch_source_bytes_20260906.json，
原配置CONFIG_PRELAUNCH_R1_20260906.json。
