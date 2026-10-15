# F1 起点的梯度与排序目标：原论文、作者代码和两项条件性改造

日期：2026-10-16。范围：只读本地结果与一手论文、作者仓库；只新增本笔记。未查询远端任务/GPU，未执行模型前向或训练，未修改任何运行中封存源码，未登记新候选。

## 1. 现在能解释到什么程度

RGBNT201 完整50轮的同一 best-E3 模型为 **77.2475/80.9809 mAP/R1**，纯角色银行为 **77.9332/81.8182**；相对完整 Signal 四项仍负。MSVR310 完整50轮的 best-E13 融合为 **54.3826/72.7580**，纯银行为 **53.1469/71.5736**，融合四项均超过本地完整 Signal。二者都是原始 OUR F1 初始化，不能与旧完整 Signal 起点结果合并。[201 原始验收](../f1_msvr_prepare_continuation_20261016_992/RGBNT201_FULL50_VERIFIED.json)、[MSVR 原始验收](../f1_msvr_full50_terminal_20261016_994/VERIFIED.json)

201 在第3轮选中最佳，后续正式分数较低，聚合训练 loss 下降。2498/2584 个 AP 阶段更新有合法支持，通常已有512个历史候选；因此“AP完全没有监督”与现有证据不符。其终点 auxiliary/rank 梯度 L2 EMA 比为 **575.89/449.38/395.53**；MSVR 为 **8.42/6.50/7.03**。这是不同任务、不同数据集中的描述统计，既不是梯度余弦，也不是 AdamW 实际参数更新份额，更不是对失败原因的因果证明。[同上两份验收]

现有机制先把当前排序梯度与历史候选 VJP 相加，再按 CNN/Transformer/Mamba 参数块计算范数和余弦；只用范数 EMA 的平方根比生成两个系数，系数和为2，最高 ranking 系数1.6、最低 auxiliary 系数0.4。它不改变两个向量的方向，也不优化损失的相对训练速度。`combine()` 已返回余弦，正式训练日志只保留权重，故目前没有这次完整 F1 训练的方向冲突频率证据。[当前平衡器](../../tools/msvr_supported_gradient_balance.py)、[F1 训练器](../../tools/train_f1_initialized_roles.py)

还应保留以下解释，不能只选“辅助梯度过大”：训练身份过拟合；固定 F1 对201银行的互补性较差；AP代理与真实排序的梯度差异；历史集合覆盖、正例难度或环境分布不同；多个辅助项内部抵消；AdamW 一、二阶矩改变原始梯度的几何关系。201 的纯银行优于融合、MSVR 的融合优于纯银行，只支持“互补性不同”的观察，不能单独证明任何一个原因。

## 2. 一手来源核查

以下仓库 SHA 来自本次 GitHub 官方 commits/tree API 的只读响应；没有下载或执行整个仓库。

### Smooth-AP：已经在用，不能重新包装成新方法

原文通过 sigmoid 近似排序；作者 `SmoothAP.forward()` 分别平滑正例内部排名和全部候选排名，默认按等数量、连续身份分组构建正例，`sigmoid()` 还夹紧指数。它没有本项目的同身份同环境忽略规则、可变历史候选和候选侧回放。[原论文](https://arxiv.org/abs/2007.12163)、[作者项目页](https://robots.ox.ac.uk/~vgg/research/smooth-ap/)、[固定作者代码，`src/Smooth_AP_loss.py::sigmoid/SmoothAP.forward`](https://github.com/Andrew-Brown1/Smooth_AP/blob/59927ec4c59565bff209301d2c2b449ed1937806/src/Smooth_AP_loss.py)

本项目实际用 `torch.sigmoid((s_j-s_p)/0.01)`，合法正例仅为跨 camera/scene 同身份，合法负例为所有不同身份，忽略同身份同环境；每个有支持 query 的权重相同。对 score 差分 t，导数为 `sigmoid(t/tau)*(1-sigmoid(t/tau))/tau`。大幅错误排序仍可能只有很小导数，但“aux/rank 范数比大”不足以证明实际错误关系处于这个区域。[本地真实目标](../../tools/msvr_cross_scene_smooth_ap.py)、[本地 TAU](../../tools/msvr_smooth_ap.py)

### ROADMAP / SupRank：改变排序的导数和跨集合分数含义

ROADMAP 区分两个问题：平滑正例排名可能产生不利的正例间导数，错误负例的大差值可能落入 sigmoid 小导数区；以及批内 AP 与全候选 AP 不可分解。前者使用精确正例排名及带线性尾部的负例排名代理，后者使用正负绝对分数校准。它们是不同干预，不能把历史候选不足直接等同于校准不足。[原文 §3.1–3.2、补充 A.1](https://arxiv.org/html/2110.01445)

作者固定代码 `step_rank()` 对正例比较使用 Heaviside，对错误负例使用分段 sigmoid/线性；`CalibrationLoss` 继承 PML ContrastiveLoss，生成身份 pairs 后调用库 reducer，并不支持本项目忽略集合。现有本地阅读已核对：公开配置固定 `offset=1.44` 会在分段点附近产生跳变，精确 tie 也有与论文约定不同之处；不能照抄并宣称严格上界。[作者排序实现](https://github.com/elias-ramzi/ROADMAP/blob/fed37d75f475f636542b2ccd1ffc0c3918498a57/roadmap/losses/smooth_rank_ap.py)、[校准实现](https://github.com/elias-ramzi/ROADMAP/blob/fed37d75f475f636542b2ccd1ffc0c3918498a57/roadmap/losses/calibration_loss.py)、[配置](https://github.com/elias-ramzi/ROADMAP/blob/fed37d75f475f636542b2ccd1ffc0c3918498a57/roadmap/config/loss/roadmap.yaml)、[已有项目阅读记录](../../evidence/roadmap_primary_read_20260920/READING_NOTE.md)

较新的作者 SupRank 实现支持不同 query/reference 集合，`SupRank` 的 rank-plus 仍是 Heaviside，rank-minus 保留线性错误尾部；`RankingLoss.forward()` 经 ranker 计算 AP/Recall 等目标。它仍从标签矩阵建立相关性，没有本项目的 camera/scene 忽略语义。代码存在不宜照搬的实现细节，例如 `SmoothRank` 的 rank-plus 选择用了恒真的字符串条件；应明确写出需要的数学目标，而非把通用类整体接入。[论文，TPAMI2025版本](https://arxiv.org/abs/2309.08250)、[作者说明](https://github.com/elias-ramzi/SupRank/blob/d3d0e3cbceee3e8f295fd411daecbe33c1b881bb/README.md)、[`tools/differentiable_rank.py::step_rank/SupRank/SmoothRank`](https://github.com/elias-ramzi/SupRank/blob/d3d0e3cbceee3e8f295fd411daecbe33c1b881bb/suprank/losses/tools/differentiable_rank.py)、[`ranking_loss.py::RankingLoss/compute_ap_loss`](https://github.com/elias-ramzi/SupRank/blob/d3d0e3cbceee3e8f295fd411daecbe33c1b881bb/suprank/losses/ranking_loss.py)

### PCGrad：只消除负内积成分，不自动解决量级差

PCGrad 以负内积识别方向冲突并作投影。作者 TensorFlow `PCGrad.compute_gradients()` 展平各任务梯度、只减去负投影并求和；它包装下游优化器，没有处理本项目的历史 VJP、无合法支持和角色专属分类头。代码中的 `tf.random.shuffle(loss)` 返回值未接回 `loss`，不能声称该固定代码严格实现论文的全部随机次序细节。若两个梯度同向，投影不发生，量级差本身不会触发修复。[原文](https://arxiv.org/abs/2001.06782)、[固定作者代码](https://github.com/tianheyu927/PCGrad/blob/c5fbd7c856526373828074f06875230f7f3ee79e/PCGrad_tf.py)

旧项目 M0 的24个正余弦也不是本次 F1 全程方向证据；不应据此证明本次无冲突，也不应忽略它而把 PCGrad 写成已证实的缺失机制。[已有梯度邻近工作记录](../../evidence/support_aware_gradient_neighbours_20260921/PRIMARY_SOURCE_NOTE.md)

### CAGrad 与 GradNorm：优化的问题和现在的平方根 EMA 不同

CAGrad 在平均梯度附近的球内改善最差任务的一阶下降量。论文的收敛分析不能直接移交给本项目的角色分块、带刷新历史和 AdamW 更新。作者 `nyuv2/utils.py::cagrad()` 用梯度 Gram 矩阵解 simplex 双问题，NYUv2 版本硬编码3任务，另有 rescale 分支；`toy.py::cagrad()` 是2任务版本，使用 Adam 做展示。仓库的 Adam 实验不等于一般 AdamW 收敛保证。[原文 §3.1–3.2](https://arxiv.org/html/2110.14048)、[固定作者 NYUv2 代码](https://github.com/Cranial-XIX/CAGrad/blob/dc3d48152b6196945cfd56144879b9d42353b095/nyuv2/utils.py)、[固定 toy 代码](https://github.com/Cranial-XIX/CAGrad/blob/dc3d48152b6196945cfd56144879b9d42353b095/toy.py)

GradNorm 还使用 `L_i(t)/L_i(0)` 的相对训练进展去学习权重，并在共享参数上匹配目标梯度范数；不是把当前 EMA 范数比直接开方。无合法支持时 AP 为0是缺监督，不能当作已经学会。此次未确认作者所有的公开 GradNorm 代码，第三方实现不作为作者实现证据。[原论文与算法1](https://proceedings.mlr.press/v80/chen18a/chen18a.pdf)、[出版页](https://proceedings.mlr.press/v80/chen18a.html)、[项目已有核查](../../evidence/support_aware_gradient_neighbours_20260921/PRIMARY_SOURCE_NOTE.md)

## 3. 至多两个具体改造；都需要当前证据触发

两项只改变训练目标或参数更新，不改变 CNN/Transformer/Mamba、F1 初始化、检索向量结构；不是学习率/batchsize 搜索。不建议同轮一起加入，也不承诺补齐201与 Signal 的差距。

### A. 在合法检索关系上分离“正例计数”与“错误负例修复”的 AP 学习

**触发证据：** 在训练身份的固定关系上，记录每个错误负例—合法正例差值 t、sigmoid 导数，以及 `dL/ds`。若错误关系数量仍多，而这些关系的导数已很弱，或存在提升错误负例/降低合法正例分数的导数方向，再考虑此项。支持 query 很多并不排除这种关系级问题；仅梯度范数比不够。

**定义：** 对同一 current+refreshed-history 候选集合，先删除同身份同环境候选。每个合法正例 p 的 `r_plus(p)=1+sum_{p'!=p} H(s_p'-s_p)` 作为停止梯度的正例计数；负例计数改为 `sum_n h(s_n-s_p)`，其中 h 在错误大差值方向保留非消失的线性尾部。每个 query 仍按其真实合法正例数平均，再对有支持 query 平均，不让身份正例数或历史容量偷偷改变 query 权重。**只替换排序代理，不同时加入绝对校准。**

**本项目改造点：** 合法性掩码先于排名；同一数学函数必须同时用于当前 encoder 反传、历史 leaf upstream 和历史候选 VJP，避免只修 query 侧。保留同环境不同身份的负例；无支持时不更新 ranking 状态。预先规定 tie 的确定性次序/半权规则及连续分段连接，不能把作者代码中的 tie/offset 选择当作已证明的项目约定。候选刷新、RNG/BN 状态和梯度回放沿用原规则。

**与现有平衡器的差别：** 现有平衡器只缩放整个 AP 向量；此项改变哪些合法关系产生梯度、梯度朝什么分数方向。它不能修复缺乏互补信息、训练身份过拟合或尚未进入候选集合的错误。它更不是 Recall@1 目标，不能自动解决首位检索。

**归属：** 精确正例排名和线性尾部来自 ROADMAP/SupRank。上述合法关系与历史双侧导数一致性是项目化适配，单凭组合不能主张原创模块；至少要与普通合法 Smooth-AP、直接合法 SupAP 控制区分。这一方向已有项目阅读，不能再次称为“从未研究的新 idea”。

**同一候选内的简化数学选项，尚未选定：** `h(t)=softplus(t/tau)/ln(2)` 满足 `h(0)=1`，对 `t>=0` 不小于1、对 `t<0` 大于0，因此可包住 step，并避免显式 delta/rho/offset。错误方向的 `h'(t)=sigmoid(t/tau)/(tau*ln(2))` 没有指数平尾；正例计数若采用固定候选顺序处理 ties，tie 负例取上界仍保守。但 logistic/log2 上界早已是排序文献的标准工具，不能把这一公式称为原创；有限检索未确认相同 AP 分式的首次使用，也不构成新颖性证据。[微软作者技术报告 §4.2，PDF第7页](https://www.microsoft.com/en-us/research/wp-content/uploads/2010/12/TechReport_DirectOptimize.pdf)

更关键的是，`h'` 不消失不等于整个 AP 梯度恒强：单正例项 `1-r/(r+sum h)` 的导数含 `r/(r+sum h)^2`，候选多或违例很大时仍会衰减。单调 score-space 导数也不等于共享参数上的多个目标不冲突。上述简化只有实现清楚、tie/连续性简单的优势；它可能产生更松的上界和不同梯度尺度，尚无证据比作者分段函数更适合本项目。

### B. 只在真实 AdamW 更新损害合法 ranking 时，作最小参数步修正

**触发证据：** 除原始余弦外，记录完整排序梯度 `g_R=g_current+g_history` 与实际 AdamW 参数增量 `delta_0=theta_next-theta` 的内积；若 `g_R dot delta_0>0` 经常发生，才说明当前步的一阶方向会增加这个训练排序目标。若内积通常为负，则这一机制没有已证明的干预对象，不应用“辅助范数大”代替触发条件。

**定义：** 保留现有损失、系数及 AdamW 状态更新，取得其实际提出的角色参数增量（包括一二阶矩与 weight decay），在有合法支持的角色块上求

`delta_star = argmin_delta ||delta-delta_0||², subject to g_R dot delta <= 0`。

当约束已满足，增量原样采用；违反时的闭式结果为 `delta_0 - (g_R dot delta_0)/||g_R||² * g_R`。分类头采用原更新；完整 F1 继续冻结。这里描述的是待实现的训练机制，不是已修改优化器。

**本项目改造点：** 约束对象是含历史候选侧梯度的合法检索目标，且约束实际参数步，而非在 AdamW 前对两个原始向量做对称 PCGrad；不能只使用当前 batch 的 ranking 梯度。AMP 必须在未缩放单位下比较；优化器 moments 每步只更新一次，实际增量与记录值一致。与按角色拼接不同的几何规则相比，应明确这是三个角色参数块上的约束，不宣称对每个辅助子项都有共同下降性质。

**限制：** 只保证当前候选集合的一阶 ranking 变化非正，等号可能停滞；不保证有限步真实 loss 下降、测试 mAP/R1、辅助目标改善或 AdamW 收敛。若大辅助更新已经有利于 ranking，该机制不会为了范数相等而削弱它；也无法消除不利的正交变化和过拟合。参数步投影可能与保留的 moments 产生后续不一致，需要如实记录触发频率、修正幅度和训练时间。

**归属：** 受 PCGrad 的半空间投影及 CAGrad 的一阶改进视角启发；把约束移到实际优化器步及历史合法梯度，是可检查的实现区别，仍不足以证明新颖性。没有当前 F1 方向证据之前，优先级低于记录诊断，不能先把它写作201失败的解法。

## 4. 选择与验证边界

先用**同一次后续正式训练**保留已计算但丢弃的每角色 `||g_R||/||g_A||/cos(g_R,g_A)`、分项 loss、score 关系导数，以及 `g_R dot delta`；不要为观察这些统计另外铺开多条 NN 队列。若需要短工程探针，它只证明目标/梯度路径能正确运行，不用8步决定科研收益。任何正式候选仍按用户规则跑完三集50轮，以一份完整 checkpoint 的全部指标验收。

不把“再做归一化分类”列作第三项：`NormalizedRoleClassification` 已实现并有历史实验，特征归一化与上述两项不同，也不是新的未试方向。[现有实现](../../modeling/trifusion/normalized_role_classification.py)

ROADMAP 的绝对校准是可用的邻近证据，但目前没有实际分解差距或训练分数漂移证据，暂不叠加。原文的校准阈值、PML reducer、512历史候选和本项目13项辅助监督共同作用时，不能直接套用原文理论或默认参数。

建议先判断两种可证伪事件：**关系级导数失效**对应 A；**实际更新对合法排名上升**对应 B。二者都未出现时，应回到互补性、训练泛化和初始化能力差距，而不是从相同范数现象继续制造新的平衡系数。当前最强结论是需要补齐测量，不能声称已经找到201失败的原因。
