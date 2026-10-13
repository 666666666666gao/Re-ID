# 新增证据的训练用途与最终比较空间

**本次得到一个有条件的源码／数学结论，没有新增训练或检索成绩：当前角色任务的训练态BN分类与raw欧氏Triplet，对同一批全部融合特征的共同平移不敏感；最终L2后余弦比较却可能改变。**这能说明“任务损失变好”不足以保证“实际比较中的新增变化受到约束”，不能据此认定当前common correction是噪声或失分原因。

## 1. 先核实真实输入和模式

本次只读核查安装在26的Signal四个文件，与原封存SHA及作者完整commit `cd1b0a672d1fe642e7608731cb4899a19dda7d51`完全一致；公开源码与installed bytes也完全一致。当前九个相关项目源码与§930输入seal一致。详见SOURCE_INVOCATION.json、AUTHOR_SOURCE_PINS.json、SIGNAL_INSTALLED_PUBLIC_COMPARISON.json；不以仓库根目录同名layers代替实际comparators实现。

- `AuthorHeadEvidence.train(mode)`恢复Signal作者头的训练模式；训练入口调用`model.train()`。冻结权重与BN是否使用当前batch统计是两件事。
- `GlobalTaskRoleHeads`先对global调用正常头；角色通过`functional_call`使用停止参数梯度的当前头及克隆BN buffer，继承训练模式。角色调用产生的buffer变化不写回共享原buffer。
- 实际融合训练特征为 \(h_i=\operatorname{sg}(g_i)+\gamma c_i\)。角色CE接收BN后的分类logits；角色Triplet接收BN前的raw \(h_i\)，不是部署的单位向量。
- 现存201/seed42两端绑定为`softmax_triplet`，ID权重0.25、Triplet权重1，label smoothing开启、`NO_MARGIN=true`，没有center项加入这条返回loss。global任务和角色任务相加；梯度职责分开。
- 部署使用 \(f_i=h_i/\|h_i\|\)；不使用上述分类BN做检索。training-only exposure VJP未增加新的角色头或loss。

这些是当前源码和既存封存绑定的核对，**本次没有重新执行PyTorch forward、训练模式探针或自动微分**。安装作者源码及raw距离路径见[Signal作者头](https://github.com/010129/Signal/blob/cd1b0a672d1fe642e7608731cb4899a19dda7d51/modeling/make_model.py)、[作者损失组装](https://github.com/010129/Signal/blob/cd1b0a672d1fe642e7608731cb4899a19dda7d51/layers/make_loss.py)、[作者Triplet](https://github.com/010129/Signal/blob/cd1b0a672d1fe642e7608731cb4899a19dda7d51/layers/triplet_loss.py)。

## 2. 一个任务项不可见、部署可能可见的变化

固定global、scalar gain和头参数，假设同一BN统计组、同一Triplet比较集合中的每个融合特征都改为 \(h_i+b\)。在理想实数计算中：

\[
\mu'_B=\mu_B+b,\quad v'_B=v_B,\quad
\operatorname{BN}_B(h_i+b)=\operatorname{BN}_B(h_i),
\]

\[
\|(h_i+b)-(h_j+b)\|_2=\|h_i-h_j\|_2.
\]

所以当前batch分类logits、普通或平滑CE、raw距离、hard-mined关系及soft-margin Triplet不变。其基础是训练BN使用batch统计，[PyTorch官方说明](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm1d.html)支持这一操作定义；平移不变性本身是本报告的代数推导。

部署方向的微分则为：

\[
\delta f_i=\frac{1}{\|h_i\|}(I-f_if_i^T)b.
\]

若改写为 \(c_i\mapsto c_i+b_c\)，上式右侧再乘固定gain \(\gamma\)。不同样本的方向响应一般不同，因而query与gallery余弦排序可能变化。

这是**输出特征层面的任务对称性**，不是证明存在可独立移动的参数平坦方向。现有结构的LayerNorm、bias-free读出、共享参数、全局任务、gain及Adam参数weight decay仍限制可达变化。在同一统计组或Triplet比较集合内，随样本、身份或模态变化的非共同平移，一般不能按上述方式抵消。平移量可以随批次变化，只要在当前组／比较集合内仍为共同平移；raw Triplet改为normalized也会改变条件。

固定运行统计的eval-BN亦不满足任意平移不变性。设 \(D=\operatorname{diag}(\gamma_{BN}/\sqrt{v_{run}+\epsilon})\)，固定classifier为W；分类概率不变需 \(WDb\in\operatorname{span}(\mathbf1)\)。不能把train-BN结论无条件搬到eval-BN。

## 3. 已执行的纯数学数值例证

工具`tools/analyze_role_objective_translation.py`只使用标准库，对四个手工二维向量及共同平移[0,2]计算BN公式、平滑CE、batch-hard soft-margin Triplet和余弦。**不是ReID样本，没有camera过滤、模型参数、权重、数据集或优化器；不重放作者PyTorch/CUDA计算。**预先写入的例证容差为1e-12，不修改历史梯度或重载容差。

| 量 | 平移前 | 平移后／最大差 |
|---|---:|---:|
| 训练BN最大绝对差 | — | 6.66133814775e-16 |
| 训练logits最大绝对差 | — | 5.55111512313e-16 |
| 平滑CE | 2.007612421628 | 2.007612421628 |
| raw距离最大绝对差 | — | 2.22044604925e-16 |
| raw soft-margin Triplet | 0.572461669284 | 0.572461669284 |
| 同一个toy query的正例名次 | 3 | 1 |
| 固定原统计的eval-BN CE | 2.007612421628 | 3.703510363541 |

hard-mined正负索引不变，训练CE差4.4408920985e-16、Triplet差0。逆向平移也能将正例从1变为3，故平移并非自动修复。共同变化在raw样本差中没有增加信息，却可以改变所用度量的结果；**用途变化不能自动解释为新增身份内容**。原向量、全部分数、索引和限制保存在MATHEMATICAL_WITNESS.json。

## 4. 外部机制只回答一个环节

本次外部方法仅用于理解“训练空间怎样连接实际比较”，没有移植：

| 来源 | 它回答的环节 | 在本项目的边界 |
|---|---|---|
| BNNeck论文与作者代码 | 明确区分raw度量特征、BN分类特征，并选择检索空间 | 不能推导TriFusion改用BN检索或中心化必然有效 |
| NormFace论文与作者代码 | 对训练与测试的角度空间差异，使用归一化feature/weight及缩放余弦分类 | face verification先例，不是当前冻结global＋角色修正结构的验证，不提供预期涨分 |

[BNNeck论文](https://arxiv.org/pdf/1906.08332)、[BNNeck作者实现](https://github.com/michuanhaohao/reid-strong-baseline/blob/3da7e6f03164a92e696cb6da059b1cd771b0346d/modeling/baseline.py)、[NormFace论文](https://arxiv.org/pdf/1704.06369)、[NormFace作者实现](https://github.com/happynear/NormFace/blob/8438887d33023bf35f16ba5e8c21a422c49cd520/prototxt/scaled_cosine_softmax.prototxt)。详见PRIMARY_SOURCE_RESEARCH.md及manifest。当前查阅的官方文档版本不构成本项目安装版本或运行行为的attestation。

## 5. 对下一项研究决策的具体影响

研究主线保持：“每一条新增证据，在最终身份比较中确实有用”。下一项设计必须说明实际监督能看见哪些部署变化，而不能只凭分支独立mAP、辅助loss、活动梯度或correction范数判断用途。换成独立可学习头，如果仍采用同样train-BN＋raw Triplet，也不会仅因“独立”消除上述输出对称性。

§931的common能量统计不是训练batch平移统计，也不证明噪声或当前失分原因；§932的margin分解不是AP；§933固定读出控制的微小变化不证明压缩根因、无须空间信息或整个角色无效。上述条件结论不能合并成一个已定位根因。现有F2/F3与joint-L2负／混合结果继续有效，**不能再次把“全部归一化”当通用修复**。

本次不按测试均值中心化、不启动新head/loss/gain/seed搜索、不添加模块或改训练代码。若后续恢复方法研究，需针对实际融合比较的用途建立控制，同时保留global净收益、多正例排序和新增损伤检查。全局角色分类loss变好并非其替代证据。

本次0新NN、0训练、0M0、0优化器更新、0新权重／权重删除；只读installed source与标准库例证完成。旧best、输入、§933区域／距离及原失败保留。源研究代理是独立上下文读取，model/effort运行身份未独立attest；例证由主代理执行与自检，不冒称独立模型复现或experiment-audit。训练STOP，三数据集超过Signal的Goal仍ACTIVE_UNMET。
