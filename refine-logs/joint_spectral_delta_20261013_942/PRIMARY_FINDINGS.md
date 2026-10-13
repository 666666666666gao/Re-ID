# 新分支怎样获得global之外的身份证据：有限原文／作者代码核查

2026-10-13。只核读来源和本项目已保存文本；未执行作者代码、模型前向、测试、SSH、训练或生产文件修改。核查范围不是完整新颖性检索。

## 判断

**本次没有找到能直接解释当前失败、或可承诺增益的外部机制。唯一值得留下的新增对照，是“显式跨光谱乘积描述子”与同参数加性描述子的区别。**它改变证据的内容运算；不是再换CE入口、repair/keep、遮蔽比例或路由权重。直接移植LMF本身不构成本项目创新。

ACoL/RSC式证据排除已在§924–929完成，gradient−random为−0.105086 mAP／−0.478470 R1。BIER已是V26职责训练的文献依据；其严格顺序boosting不等于已完整复现，但“困难关系重加权”不能再作为新点。BDB核读后只是随机结构遮蔽和独立分支监督，不能解决这条重命名风险。相关下载仅作排除性阅读，没有产生新候选队列。

## 可核实的来源机制：低秩跨模态乘积

[LMF原论文§3.2及实际Eq.(8)](https://aclanthology.org/P18-1209.pdf)用模态独立因子构造紧凑的联合表示，避免显式建立巨大的外积张量；其任务是情感、说话者特征和情绪分析，不是ReID。论文脚注明确链接下面的作者仓库。[TFN原文](https://aclanthology.org/D17-1115.pdf)是包含单模态／双模态／三模态项的外积先例；此处不依赖第三方TFN重写代码。

作者LMF固定commit为`ea4755124d19bddf7941695215e98f16fbb1fff9`。[model.py:122–126](https://github.com/Justin1904/Low-rank-Multimodal-Fusion/blob/ea4755124d19bddf7941695215e98f16fbb1fff9/model.py#L122-L126)分别创建三个模态的rank×输入×输出因子；[155–167行](https://github.com/Justin1904/Low-rank-Multimodal-Fusion/blob/ea4755124d19bddf7941695215e98f16fbb1fff9/model.py#L155-L167)追加常数1，分别投影，再逐元素相乘，最后沿rank加权求和。不是先把rank求和再相乘。任务反传没有在这些因子间detach；[train_mosi.py:129–134](https://github.com/Justin1904/Low-rank-Multimodal-Fusion/blob/ea4755124d19bddf7941695215e98f16fbb1fff9/train_mosi.py#L129-L134)对最终输出的L1任务反传并更新整个模型。它没有专门奖励“比强global更有用”。

由该实际计算可推得：

\[
j_d=\sum_r\lambda_r u_{R,r,d}u_{N,r,d}u_{T,r,d},\qquad
\frac{\partial j_d}{\partial u_{R,r,d}}=\lambda_r u_{N,r,d}u_{T,r,d}.
\]

因此RGB因子的任务梯度取决于另两模态的内容；乘积表示可以在单对象编码时生成并离线建库，无须query–gallery联合推理。该结构性质不证明新增互补身份证据，乘积还可能放大无关的共同成分或因某一模态趋零而失去活动。

## 本项目已有相似运算，必须正面区分

本次实际核读`modeling/trifusion/signal_preserving_v17.py:67–99`：V17对冻结V8的CNN／Transformer／Mamba残差做共享投影，把两个peer的Hadamard乘积、receiver和三支均值送入MLP，再纠正三个角色。**项目不是从未用过乘法交互，也不能声称既有模型全是单模态加和。**当前Transformer/Mamba及Signal注意力本身也能表达非线性跨内容关系。

可区分对象只能限定为：**RGB／NIR／TIR模态内容的独立因子乘积被明确保留为检索描述子**，而非角色peer对既有残差的纠正、或相似度仅用于给value分配注意力。这个差异是结构定义，不是“表达能力必然更强”的证明。

## 一个可以落实的有限候选

从现有证据形成前的三模态内容取每模态集合表示，保持同一编码器、任务、头、部署维度及global训练职责。先不对同编号槽位直接相乘：独立增强和未获证实的槽位语义，均不支持这种部件对应假设。

使用相同的三套**无bias**因子`u_m=W_m x_m`和同一rank读出，唯一比较：

- 加性控制：`j_add=sum_r λ_r (u_R+u_N+u_T)`。
- 联合条件：`j_joint=sum_r λ_r (u_R⊙u_N⊙u_T)`。

两边全部因子参与forward和训练，参数数目相同。第一版不复制LMF的追加1／融合bias，因为它们混入常数和单模态项，容易让“新增联合证据”又退化为另一份单模态投影；这是针对当前薄增量现象的**归因设计**，不是已证实的修复。无bias也不能保证实际数据上不存在退化或冗余，不应宣称获得严格的信息隔离或互信息增量。

三个模态在这里是任务输入，三个算子不因此自动成为三个贡献。该有限比较只回答“明确的跨光谱乘积内容是否比同容量加性内容更有净用途”。必要报告是同best的global／joint／fused及修复／新增错误、模态因子活动与乘积幅度、同模型与独立global两条增量；只提高joint单独mAP仍为失败。未接近完整Signal前不搜索rank、gain或seed。

**与历史尝试的区别：**不重复头约束、repair/keep、几何代理、patch查询anchor或位置排除；改变的是模态联合内容的形成运算。但是与V17和公开LMF足够接近，只有匹配加性控制、项目既有交互对照和真实检索收益成立，才有依据继续研究；当前不能把它写成原创主模块或解释已发生的全部失败。

## 文件与来源边界

`TENSOR_SOURCE_MANIFEST.json`记录作者LMF三个源文件、两个ACL原文PDF的固定URL、bytes和SHA256；下载文件在本目录`source/`及PDF文件中。`SOURCE_MANIFEST.json`另保存ACoL／BDB／BIER排除性核读源码，其既有作用已由项目来源文档说明。没有运行这些实现，没有更新环境或生产代码，没有新正式成绩。
