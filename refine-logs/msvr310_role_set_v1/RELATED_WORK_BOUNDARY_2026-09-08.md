# Role-set 近邻边界与终态分析口径

2026-09-08：在已固定的26c9739 Q1运行期间补核论文，只补研究论证，不改变TRAINING_PLAN、配置、训练代码、晋级门槛或官方测试访问。

## 可核实的近邻

| 工作 | 本轮核实范围与机制 | 与当前实现的关系 |
|---|---|---|
| HDC，ICCV 2017 | 原文§3.1–3.3及Algorithm 1：递增复杂度的级联模型逐层筛困难样本，分层损失驱动相应模型，最后拼接表示。[原文](https://arxiv.org/html/1611.05720)、[会议信息](https://openaccess.thecvf.com/content_iccv_2017/html/Yuan_Hard-Aware_Deeply_Cascaded_ICCV_2017_paper.html) | 多模型难例挖掘已有先例。当前是平行角色提议的去重并集，在统一fused空间施加均值hinge；没有级联复杂度递增和逐层淘汰。这些是实现差异，不是效果或新颖性已成立的证明。 |
| DiVA，ECCV 2020 | 原文§3.1–3.2：区分类别判别、类别共享、类内变化和实例特征任务，独立embedding共享底层编码，并以去相关促进多样性。[官方PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530579.pdf) | 从不同关系获取互补监督已有明确研究。当前不新增四类任务，不翻转真实身份正负定义，不加去相关或自监督队列；只改变原身份目标中的负候选组织。 |
| DCML，CVPR 2021 | 本轮只可靠取得CVF原始论文页面摘要，PDF直接打开失败。摘要描述对子embedding的组合施加监督，并通过可学习compositor分配不同任务信号。[CVF论文页](https://openaccess.thecvf.com/content/CVPR2021/html/Zheng_Deep_Compositional_Metric_Learning_CVPR_2021_paper.html) | “保护独立判别能力，同时在组合表示上学习”并非未被研究的问题。当前保留原独立角色监督、固定拼接，没有compositor；不据摘要补写其全文损失或复现结论。 |
| BIER 扩展版 | 本轮复核作者arXiv摘要：embedding ensemble采用在线梯度boosting，各学习器接收前序学习器重加权的样本。[作者论文页](https://arxiv.org/abs/1801.04815) | 已有关系/样本重加权与embedding多样性机制。当前没有顺序boosting或前序学习器权重传递，不能宽称首次从训练关系形成集成互补。 |

以上工作为机制近邻，不是三个多光谱数据集的当前SOTA表；没有将其他数据集的Recall指标当作本项目mAP参照。没有复制这些作者实现，也未据此新增模块或实验。

## 当前实现可以准确表述到哪里

实际代码tools/msvr_role_set_relations.py的relation_objectives：fused和三个完整角色各提出最近负例位置，按候选位置去重。正例仍为既定最难正例；当前重复视图保留，历史同记录排除沿用旧队列。当前没有负身份配额、scene配额或全部历史anchor。角色提议索引停止梯度；新损失仍为原0.3未平方欧氏hinge的均值。

候选集合包含fused最难负例，所以同一固定状态下每个新增hinge不大于原hardest hinge，均值不大于hardest目标。此关系来自现有定义，非新定理。数值更低本身不能证明训练改善；额外关系覆盖与最极端项权重下降是同时发生的干预。

因此，潜在贡献最多先定位于“异构角色提议与统一检索空间的关系学习”，不能先称稳定互补、全新loss、首次完整梯度或首次多模型难例挖掘。是否具有方法贡献要结合完整配对结果和将来获准的直接对照判断。

## 完整Q1结束后用原产物回答的问题

- 首先按两组原五项条件报告全部600query/60身份/三折结果，固定seed42，不改变评价资格，不用局部fold修改方案。
- 来源优化同时报告两端的原batch Triplet、扩展hardest Triplet与role-set目标。比较相同标量的变化，不将candidate自身较小的均值损失与control的hardest值直接当成优化优势。
- 将65步预热前后分开。预热前记录的role提议只用于诊断，不能称为已用于目标；所有关系计数注明重复曝光，不能称独立训练样本。
- 从原proposals/negative_counts/extra_active_counts检查：角色提出了多少额外位置，其中多少hinge激活，是否主要是同一负身份的不同视图。位置去重不等于身份去重；单角色提议不同也不自动证明独有泛化能力。
- 从原两端训练与memory日志报告重编码、历史VJP、实际epoch耗时和峰值显存，匹配更新次数不等于匹配计算成本。
- 区分参数梯度运行见证、首历史组直接图核对、CPU保存数组复算与独立上下文审计。M0的6次直接检查不能外推为全部Q1梯度被独立重建。

这些是现有产物的分析清单，不新增训练评价门槛、超参数或来源采样合同。

## 终态接收准备

本地接收脚本receive_role_set_q1_terminal_20260908.py已完成本地和嵌入远端代码的AST语法检查，尚未对Q1终态运行。它要求原pipeline完整结束、原PID退出、五阶段退出0、CPU及summary哈希匹配后才接收全部json/jsonl/log，并逐文件核对字节及SHA。模型权重、原图和距离/检索二进制留远端。

该脚本只支持本次既定成功执行完毕的终态收据，包括科学Q1_PASS或Q1_FAIL；若实际pipeline工程阶段停止，应先读取对应失败日志和原PID状态，不能伪造终态、绕过检查或重启原任务。完整接收后再调用experiment-audit规定的fresh-context reviewer；当前未启动该终态审计。
