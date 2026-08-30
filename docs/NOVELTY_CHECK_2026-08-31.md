# TriFusion-ReID 新颖性查核报告（2026-08-31）

状态：完成 2024–2026 主源检索、最近六个月补充检索与四轮独立
GPT-5.5 xhigh 对抗式复核。新颖性结论为 **PROCEED WITH CAUTION**；
实现就绪性经历两次 **REVISE** 后，最终复审为 **PASS**。该 PASS 只表示
规范可在公开接缝获用户同意后进入测试先行实现，不证明新颖性、有效性或
SOTA。本文不把尚未通过消融的候选机制写成已成立的论文贡献。

## 1. 候选方法

TriFusion-ReID 让 CNN、Transformer 和 Mamba 三个完整架构专家分别处理
RGB/NIR/TIR 全部可用模态，并保留三个可独立检索的分支头。候选方法原由
HFER 深层双向中继、URGC 可靠性路由和 RDPT 方向式同伴教学组成；同一
expert×modality 可靠性后验计划控制中继、融合、缺失/退化抑制和教学方向。

## 2. 需要查新的核心主张

1. **HFER**：三个完整异构架构专家之间进行同步、低秩、角色保持的深层
   双向交换，同时保留私有残差并改善接收分支本身。
2. **可靠性学习**：在共同尺度上估计逐样本 expert×modality 的实际任务
   效用，而不是把 attention weight 改名为 reliability。
3. **统一控制**：同一个校准后验同时控制深层中继、最终融合与缺失/退化
   抑制，并在单次推理前向中工作。
4. **RDPT**：根据可靠性差动态选择、拒绝异构专家教学，并传递架构角色
   特定的关系知识。
5. **整体链路**：可靠性估计 → 深层通信/融合 → 分支互促，面向 RGBNT
   ReID 的三个完整异构专家。

## 3. 检索范围

每项主张使用至少三种查询表达，覆盖 arXiv、CVF、AAAI、NeurIPS、PMLR、
期刊官网与官方项目页，并重点补查 2026-03-01 至 2026-08-31。查询族包括：

- heterogeneous CNN Transformer Mamba deep mutual relay/fusion/ReID；
- multimodal expert routing, calibrated uncertainty, leave-one-out utility；
- expert×modality counterfactual/causal routing and randomized intervention；
- adaptive/selective/rejectable heterogeneous peer distillation；
- RGB/NIR/TIR missing/degraded modality robustness。

有限检索不能证明不存在其他先例；投稿前必须按提交日再次滚动查新。

## 4. 核心结论

| 主张 | 评分 | 结论 | 最接近工作 | 最强否定理由 |
|---|---:|---|---|---|
| HFER | 5.0/10 | MEDIUM | [FusionReID](https://arxiv.org/abs/2412.17239) | 已有 ReID CNN–Transformer 多层异构传输；增加完整 Mamba 专家若只提高 ensemble，容易被判定为架构组合 |
| 原 URGC | 4.0/10 | LOW | [TIER-MoE](https://arxiv.org/abs/2607.27289)、[MRUF](https://arxiv.org/abs/2607.10599) | 条件风险、模态–专家路由、留一效用监督、校准和缺失鲁棒性均有直接先例 |
| RDPT | 3.0/10 | LOW | [HeteroAKD](https://ojs.aaai.org/index.php/AAAI/article/view/32399)、[MST-Distill](https://arxiv.org/abs/2507.07015) | 选择式、可靠性感知、异构和动态蒸馏均已拥挤；角色 payload 本身不足以成为主创新 |
| 统一系统 | 5.5/10 | MEDIUM | [UGG-ReID](https://arxiv.org/abs/2507.04638) | 不确定性引导的多模态 ReID 图/MoE 已存在；可防守的是三个完整专家和单一路由器的精确闭环，而非任一原料 |

独立复核明确否定“三个同等强度主创新”的原始叙事。强行保留 RDPT 为第三
主创新会增大拒稿风险。

## 5. 最近工作碰撞表

| 工作 | 年份/场所 | 已覆盖内容 | 对 TriFusion 的约束 |
|---|---|---|---|
| [FusionReID](https://arxiv.org/abs/2412.17239) | 2024，IEEE T-ITS | CNN+Transformer 完整双分支、多层异构传输和反复互融 | 不能声称深层异构交换或 CNN–Transformer 协同本身新颖 |
| [ReMamba](https://www.nature.com/articles/s41598-024-80766-8) | 2024，Scientific Reports | CNN 局部特征渐进注入 Mamba，全局/局部交互和四向扫描 | CNN→Mamba 渐进交互已有先例 |
| [MambaReID](https://www.mdpi.com/1424-8220/24/14/4639) | 2024，Sensors | Mamba 用于 RGBNT 多模态 ReID | 不能声称首个 RGBNT Mamba |
| [MambaVision](https://openaccess.thecvf.com/content/CVPR2025/html/Hatamizadeh_MambaVision_A_Hybrid_Mamba-Transformer_Vision_Backbone_CVPR_2025_paper.html) | 2025，CVPR | CNN/Mamba/Transformer 混合视觉骨干 | 三种架构共存不是新颖性 |
| [UGG-ReID](https://arxiv.org/abs/2507.04638) | 2025，NeurIPS | 局部/样本不确定性、MoE 路由和可靠融合 | 可靠性感知 ReID 路由已被直接覆盖 |
| [RoDI](https://openaccess.thecvf.com/content/CVPR2026F/html/Li_Rolling_and_Denoising_Rethinking_Dynamic_Modal_Fusion_for_Multi-Modal_Object_CVPRF_2026_paper.html) | 2026，CVPR Findings | evidence/belief/uncertainty 驱动滚动和去噪 | evidential 动态融合不是独立贡献 |
| [MRUF](https://arxiv.org/abs/2607.10599) | 2026，IEEE SMC | 留一误差增量监督模态路由、不确定性校准融合 | 原 head-level LOO 效用监督高度碰撞 |
| [TIER-MoE](https://arxiv.org/abs/2607.27289) | 2026，arXiv | 交叉拟合的条件模态风险、模态–专家稀疏路由、校准 | 交叉拟合和风险路由也不能单独声称首次 |
| [TMUR](https://arxiv.org/abs/2604.09288) | 2026，arXiv | 证明独立视图 evidence 尺度不可比，提出统一全局路由 | 九个独立 evidence head 不成立；必须共同目标、共同尺度和分组校准 |
| [TIGER](https://arxiv.org/abs/2606.15765) | 2026，arXiv | 异构视觉基础模型按排除专家后的预测变化进行反事实路由对齐 | “专家排除 + causal routing”已有先例 |
| [From Observation to Intervention](https://arxiv.org/abs/2606.10703) | 2026，arXiv | 观测路由统计不能代表专家因果重要性 | 必须用实际干预而非 gate correlation 支撑因果措辞 |
| [When Does Context Routing Help?](https://arxiv.org/abs/2608.25128) | 2026，arXiv | 容量底线、上下文腐化和因果干预辨别真实路由收益 | 必须加入容量底线与干预归因控制 |
| [Adaptive Teacher Modality Selection](https://www.sciencedirect.com/science/article/pii/S1568494626004497) | 2026，Applied Soft Computing | 模态贡献排序、自适应选择强教师并向弱模态蒸馏 | 动态教师和强→弱教学不新 |
| [CoReTrack](https://www.sciencedirect.com/science/article/pii/S0957417426009942) | 2026，Expert Systems with Applications | RGBT 可靠性先验、教师调控非对称蒸馏和门控 | 可靠性控制的非对称跨模态蒸馏已有近邻 |
| [FF-DML](https://proceedings.mlr.press/v260/lin25a.html) | 2025，PMLR | 可见–红外 ReID 融合辅助分支和深度互学习 | ReID 内同步互学习已有先例 |
| [Miss-ReID](https://papers.nips.cc/paper_files/paper/2025/hash/ef3a55fa15aa5fe39b7a2617b3a5d06e-Abstract-Conference.html) | 2025，NeurIPS | RGBNT 缺失模态训练/推理与视觉–文本补偿 | 缺失模态鲁棒性必须比较，不能写“首次解决” |
| CTMambaFuse | 2026，Infrared Physics & Technology | CNN+Transformer+Mamba 用于红外–可见图像融合 | 禁止“首次组合三种架构” |

## 6. v1.1 研究决策

### C1：HFER 作为架构中心，保留但窄化

可防守主张不是“三架构组合”，而是：

- 三个专家都是可独立训练、可独立检索的完整网络；
- 每个专家都处理 RGB/NIR/TIR，而不是一架构绑定一模态；
- 中继发生在后续原生层之前，并必须提高至少两个接收分支；
- 私有残差、独立头和容量匹配对照证明增益不是 ensemble 容量。

若只改善融合输出，HFER 降级为普通 feature fusion。

### C2：CIRC 取代原始 head-only 可靠性主张

候选 **Cross-fitted Interventional Reliability Calibration (CIRC)** 使用：

1. 身份不相交的 K-fold 外折 HFER-uniform 教师；
2. 冻结教师上对九项全量执行 total、direct、relay 干预；single-edge 仅作
   audit-only，每个 query-condition、每个 relay stage 用冻结哈希抽一条
   有效边，额外两次而非 36 次完整前向；
3. 用 held-out 冻结特征和强制跨相机正样本构建非参数身份 reference
   bank；同相机或无有效正样本的行不进入主损失；
4. 对每个腐化 family、severity、seed 独立完整重跑 query；\(r\) 只学习
   helpful-vs-not-helpful，neutral、harmful 和非加性交互残差仅作有符号
   审计，不声称三态分类；
5. 一个联合输出九项、共同目标尺度的全局 Beta 路由器；
6. 按条件报告 BCE/Brier/ECE、相机/身份频率分组校准、过度离散和经验
   coverage；不把异质条件或 seed 当作 iid trial 合并；
7. 预注册 query-only 与 query-gallery 对称审计，以及外折代理教师到
   最终部署模型的 target-transfer 审计；
8. 仅把 cheap head-level LOO 当作经完整干预验证后的稠密辅助量。

这仍不能声称“首次 causal routing”，因为 TIGER 已用专家排除反事实对齐。
可检验差异仅限：RGBNT retrieval、身份外折、expert×modality 粒度、relay-edge
干预，以及一个后验跨深层控制位置复用。

### C3：URGC 改为统一控制/系统贡献，而非新 gate

CIRC 后验同时控制 HFER edge bandwidth、最终 expert×modality 融合和
缺失/退化抑制。第三点的价值来自“同一个经干预校准的信号形成闭环”，而
不是 Beta、softmax 或 dynamic fusion 本身。必须比较：

- 同一路由器处处复用；
- 三个独立同容量路由器；
- fusion-only、relay-only；
- shuffled target、permuted expert identity、温度重缩放；
- 等参数、等 FLOPs 的无上下文 capacity floor；
- 完整/缺失/退化下的 fused 与三个 branch 检索。

C2 与 C3 在论文中应作为一个统一可靠性贡献的“学习”和“使用”两部分，
不能伪装成两个互不相关的首次机制。

### RDPT：保留为辅助机制，默认不列主贡献

RDPT 可以继续实现和消融，以满足“互相促进”的训练研究，但 v1.1 不把它
列为第三主创新。只有同时显著超过 symmetric KL、固定教师、HeteroAKD/
MST-Distill 风格选择、wrong-payload swap，并改善弱分支且不导致 CKA
坍缩时，才允许在论文中晋升。否则从主模型删除或仅作为补充训练正则。

## 7. 共同尺度与因果可信度门禁

一个 global router 并不自动解决 TMUR 的可比性问题。晋升前必须满足：

- 九项使用完全相同的目标定义：有跨相机正样本的外折 query 上，total
  干预导致正确 ID margin 下降的条件化概率；
- target generator 与被训练 router/fusion head 参数隔离；
- router 不接触生成其 target 的同身份训练数据；
- folds、阈值、腐化套件、reference encoder、cache 代码与指标必须在
  target 生成或读取 dev label 之前冻结并哈希；
- 每个 family×severity×seed 独立计算 BCE/Brier/ECE，并按 expert、
  modality、expert×modality、camera 和 identity-frequency 分组；
- 全量分拆 total/direct/relay 效应，edge 按冻结哈希抽样审计；记录
  helpful/neutral/harmful 与 total−direct−relay 交互残差，但不把低 \(r\)
  解释为 harmful；
- posterior concentration 只有通过逐条件经验 coverage 和过度离散审计
  才能称为校准；有效样本量按 identity/query cluster 计算；
- 完整重跑干预与 cheap head approximation 在冻结 dev 子集上相关；
- query-gallery 对称性和代理 target→部署模型迁移通过预注册门；否则主张
  分别缩窄为 query-side 或 proxy-specific；
- 缺失条目严格为零，低质量但存在的条目单独评估；
- 随机 target、身份/相机泄漏、expert permutation 和 capacity-floor 均不能复现增益。

达不到这些条件时，只能写“动态门控”，不能写“可靠性估计”或“因果校准”。

## 8. 最小论文证据

### HFER

无交换、末端融合、参数/FLOPs 匹配融合、两专家子集、无私有残差、无角色
变换、stage 数量、每分支协同前后 mAP/R1、CKA/多样性和梯度覆盖。

### CIRC/URGC

softmax/scalar/expert-only/modality-only gate、独立 evidential heads、
TMUR-style global router、无 Beta、无干预 target、cheap-vs-full 干预相关、
逐条件 BCE/Brier/ECE、全量 total/direct/relay 与 sampled-edge 审计、
helpful/neutral/harmful signed strata、过度离散/coverage、query-gallery
对称性、proxy-to-deployed transfer、
腐化单调性、缺失模态均值/最差检索、同一 vs 分离路由器、随机 target 与
容量底线。

### RDPT 辅助项

无 KD、symmetric KL、固定教师、confidence-only、reliability-gap、
wrong-payload、无 private hinge、拒绝率分布、分支提升、CKA 和受控退化
方向变化。

### 统计与最终测试

至少三个固定种子；query-level paired bootstrap/permutation；配置只在
141-fit/30-dev 上冻结；最终全 171 identity 重训后每 seed 只做一次官方
test；静态 CLIP 赛道同时超过 84.1 mAP 和 87.2 Rank-1 才允许 SOTA 措辞。

## 9. 建议论文定位

建议统一表述：

> Causally calibrated full-expert collaboration for tri-spectral ReID:
> three independently retrievable heterogeneous experts exchange
> role-preserving information under one intervention-calibrated
> expert×modality reliability posterior, with branch-level and fused-level
> evidence under missing and degraded modalities.

禁止使用：

- first Mamba for RGBNT ReID；
- first CNN–Transformer–Mamba fusion；
- novel uncertainty/reliability gate；
- novel selective KD 或 novel dynamic teacher selection；
- first multimodal MoE；
- solves missing modality；
- SOTA dynamic fusion；
- unified reliability（若没有外折干预和校准证据）。

## 10. 总体判定

- 分数：**5.5/10**
- 建议：**PROCEED WITH CAUTION**
- 实现就绪性复审：两次 **REVISE** 后最终 **PASS**；已补入跨相机
  support、全量 total/direct/relay 与 sampled-edge 审计、逐条件 proper
  scoring、对称性、target-transfer、post-freeze all-171 和共享 Beta
  参数化合同。PASS 不替代任何实验门。
- 最可防守差异：三个完整、独立可检索的异构专家；外折实际干预校准的
  expert×modality 信号；同一信号贯穿深层中继与最终融合。
- 首要风险：审稿人把模型视为 FusionReID + Mamba + MRUF/TIER/TMUR +
  selective KD 的组合。
- 已完成的设计动作：降级 RDPT；把 self-referential head-only LOO 改为
  冻结外折的全量 T/D/R target 和 sampled-edge 审计；冻结并哈希 target
  协议；把 HFER 的分支改善、容量匹配、CIRC 对称性和 proxy transfer
  设为不可绕过的主门禁。下一步仍需用户明确接受 TDD 接缝。
