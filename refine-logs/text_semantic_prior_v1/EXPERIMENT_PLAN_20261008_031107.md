# 冻结文本包必要性pilot：六端执行计划

**日期**：2026-10-08。**当前状态**：方法R2实际REVISE/6.40、有限pilot值得检验。最小组件源码复核SOURCE_COMPONENT_PASS，生产builder/entry/queue/report尚未接入；下文不表示实际prefix、NN或M0已合格。
**问题**：现有角色已学习、有独立判别能力，仍未稳定补充global的身份排序。
**Method Thesis**：固定CLIP文本预训练包是否比同结构固定随机包，更能让现有区域query从同一视觉patch中读到有用证据。不是论文原创性或独立新像素信息。

## Claim Map

| Claim | 为什么必要 | 最小证据 | Block |
|---|---|---|---|
| C1：此固定预训练包在现有区域上下文接口具有条件净效用 | 排除只增加592k参数和12层非线性容量 | pretrained−random三集每组mAP≥+0.5且R1不降；同时完整模型对旧semantic/global无亏损并有净收益 | B2 |
| C2：实验确实隔离文本包而非改视觉/global/训练任务 | 旧实验曾因目标牵连global、活动代理与实际读取不一致而失效 | 原common state/RNG/训练顺序一致、plain causal text path、角色梯度隔离与新参数真实活动、冻结state不变和严格重载 | B1/B3 |

反主张：收益只有容量、固定随机包偶然差、head/采样/维度改变或错误梯度接入；新增两个臂可排除其中部分，但单seed不能排除随机包总体/训练种子方差。word semantics、真实部件、SOTA与论文新颖性不在当前支持范围。

## Paper Storyline

当前产物是开发pilot。若为正，主文仍须后续完整流程稳定性、同资源强近邻和必要性证据；不提前写成paperREADY。当前附录可以保留执行边界和负结果。主动删除额外style prompt、text contrastive、原生CNN、matcher、独立新头和新排序loss；不用“更复杂版”占用本轮预算。唯一资源必要性对照就是同架构随机T。

## B1：最小代码及逐端资格（MUST）

只新增一个冻结文本上下文组件及semantic子类、一个生产训练入口、一个唯一六端队列、一个全量原始报告入口；复用raw职责分离训练器和原作者loader/eval/head。原428源码冻结字节，不因新的普通text路径修改作者模块。不复制整套角色forward、不加fallback。实际ResidualAttentionBlock构造参数名是d_model/n_head，12层显式forward_ori；构造随机初始化的draw不能消费已登记random填充RNG。详见完整R1修订。

无NN的source阶段检查完整调用路径、exact text-state映射、token模板、私有RNG及拟保存字段。之后才构造实际组件和执行固定prefix/full77输出/VJP对照：输出atol1e−5/rtol1e−5；pseudo-word梯度atol1e−4/rtol1e−4；两冻结包×person/vehicle模板，输入及上游向量预先固定。保留任何首次失败，不挑重试，不事后改阈值。

每端prepare→真实8步M0→fresh50→首次strict评价，M0不作为正式温启动。原common参数初值和全体原语义state与封存控制对应；新增ψ/W两个臂初值完全相同、文本state区别精确封存。原视觉first6GPU1/last6+headsGPU0，textGPU0，单个pair不并发占卡。original global/head只受L_g影响，新ψ/W/role/readout/gain只受L_r影响。冻结T参数不进optimizer、不收grad、不改变state；ψ/W准确增加592,000参数/5tensors。原trainable参数数20189,014,730、MSVR88,993,226、10088,831,946，加592,000；原trainable tensors281/285/285，加5。状态总参数数含冻结T另计。

第1次角色backward检查C=context_queries有限非零任务梯度与first actual update；首步ψ/W目标梯度预期0，不用weight decay变化冒称活动。第2—8次各新增tensor至少一次有限非零目标梯度及累计实际更新，原所有训练tensor8步活动规则保留。每步before optimizer的真实任务grad与effective updates记录，AMP overflow/作者BN次数/初始输出raw/fused/logits及buffer模式均实测。T内保持autograd，不使用no_grad；detached原视觉inputs之后构造text，再直接roles调用，禁止augmented_context再次detach。

检查原cpu/两GPU RNG与batch/augmentation流，记录原common输出init行为及全部实际梯度所有权。C0开通之后两个臂global仍应遵循相同原任务，但不能以单次参数/分数相同推定完整轨迹相同。严格保存/load整个持久T+template+ψ/W及旧state，不宽松省略frozen参数。M0合格后只退役本端已验收probe、fresh重建，不能原失败重试挑PASS。端级失败保留缺失，其余端只按事前合同独立进行，不被中途test分数改变。

## B2：两臂×三集全量效用（MUST）

固定队列顺序：RGBNT201-pretrained、RGBNT201-random、MSVR310-pretrained、MSVR310-random、RGBNT100-pretrained、RGBNT100-random。每合格端fresh50、seed42、公开CLIP/新camera/head，与sealed semantic相同初始化顺序和author配置。201B64/K8、MSVRB64/K4、100B128/K16；MSVRbase5e−6/bias×2/classifier×100等原分组保留，不统一新adapter LR，不按分数改变任何比例/词/seed/margin/epochs。

复用refine-logs/incremental_role_objective_v1/INPUT_SEAL.json（三raw semantic+三独立global，45artifact绑定；SHA2dd7dedf…ec2f1ea），不要重新训练六旧控制。prepare前真实验证所有依赖hash/protocol/公开CLIP；引用完整原initializer(cfg_yaml)而非仅写“作者配方”。common旧state应逐tensor对应，新T/ψ/W另独立statehash；不要拿添加后全statehash去要求等于旧全statehash。

主要比较pretrained−random，+0.5mAP/R1不下降；另报每new−sealed semantic、每new−independent global以及同权重f−own g。主表全部来自各自同一fused mAP-best，201R1/R5/R10，其余至少mAP/R1且保留完整CMC，50轮及末轮都公开。官方合法完整query/gallery和camera/scene过滤不变，不用留出子集补正式数字。shared global只在同一best、同一次visual forward获得，不另选global epoch。

pretrained与random有相同新增trainable数/固定text架构/推理成本接口。原semantic没有592k+T，因而pretrained−semantic不隔离预训练。随机token/LN等整体包对比不能独立归因于可读词义，也不能把单固定随机seed称随机包总体效应。项目推进阈值不是统计显著性或严格等价界限。

## B3：全量诊断与闭合（MUST）

六正式端或明确缺失端都进入唯一完整原始CPU报告。fused/own-g一次编码距离以及sealed controls输入严格绑定；完整全query首次修复/新增错误、AP差分、身份宏平均/收益分布，三集所有配对保留，不挑正例/指标。原始报告只完成一次；如果调用失败保留原exit并单独登记具体续接，不能声称总调用一次。

训练记录ψ/W实际更新、冻结T hash、text/context幅度、原Q/K/logit_std/读取熵和g/c/f，不把分支单独mAP或可视化当互补证明。记录实际全墙钟、峰值GPU显存、1536离线推理的额外T成本；history.seconds仅training loop。源码/工程资格只支持可执行性，方法review分数不作为科学PASS。

## Run Order and Milestones

| Milestone | 目标 | 新正式端 | 门 | 预计成本 |
|---|---|---:|---|---|
| 方法R2 | 修复已识别接口并判断有限pilot价值 | 0 | 实际final，同一reviewer；paper与pilot分开 | CPU文件/评审 |
| source实现/复核 | 最小入口、mapping、严格state/RNG和唯一队列 | 0 | 真实source复核，不宣称NN通过 | 无NN，时长待实际 |
| 组件资格+首端M0 | prefix输出/VJP、真实B64全图、staged活动 | 0 | 固定门，不重试挑通过 | 少量NN，未测 |
| 每端fresh50/strict | 6端300epochs | 6 | 各自M0通过、匹配初始化和完整官方评价 | ≥7.6h加T开销；16—30GPUh仅估计 |
| 全量报告/审计/贡献账 | 接收全部文本及明确缺失 | 0 | 原始CPU报告、完整query/消费者闭合 | CPU，待实测 |

不把全六M0统一通过作为首端长时间前置门；仍逐端M0→fresh50，有端失败不填分数。但统一组件数学/源码/空间合同应先闭合。长任务由持久远端进程运行，按实际M0/epoch预测里程碑或180—300秒观察，不原样重启旧任务/observer，不从临时低利用率推断结束。

## Compute and Data Budget

不下载数据/模型、不安装环境、不添加外部caption/SAM/DINO/人工作业。公开权重已有完整text12层；T只是冻结预训练包，推理也使用它。encoder本体148state tensors/38,131,200FP32元素=152,524,800B；另有固定template buffer1tensor/6,144元素/24,576B，组件总149state tensors/38,137,344元素/152,549,376B。均是静态估计，实构造/保存要核对。ψ/W2,368,000B新增trainable状态，Adam另计。单best规划上限512MiB，六best+一个活动M0probe+2GiB缓冲=5,905,580,032B；distance/log预算在2GiB内须按原全量shape确认，实际weight超过512MiB则资源未合格，不删T/宽松reload挽救。

2026-10-08T02:24:50+08实查free2,961,723,392B，reserv不足；未删除任何文件，不能视为launch storage门通过。只对明确OWN闭合、SHA/原回执/报告已确认、无当前/后继consumer的冗余weight/cache另列精确退役清单，不盲删。六sealed控制、三当前H2a best、strong Signal/V8/V27赢家及作者/公开初始依赖保留。文本已消费官方开发协议和单seed限制持续披露。

## 失败解释及后续边界

- 活动/显存/空间/重载不合格：该端没有正式性能证据，记录工程缺失，不推断text无效。
- pretrained≈random或不优于semantic/global：本固定包没有所需净效用，关闭本实现，不加新模块救分；不普遍否定所有文本方法。
- pretrained胜random但完整不如基线：先验包对比局部为正，完整目标未满足，不宣传方法成功。
- 只有一集为正：报告分化，不以201正值覆盖车辆负值，不挑seed。
- 三集净效用成立：另立同流程配对多种子、PromptSG/DEEP直接近邻及强Signal/V8/V27资源匹配研究；不自动SOTA/READY。

## Final Checklist（未完成项如实保留）

- [x] 一个主机制、一个支持claim与六端匹配对照已经具体定义
- [x] 无额外模块/目标，无同维度或容量混账
- [x] 原控制/数据/预算/消费边界明确
- [x] R2实际最终方法回执：有限pilot值得检验，paper仍REVISE
- [ ] source实现和独立复核
- [ ] 实际资源、prefix/VJP、初始化、8步M0与严格重载
- [ ] 六端fresh50+全量正式成绩与报告
- [ ] 完整流程稳定性/强近邻/SOTA/论文原创性证据

## 本轮实际来源和资源补充

2026-10-08T02:44:33.151671+08:00精确退役32个闭合旧distance缓存3,896,408,208B；原始64metric/train texts与SHA保留，无checkpoint/初始化删除。free_after=6606270464B，当次超过5,905,580,032B规划预留，但不代替未来启动前再验空间/实测保存预算。历史32raw distance不能再直接重放，不能称全部旧二进制仍存。

组件SOURCE_COMPONENT_PASS只覆盖新core文件及其静态接入边界；生产builder、唯一六端queue、prefix/VJP工具、真实M0、full50/eval/report尚未完成。0新NN/0正式端，无检索结论。
