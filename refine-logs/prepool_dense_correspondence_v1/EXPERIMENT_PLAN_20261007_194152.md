# H2a 固定三集实验计划

2026-10-07。方法评审2/2，6.35/10、REVISE，低置信度有限pilot值得推进；源码/真实M0尚未通过。完整三集强基线/SOTA/稳定性目标保持ACTIVE_UNMET，不被本单轮替代。

## Claim Map

| Claim | Minimum Convincing Evidence | Block |
|---|---|---|
| H2a：固定同模态几何监督包能改善现有semantic角色的净检索效用 | 三集完整50轮，相对匹配raw-semantic的mAP≥+0.5百分点且R1不降；原失败/缺失不删除 | B2 |
| 排除“只是多参数、换训练基座或改推理” | 0新增参数，原state/初始化、主输入/顺序/RNG/BN、原raw目标与1536部署对应；实际辅助梯度/支持检查 | B1 |

非主张：可靠跨光谱部件对应、H2根因、全流程稳定性、公式原创、SOTA或三模块已经有效。同模态对应精度/读取熵不替代身份指标。实际熵是全128-token主读取熵，mapped L1是未归一化共同可见mass差，不能称条件分布指标。

## B1：最小实现和生产入口

神经路径仅新增 modeling/trifusion/prepool_dense_correspondence.py 与 tools/run_prepool_dense_correspondence.py；另有本轮唯一队列/全量报告两个运行源，原训练器不变；复用原作者DataLoader/sampler/collate、raw任务、optimizer与两卡分段。Dataset包裹原transform并记录最后原eraser实际支持，额外剔除原Pad→Crop可能留下的10px边框；显式DenseTrainingBatch分支由真实车辆评价会调用_training_batch的现有证据要求。第二视图每模态batch共享、私有Random(42)、flip0.5→pad10/-1→均匀整数dy/dx±10裁回；不耗原随机流。teacher只是当前backbone/池化前路径的另一视图，不加EMA/模型/资源。

stage/global停止梯度；aux自然反传原depth混合、norm/adapters、模态embedding、CNN和key投影，也间接改变value的输入。原query/value/output投影、桥接、读出、gain继续由主身份任务训练。dense logit sqrt128*cosine，权重1.0；三角色×三模态等权平均。已知像素重叠是软目标，不把contextual token说成有限感受野真值。无支持teacher列退出辅助softmax分母；无支持行不计权重，实际每样本每模态必须有支持，否则保留资格失败，不提供替代loss。

必查：新source AST/实际导入；独立逐像素/矩形/张量计数；真实主loader像素/标签/camera/paths顺序与原配方一致；第二forward的原CPU/0/1 Torch RNG和buffer/作者BN中性；实际辅助VJP不回到shared_global，三个key投影均有限且累计非零，仍保留原全参数8步更新与完整state重载门。M0仅工程资格，不宣称效用。无旧Mamba kernel/反向重复性修复。

已有证据：CPU40几何参考、Torch40 batch2夹具通过。首次Torch归一化索引错误保留，仅修正mask索引的unsqueeze位置。4新运行源本地Python3.10 AST通过；2神经源初始源码FAIL后仅教师cache_enabled=False修补，第二次SOURCE_ONLY PASS，保留父FP16/主AMP。现有Torch CPU组件复现风险；三集真实作者CPU loader前8批主图像/5字段/RNG/log精确，CPU随机key VJP通过。首个关闭generator夹具失败保留，r2修订通过。尚无CUDA全模型构造、导入、M0或新正式成绩；queue/report/私有启动器整体SOURCE_ONLY第二轮PASS；端级验收移入独立子步骤，原条件不变，源码/磁盘共同门保留。

## B2：唯一新条件，三集完整效用检验

| 数据集/顺序 | 原raw-semantic mAP / R1 | 独立global mAP / R1 | 新端 |
|---|---:|---:|---|
| RGBNT201 / 1 | 74.3748615 / 78.8278 | 74.2966636 / 78.9474 | 原semantic＋H2a固定aux |
| MSVR310 / 2 | 50.5422249 / 67.8511 | 50.5421289 / 68.0203 | 同上 |
| RGBNT100 / 3 | 84.0903051 / 95.8601 | 84.5337842 / 96.6181 | 同上 |

三个旧raw控制及独立global都只复用封存成绩，不重新训练。主要配对永远新−raw-semantic；另报f−own-g及own-g−独立global，同一fused mAP-best严格评价、一次视觉forward收集own-g。不以own-g重新选epoch。仅现有公开CLIP、新camera/作者head、匹配初始化，seed42、B20164/K8、MSVR64/K4、100128/K16及原作者分组/日程/增强。每个端prepare→真实8步M0资格→fresh50→首次严格全量评价；fresh不继承M0。M0失效端缺失，不重试挑通过，不把未跑当0mAP。队列不能因某个中途官方成绩停止其余合法端或修改scale/weight/gain/seed/epoch/视图。

保留所有合法query/gallery与camera/scene过滤、所有50轮和step/batch日志；同一best的mAP/R1/R5/R10、末轮、own-g/fused全量距离、逐query修复/新增/AP、身份宏平均分布、实际参数/峰值显存/训练与推理成本。history.seconds只表示训练循环，不能求和称完整墙钟。预计正式6484更新仅是历史参考；由实际loader确认，不为凑数改采样。

## B3：仅在B2三集主要配对过线后另立合同

保持可见支持与计算的置乱目标及完整流程种子，检查监督内容和稳定性；当前不运行。aux=0第二forward没有学习信号，先做中性检查即可，不无谓增加三次完整50轮。完整Signal/V8/V27及最近近邻强参照仍是最终Goal需要，H2a成功不会消除这些义务。

## Run Order and Milestones

| Stage | Status | Gate | Cost |
|---|---|---|---|
| 方法定义/2轮评审 | COMPLETE_REVISE | pilot可行，非paperREADY | 0生产NN |
| 独立几何标签组件 | COMPLETE_WITH_INITIAL_FAILURE_RETAINED | 40类整数计数精确 | CPU，无模型 |
| 生产源码接入/新鲜源码审查/CPU真实loader检查 | SOURCE_ONLY_PASS_CPU_REAL_LOADER_PASS_CUDA_PENDING | 主路径中性、固定合同、旧424不变 | 尚未执行 |
| 每端实际M0→fresh50→全量严格评价 | NOT_LAUNCHED | 原逐端门，不设九端统一障碍 | 预计4–6h、8–12GPU·h，待实测 |
| 完整报告/审计/主张/无用权重退役 | NOT_LAUNCHED | 全量接收及消费者闭合后 | CPU/只退役本轮无消费依赖二进制 |

原五端空间门5,192,548,352B不改。新三端初步预算3best+1当前M0探针+2GiB缓冲=3,758,096,384B；M0探针验收闭合即退役、fresh重建。真实队列需确认3checkpoint/两类距离/日志预算与当时空闲；15:59空间4,396,077,056B不是当前资格。共享盘空间变化不可归给本项目。只2026物理0/1/单pair，不查25/2/3/功率温度，不安装或改环境。

## 固定结果解释

- 对应改善但主要效用没过线：关闭本实现；不再次调整参数救分，不否定所有跨谱对应方法。
- 检索过线而读取链不支持：只认经验收益，机制未定。
- 三集过线且读取链同向：可以登记后继必要性/完整种子，不称稳定SOTA。
- M0失败：该端未取得正式效用证据，保留失败与缺失。

官方集已参与研发，后续同样明示开发边界。0新增参数不等于0计算或0资源变化；文献公式已有先例，不以项目命名或位置变化取代原创性证据。
