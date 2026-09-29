# 全局条件查询与联合局部身份职责：固定后继实验

日期：2026-09-29。状态：定义与实现准备，尚无生产M0、正式训练或结果。当前M3原12端继续完成，不修改其18份科学源码、8文件manifest、配置、已选权重或失败记录。

## 问题与两个待验证主张

现有完整24消融、3个独立global-only、15项采样/读出和RGBNT201 M3四条件显示：共享适配能提高检索，但角色额外收益很薄；匹配位置/预测器没有补回RGBNT201的差距。这不等于已证明global捷径或静态查询是唯一原因。

- C1：样本的共享全局内容进入查询，而不进入局部value，能帮助角色选择有身份价值的局部证据。
- C2：联合局部表示有自己的身份出口，能减少只依靠global完成最终分类的学习路径，并改善未见身份检索。
- 排除解释：更多参数、更多分类监督本身、扩大候选邻域、换M3系数、不同初始化/训练预算、跨轮拼列。

这些是待验证主张。全局引导局部已有先例，例如DSGM作者的Mask_patch.py把global与区域patch共同送入注意力；DeMo已有不同信息来源的专家融合。这里不主张“首次全局局部协作”，也不使用它们的文本/SAM2资源，新增边界在查询条件与联合局部职责的受控组织。

参考：https://github.com/zw-absin/DSGM/blob/main/modeling/fusion_part/Mask_patch.py 。仅思想参照，不复制其模块或报告其性能为本实验预期。

## 固定结构

沿用纯CLIP ReID权重、M1、16锚点、九候选内容查询、CNN→Transformer→Mamba与区域读出。冻结CLIP/原纯baseline；M1和角色可训练。最终仍为1536维、一次L2归一化的global加角色修正，不增加外部资源或测试时自适应。

令g为三模态global的1536维拼接，将它reshape为3×512后取模态均值并L2归一化得到c(x)。每个角色e有512→128无bias查询投影A_e。候选内容评分使用LN(q_k+A_e c(x))与LN(候选特征)的点积softmax；global仅进入query，不添加到候选value、角色token或局部读出。桥接中的slot向量仍为原q_k，避免直接把条件global复制入局部表示。

参数匹配静态控制：同样三个A_e，但输入固定单位向量c0=ones(512)/sqrt(512)，形成可训练角色静态查询。它与样本条件版具有相同存储/可训练参数量；静态输入的有效自由度较低，不能把该对照称为相同函数容量。A_e均零初始化，所有配置初始主检索forward一致。新增构造用fork_rng保持随后增强/loader随机流，共同初始完整state核对。

联合局部表示l=Normalize(区域读出(CNN,Transformer,Mamba))，不包含加回的global。辅助BN+无bias身份分类头宽度1536，各配置都构建同样初始state；无aux配置冻结且不执行该头。aux=local时监督l；额外监督控制aux=global时监督Normalize(g)，头参数量、CE标签、权重相同。只给联合局部出口，不给三个角色分别添加完整ID/Triplet目标。

M3在本组所有配置关闭，以隔离查询与身份职责，不从当前M3四条件中选择赢家或沿用其最佳checkpoint。这是新统一结构对照，不把它的收益归给删除M3。CLIP从已训练ReID baseline继续训练新增模块的两阶段成本必须注明。

## 五个条件与三数据集

| 条件 | 查询输入 | 辅助身份目标 | 作用 |
|---|---|---|---|
| static_none | 固定单位向量 | 无 | 同结构控制 |
| context_none | 当前样本global | 无 | 查询条件作用 |
| static_local | 固定单位向量 | 联合局部 | 局部职责作用 |
| context_local | 当前样本global | 联合局部 | 完整候选 |
| context_global | 当前样本global | global | 额外分类监督控制 |

全部5×RGBNT201/RGBNT100/MSVR310=15正式端必须保留，不因201或车辆中间表现取消其它端。首先固定seed42作研发对照；以后稳定性补充另行登记，不以换种子代替方法改进。

每端生产M0八批，确认真实Mamba、所有应训练参数累计非零/有限梯度、原baseline不变、紧凑学生权重独立重载一致；M0不提供检索成功结论。随后正式50轮、既有每轮官方fused mAP保存best规则，同一份best独立重载，全query/gallery与作者camera/时间段过滤核对。其它指标随同一权重，不能跨轮或seed拼列。

损失固定CE_fused(label_smoothing=.1)+Triplet_fused(margin=.3)+CE_aux(label_smoothing=.1)，aux关闭时最后一项为0，aux权重1.0固定，不扫描。global/local额外头相同容量，CE_fused与CE_aux分别记账；无EMA/预测器。原loader、AdamW、新增模块学习率、weight_decay、warmup和50轮日程完全复用。

## 最小证据与判定

核心配对：static_none→context_none；static_local→context_local；static_none→static_local；context_none→context_local；context_global→context_local。先报告各数据集条件差，不将它们相加成十点贡献；查询平均作用和职责平均作用只由前四完整端计算。主要看正式mAP/CMC及修复旧错/新增首位错误、身份等权AP变化。

完整候选相对同结构控制与同监督global控制的mAP/R1若没有稳定一致增量，则不能宣称联合局部职责已经解决瓶颈；来源aux分类准确率或loss下降不替代未见身份检索。单seed和官方逐轮选点/历史消费的限制保留。方法成功仍不能替代三集同协议强baseline/SOTA或纯基线+10的原目标。

最后同一best还需保存完整global-only及joint-local距离和诊断指标，确认局部身份证据而不把它当独立训练单角色消融。不得依据正式失败query/身份定制位置、损失或门槛。

## 顺序、成本与执行边界

先实现与fresh Codex复核、CPU结构/初始化/梯度测试；再在符合既有非抢占规则的空卡做真实生产M0。新文件与独立入口，不修改当前训练源。启动依赖在首个新GPU任务前登记为：原M3全部12端已经启动、没有pending或failed，201/MSVR四条件已完整CPU验收。新任务只使用空卡；首个GPU子进程前保留该卡旧worker，等待它完整train/evaluate结束，防止阶段切换显存暂时释放造成竞争。原100最后端不抢占、不改源码或日程，新15终态必须同时保留并验收原12完整结果；当前原12训练/重载必须继续，既有结果后续仍完整分析，不能以新方案取代剩余任务。

预计15端约12–20训练及逐轮评价job小时，沿用四卡并行；这不是实测GPU小时或受控速度，M0/重载和并行负载另记。失败先保留输入/日志，停止新增pending，不自动重试、fallback、跳过数值断言或盲目调参。

## 状态

定义、五份独立源码、模型/入口及执行链fresh来源复核已完成。五条件CPU合成检查已通过；真实CLIP/Mamba M0和15正式端尚未运行，复核PASS仅source-only/same-family/provisional。13:15原M3只剩一项100训练，三张卡释放；经启动前登记和fresh复核，新任务在空卡接续，并保留旧worker GPU直至其完整评价结束。新方法没有正式结果，不能转移旧Signal增强83分或原111成绩到这个名字下。
