# 新增证据的最终用途：当前信息路径核查

研究主线：**不再追求让每个分支都更强，而是检验新增证据在最终身份比较中是否带来净收益。**外领域方法只回答其中一个具体缺口。当前六端训练与固定 best 五分数诊断均已闭合，训练保持停止。

## 当前实际路径

本轮只读源码，18份所用源码与原911封存SHA一致。定义位置见SOURCE_PATHS.json。动态入口以`tools/run_region_use_probe.py::configure`为起点，经global-task、role-detach、native-research、partitioned包装，最终使用semantic变体。这里没有原生图像CNN分支，也不是历史CrossDepthStateRoles或旧grid_sample读取。

`DetachedSemanticTriFusion`在角色边界停止stages、context、shared_global梯度；它继承`RawFeatureSemanticTriFusion`，其中角色对象是`GlobalTokenRoles`，读取由`FP32SlotCompetitionRoles.sample_context`实现。三个context query对应三个角色，不是三个模态。每个角色使用全部128个patch的FP32 Q/K/V注意力；`positions`参数在该有效覆盖方法中未使用，不能将旧位置采样解释套到本轮结果。

| 信息通路 | 当前实际行为 | 能说明什么 |
|---|---|---|
| global→context→query | 三模态global平均归一化后，映射成三个角色query，并与anchor组合 | global影响实际读取权重；detach只切断梯度，没有删除信息 |
| patch→key/value→区域 | key/value来自各角色多层语义patch；CNN先对语义网格卷积 | 没有把global直接广播成value。patch本身已包含CLIP交互后的上下文 |
| global→额外token | 当前构造固定`token_mode='static'`，额外token投影的输入是常量buffer | 该token不是样本变化的global投影；`token`/`direct`是旧候选模式，当前未用 |
| CNN→Transformer→Mamba | CNN区域经桥接进入Transformer，后者再进入Mamba；最后读出修正 | 完整排除某项证据须覆盖这些路径，不能只删除最终一个输出分量 |
| global＋修正→作者head/部署 | fused角色任务使用detach的global及当前detach head，部署仍为归一化1536维加法 | 职责分离已做；它不是新增证据互补性的保证 |

因此，**不能把重复信息的潜在风险直接归因于“global作为value/残差的直通旁路”**。query条件仍携带样本信息，且所有深层patch已共享上下文；当前也没有证明真实区域就是global的复制。

## 如果以后研究global引导的证据排除，依据必须来自真实计算路径

两卡前向在CLIP第4、8、12块输出处形成各角色snapshot，同时返回`output + mean(adapter_deltas)`继续计算global。供角色读取的`stages`是在三模态视觉前向结束后才把snapshots堆叠出来的；它不是global的计算祖先。不能对这份后堆叠张量直接求global梯度，然后把未使用梯度当成“global没有依赖patch”。当前角色输入还会detach。

作者视觉模型最后逐token做LayerNorm与投影，随后只取CLS。第12块**输出后**的patch不再被global聚合；对该出口patch的梯度也不能解释为原图区域不重要。若未来检查global关系对patch的敏感性，应在仍参与后续CLS注意力的真实完整hidden tensor处取依据，例如对应块的完整输入，先对完整tensor求梯度，再查看patch分量。不能拿事后切片的patch view替代真正传入下一层的tensor。这个接入位置仅由源码支持，尚未做实际VJP/有限性或设备检查。

屏蔽位置前的CNN、不同深度的信息混合、条件query和桥接都需要纳入解释。只把最终value置零并不等于令该位置退出注意力；只遮某个深层位置也不保证其他patch不含同一语义。空间排除至多改变新分支可读证据，不能提前宣称获得了global缺少的身份信息。

## 当前结论与下一项问题

§919的既有结果中，区域MaxSim R与固定联合J均低于同模型global；没有建立更强局部教师，也没有支持把压缩损失写成已确认原因。§920说明当前辅助目标可奖励区域身份判别性，但没有单独要求新增信息。此次源码核查又排除了一个过于简单的直通解释，并明确了后续定位应取的实际计算边界。

ACoL只作为**改变新分支可用证据**的形成环节参考；作者机制和限制见[已核查来源](../evidence_learning_gap_20261013_920/PRIMARY_SOURCE_SCREENING.md)。本轮不复制其类别mask，不把一个已知空间遮蔽规则重新命名为创新。

若以后恢复实验，先回答一个有界问题：在同一真实训练关系上，global引导的排除是否比等预算随机排除更能让新增分支补充clean global？需要同时检查真实读出改变、合法关系支持、同模型global/fused、所有正例AP的收益与损伤，并保留已有职责目标/repair-keep为对照。仅看saliency图、mask活动或分支独立mAP不够。

本轮未定义排除比例、启动新队列或实现候选。上述未来问题还需具体监督和成本设计；当前源码事实不是运行通过或性能结果。全三集超过Signal目标仍未达到，原best/诊断证据/依赖/失败继续保留。
