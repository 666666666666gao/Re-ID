# 当前role-input-detach的实际边界（源码检查）

两种新类仅在进入role_evidence时对stages、context、shared_global停止梯度。最终raw_fused仍使用未detach的shared_global；作者分类头和度量损失都接收这份融合后的raw表示。分段backbone仍把三组adapter增量的均值写回共享CLIP流，视觉和camera仍参加优化。

因此，该实验隔离的是角色读取输入到共享编码器的直接反向通道，不是严格冻结或保底global。在固定参数处，若把共享参数记为θ、角色参数记为ψ，其简化关系为h=gθ+αcψ(sg(Eθ),sg(contextθ),sg(gθ),I)。共享参数仍接收Jg,θ转置乘以融合损失对h的梯度；后一个量依赖修正内容和训练后的分类头。停止一条直接导数，不等于把共享路径的学习任务变成独立global-only。

这只说明源码定义，不能据此宣布梯度冲突成立或把某一端的下降唯一归因于融合耦合。收齐六端正式50轮与首次严格评价、原一次CPU报告后，再按已登记固定best诊断比较各模型global、correction、raw_fused、fused及独立global-only。当前不改变网络、损失、学习率、gain或queue。

本检查只经SFTP读取十个冻结文本文件并对照当前322源SHA；没有导入torch、构造模型、调用GPU、进行forward/反向或新增检索分数。
