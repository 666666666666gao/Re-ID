# 实施前锁定与源码自查

直接按行softmax，固定null=1；不运行已失败100迭代OT数学检查。原失败源独立保存。两臂均调用同一match矩阵、同一row-null和同一出口；只有r在接收槽位间保留或平均的差别，rbar可微。matching_projection默认Kaiming均匀；出口唯一置零；CPUfork RNG恢复。score双向转置后分别按行归一化，不转置最终W。Q列集中度是conditional Q分布，不声称OT列边际。所有新增参数真实参与forward/训练，旧权重在fork构造后逐项拷贝。

RAW职责与作者head/BN/optimizer由封存入口直接复用；每个新worker全新进程，不把monkeypatch串入旧控制或原已完成NN。M0诊断覆盖两矩阵累计梯度/更新。源绑定保持旧公共CLIP/camera/shared/author/batch/cfg/head/梯度政策/placement；新初始forward只要求两臂一致和旧global/公共state一致，不要求不同Mamba序列的fused与旧RAW一致。正式验收复用原receipt/全gallery检索且只接受首次strict。

方法两轮评审针对旧Sinkhorn完整稿；新直接softmax是根植同一slot_mass/mean-mass代数控制的简化，未独立获论文评分。根审阅没有将“可实现”冒充“机制有效”。组件数学/活动通过不能替代真实8M0或六端检索。无额外fallback/try/except/兼容层/无关重构。
