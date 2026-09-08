# Smooth-AP Q1后来源诊断设计草案

状态：设计草案，未登记、未执行；不是新的训练方案。独立Q1审计已闭环；本草案未登记执行。只使用当前已结束Q1的六个固定epoch20端，不能修改原checkpoint、训练或Q1门。目的为补足用户指出的两个测量缺口；不以诊断本身代替三数据集性能目标。

## A. 同角色上的fused排名与其余13项参数梯度

复用probe_msvr_history_candidate_gradients.py的固定状态、完整注册来源batch序列、ViewFields、分组RNG恢复及candidate_vjp思想，目标换为各端真实active fused目标。六端各自固定参数，260步注册序列，仍保留65步预热和原512/8记忆规则；全部采样及增强像素与原日志核对。没有optimizer、scheduler或更新；每步恢复七个neck等buffer并检查模型状态不变。

只比较同一encoder角色参数块。定义加权fused梯度gF=wF*(gF_current+gF_history)，其余13项gO=grad(weighted_total-wF*fused)。gO没有历史候选分支；当前样本不只是query，也在当前候选中参与导数。记录gF/gO/两者之和的norm、cosine、重复反传数值噪声、分角色和全来源阶段分布。必须核对gF+gO等于同一固定总目标的完整梯度，不能拿现有total-current与history的日志代替这一分解。

沿原模块AMP范围计算，诊断采用已有固定状态probe的SCALE=256并明确它不恢复训练各步动态GradScaler；历史重编码/RNG分组语义不变。每步完整历史group先累加表示上游再VJP；只跳过实际零上游group，不能采样一小部分group声称全量。不能使用不同模型的CNN与Mamba梯度做余弦。保存足以重算norm/dot及loss账本的文本；大张量及原图只留远端。直接全图检查的范围、未保存全程参数梯度的限制必须明示。

解释范围：固定终点模型在原注册来源视图/队列上的诊断，不是恢复已经结束训练的逐步参数轨迹。不能依据它声称训练过程中每一步的实际目标占比已被重建。

## B. 同一固定表示下候选集合与完整来源图库

复用diagnose_msvr_source_relations.py的source_loader、固定seed42视图和完整source身份/路径核对；使用当前Smooth-AP的build/reload接口，不沿用旧source_style状态名与合同断言。六端各两个明确视图（clean、一次seed42几何增强），模型eval，无统计扰动、无模型更新；每条件完整覆盖各fold全部672/683/709来源记录。

每个条件先一次取得完整来源表示表；之后所有候选子集与完整图库都从同一表取行。保存五个部署输出，按同一真实身份/scene规则分别计算all-identity与合法cross-scene排序。候选子集由原登记batch+历史记录索引构造，按record_index去重并排除query自身record_index；完整图库也排除query自身record_index。当前anchor位置可重复计入曝光，但不把其自身记录的另一副本作为正例。明确这是固定单视图、唯一记录池的覆盖诊断，不复现原训练中独立随机重复视图及其权重。

候选AP与全图库AP只对共同合法query报告配对差；另外保留全来源合法query结果和无正例query计数。无合法正例的记录不被从其他query的负例图库删除。报告每个query正/负例数量、反序正例数量和AP、rank1，不用不同候选规模下的AP差单独证明分数校准错误。

解释范围：相同固定模型/同一表示表上的候选覆盖效应。它不等于实际train模式/历史dropout重放的损失，也不是官方测试或新的身份外检索结果。

## 待最终固定后才执行

- A沿已有固定状态probe：进入状态时快照全部buffer，每步结束恢复；模型总状态SHA最终相同。完整梯度分解及历史重编码仍须真实预检，不能以源码阅读声称已验证。
- B已明确唯一记录池与self记录排除；该条件差异必须写入结果，不把差距全归因于原训练缓存。
- 先完整审计当前Q1，按真实诊断成本确定持久运行预算；尚未承诺耗时或磁盘。
- 不扫描loss权重、temperature、跨scene过滤或角色分配；上述诊断不能被改写为已实施的算法贡献。
