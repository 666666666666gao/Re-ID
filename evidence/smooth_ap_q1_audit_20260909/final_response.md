独立审计已完成：**总体 WARN，确定性核验 PASS，科学资格 Q1_FAIL，保持不晋级。** 审查使用独立上下文的 gpt-6-astra / max，记录 `review_independence: same-family`、`acceptance_status: provisional`。未发现假 GT、性能自归一化或虚构 Q1 终态的证据；WARN 来自真实模型梯度重建缺口、单 seed 内部协议及数值解释限制。

全部报告、脚本、失败尝试和冻结文本快照已保存到 [审计目录](C:/Users/gb/.codex_tmp/smooth_ap_q1_independent_audit_20260909)：

- [完整审计报告](C:/Users/gb/.codex_tmp/smooth_ap_q1_independent_audit_20260909/EXPERIMENT_AUDIT.md)
- [机器可读审计结论](C:/Users/gb/.codex_tmp/smooth_ap_q1_independent_audit_20260909/EXPERIMENT_AUDIT.json)
- [全部排名、更新日志与固定门独立重算](C:/Users/gb/.codex_tmp/smooth_ap_q1_independent_audit_20260909/independent_text_results.json)
- [远端全部数组、解析导数和 checkpoint 独立核验](C:/Users/gb/.codex_tmp/smooth_ap_q1_independent_audit_20260909/independent_remote_arrays_results.json)
- [全量正例导数精度比较](C:/Users/gb/.codex_tmp/smooth_ap_q1_independent_audit_20260909/independent_derivative_census_comparison.json)
- [最终输入哈希](C:/Users/gb/.codex_tmp/smooth_ap_q1_independent_audit_20260909/input_hashes_final.json)与[失败尝试记录](C:/Users/gb/.codex_tmp/smooth_ap_q1_independent_audit_20260909/failed_attempts_final.json)

| 检查 | 结论 | 独立核验结果 |
|---|---|---|
| A. GT / 完整路径隔离 | PASS | 1032 条训练记录、155 个身份、3096 条模态路径；三份 source checkpoint 与全部1950条来源训练索引记录均只属于本折 source。95 个无合法 query 身份、432 条干扰记录全部保留在 gallery。 |
| B. 原始指标 / 归一化 | PASS | 完整重算6000个 query×输出×端点 AP/CMC、2,069,520个排名位置、全部身份/折统计和固定 bootstrap，两组原门匹配。 |
| C. 文件 / 哈希 / 终态 | PASS | 57份原始文本、83,471,329字节匹配；六个最终 checkpoint 各472个完整状态张量重建一致。全部30组FP32检索特征→距离→排名逐位重现。 |
| D. 实际目标 / 导数 | WARN | 全1560更新、99,840个anchor AP和116,501,504个保存距离元素核验通过；完整模型参数梯度所需字段未保存，不能独立重建。 |
| E. 范围 / 成本 / 数值 | WARN | 完整三折两端、每端20epoch/260更新，但只有seed42、重复内部开发协议；预热step2起已有轨迹差异，极小导数符号受dtype影响。 |
| F. 评价类型 | PASS | Q1为真实标签的内部身份OOF；训练近似AP与数值一致性检查均未被当作官方性能。 |

独立重算的核心结果是：fused **52.444135613 → 52.787590269 mAP，+0.343454657 pp**；R1 **59.833333 → 61.000000**。配对fused分折增益为 **−0.200114287 / +0.245762205 / +1.077726874 pp**，身份bootstrap下界 **−0.101124438 pp**。因此配对门仅 **1/5**，候选相对Signal门 **0/5**。候选fused高于三个角色，但仍低于Signal的 **53.129380561 mAP**。原科学FAIL没有被更改。完整3000行query变化CSV和300行身份变化CSV的全部字段也已逐一核验。

距离导数的独立检查覆盖每个目标、每种dtype的全部 **29,125,376个位置**。NumPy解析Smooth-AP导数与部署自动微分最大差为 **Float64：3.5344990823e−17；Float32：6.8862138988e−8**；hard导数两种dtype完全一致，self距离导数为0。逐anchor AP最大误差 **2.5504035750e−7**，14项加权账本最大误差 **6.2833229730e−7**。

需要原样保留的边界有四点：

1. **Q1没有独立完整模型梯度重建。** 历史fields、RNG向量、逐步feature leaves、参数梯度和optimizer状态未保存；VJP应用、RNG/buffer保留和role梯度范数是运行见证。203/203是累计非零覆盖，不是每步203张量全部非零。六个checkpoint完整状态核验通过，也不等于重新从图像提取了特征。M0审计没有重做。
2. **配对不等于位级轨迹相同。** 三折共享目标的前65步从step2开始存在差异，最大total loss差分别为0.0018461943、0.0014842749、0.0021481514；没有独立训练重复的方差界，不能把单seed的小正增益全部解释为确定因果效应。
3. **极小导数符号需要精度限定。** 所有120组关系数量匹配；严格反序/非最大反序类别的符号计数亦匹配，包括candidate末段974个非最大反序正例的973有利/1相反。但末段所有正例中，Float32的7882个负号在Float64中仅6个，绝对梯度误差仍很小。这是数值解释限制，不能把每个极小负号当作实质错误监督。
4. **更新预算相同，计算成本不同。** epoch训练耗时control6201.269秒、candidate6557.227秒，增加5.7401%；历史VJP记录前向265344→292608。末段候选AP loss下降，而同定义expanded hard由0.141149升至0.158711；两个不同active total不可直接作优化优劣比较。

启动提交、Q1启动时提交和终态提交虽然不同，五个核心文件SHA一致；93项当前配置链绑定及143份既有远端源码/协议快照身份均通过。初始缺失的`EXPERIMENT_PLAN.md`、真实固定`TRAINING_PLAN.md`、审计中三个当前状态文档更新、Windows换行差异，以及审计器自己的失败和60秒SSH静默超时都已保留。最终295份文本输入快照已重新哈希验证。

原样最终回复已保存为 [final_response.md](C:/Users/gb/.codex_tmp/smooth_ap_q1_independent_audit_20260909/final_response.md)。数组、checkpoint和图片留在远端；本次没有模型前向、图像读取、optimizer更新或官方测试访问，不要求重训，不调整门槛，也不授权后继GPU实验。

<oai-mem-citation>
<citation_entries>
MEMORY.md:189-190|note=[workflow boundaries used to require fresh complete Q1 verification]
</citation_entries>
<rollout_ids>
01a053d1-497e-7511-8c2d-b37fd6d27278
</rollout_ids>
</oai-mem-citation>