# MSVR310 原三角色比较：第0折训练完成后在基线特征一致性门停止

记录时间：2026-09-06T06:41:47.899579+08:00。状态 **COMPARISON_STOPPED_AT_BASELINE_FEATURE_PARITY**。
原wrapper70422已退出1，执行1c444cd；没有运行中的训练进程。
第0折完成固定20epoch、260更新/16640训练记录曝光，所有训练工程条件通过。
完成最终checkpoint写入及严格重载，随后提取原完整360条heldout gallery。
进入evaluate后，在torch.equal(features[baseline_only], original[features])断言停止。
检索排序/AP/Rank未计算，fold1/fold2未建立或启动；不能判为三角色检索有效或无效。

原summary字段RUNNING和训练前checkpoint保存之前写入的receipt均保持原字节；
退出码/堆栈/实际最终checkpoint以单独失败收据说明，不补造未落盘的终态字段。
原epoch20 checkpoint SHA b8a85e167861c51bb7d9a5854d700d11468ee9ca2a6e7ba21b75ce557130003c，
partial summary SHA7b57f573889bbff1591fb2bb01350c59a260949292aea98c4b8f8cf6c2b7bef0。
完整10文件及原fold training/receipt与summary一致性已核对。

M0此前在8条source上的独立Signal/完整prefix与严格重载一致性确实通过；
它没有覆盖本次完整heldout矩阵与既存B0特征的跨进程逐元素比较，不扩大它的证据范围。
模型参数冻结通过仍不能单独解释特征差异，当前原因待只读诊断。

固定诊断覆盖全部已读360条gallery，含5个64批与1个40尾批，比较四种Signal输出路径；
记录实际后端flags但不改变设置，只比较特征数值，不计算任何检索指标。
预算1440次记录前向，0optimizer/backward/checkpoint；原模型/配置/数据/门不更改。
原首64草案从未执行，因全矩阵失败包含尾批而在执行前改成全量覆盖，原稿留存。

不重训已完成fold0，不放宽逐元素门，不把失败改写为PASS，不启动未诊断的后两折。
后续动作须依据真实诊断另行登记；当前无完整三折结果、官方/dev访问或消融。
计划refine-logs/msvr310_trifusion_v1/BASELINE_PARITY_DIAGNOSIS_PLAN_20260906.md；
原失败evidence/trifusion_msvr310_trifusion_v1_comparison_failure_receipt_20260906.json。
