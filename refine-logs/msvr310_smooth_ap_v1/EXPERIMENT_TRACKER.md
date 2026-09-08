# MSVR310 Smooth-AP v1

当前执行入口：§41.192；Smooth-AP M0独立审计已关闭WARN/确定性PASS，原Q1继续。09-09 00:29原wrapper48170/Q1 49157存活，fold0control固定260步完成，smooth_ap5/20epoch；1/6端完成，无完整检索终态。下一观察约00:40确认候选端后预热耗时。Goal ACTIVE/UNMET。

独立M0审计已完成：总体WARN，工程PASS_WITH_LIMITS，deterministic_checks_status=pass；gpt-6-astra/max新上下文，同模型家族/provisional。178项递归哈希匹配，1032来源triplet/3096模态路径/155身份、全部780batch sampler与真实队列重建通过；248更新、15872anchor AP、4945920四空间距离均独立核验。逐anchor AP最大误差2.3559432371644817e-7；每目标每dtype1236480距离位置，Smooth-AP解析导数对部署实现最大差Float64=3.122502256758253e-17、Float32=1.7623613799214177e-8，hard导数精确一致；14项总和最大误差5.62518835067749e-7。六个checkpoint各472张量完整/冻结状态及三源初始化绑定重建匹配。

WARN保留：真实图像内容/压缩包未重读；历史字段/RNG/向量和逐步optimizer/参数梯度未保存，不能从距离导数及范数恢复。六次direct-check只覆盖step4首个历史group的189encoder张量，最大相对L2=2.0929868300782436e-5属于运行见证及摘要恒等式核验；strict reload前后输出数组未保存，输出逐位相同仍是执行时断言，完整checkpoint状态则已独立核验。203/203是累计覆盖；固定过拟合只有53独立记录、无历史候选，不证明历史反传过拟合。审计未消费Q1成绩，未要求重训、改合同或改代码。

原文、独立脚本/输出、快照及失败尝试归档evidence/smooth_ap_m0_audit_20260909，正式报告refine-logs/msvr310_smooth_ap_v1/EXPERIMENT_AUDIT_M0.md/.json。一次审计transport捕获接口失败和13份副本文本CRLF归一化均已保留/修复，最终232/232远端文本快照、29/29原intake文件字节匹配。私有请求/调用trace留本地.aris与审计目录，公开清单明确排除项。不得重复已关闭M0审计。
