# 角色度量输入与部署几何的单项对照

2026-10-05。候选已实现，尚无真实初始化、M0或正式成绩。前一轮六端及全量fixed-best诊断闭合后登记，不回改历史结果。

## 问题与唯一干预

Global职责保护恢复了部分绝对性能，但角色仍未提供稳定增量。固定best分解显示RGBNT100角色使同模型global的mAP下降0.278914/0.802213点，201/MSVR增量很薄。当前角色训练Triplet读取raw特征，而实际部署距离来自整幅1536维L2(h)。训练BN与raw距离对部分共同偏移不敏感，L2后的检索几何会变化；这支持检验度量入口，不证明唯一失分原因。

唯一改变：角色任务的Triplet输入改为实际部署的整幅1536维f=L2(h)，h=sg(g)+gain*c。原global作者任务仍使用raw g（车辆按原三模态512维）；分类logits仍由raw h进入原BN/head的stateless调用生成。原head参数detach、buffers克隆；persistent BN仍由global每batch更新一次。角色读取边界仍detach。

RGBNT201原单1536头只改变角色度量尺度。车辆原三512维头改为同一个整1536维度量，同时改变归一化及模态联合几何，不能称“仅取消/添加L2”。实际作者接口对各头损失求和，不是平均；本对照原样保留三份CE与三份Triplet的求和，后三份Triplet现在读取同一个联合f，等价于3倍该联合Triplet，未新增系数或另改损失聚合规则。CPU witness及登记保留这一边界。

模型forward、原global任务、结构/state/参数量、初始化、视觉/camera更新、optimizer/LR/scheduler/WD、soft-margin形式、gain、seed42、输入/增强/采样和推理全部保持前一轮。无新head、辅助任务、N2/N3、文本/SAM/DINO、记忆库、测试更新、参数搜索或parity修复。

## 匹配实验与判断

六端：semantic/native各自训练RGBNT201、MSVR310、RGBNT100。各端独立prepare核对初始化，8步真实M0通过后fresh50，首次严格重载。重用封存的global-task同variant六端为主要control；原独立global-only三端为共同能力参照。控制不重训、不重选best、不运行已退役M0依赖的旧verify/report。控制187原输入封存于global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json。

主要候选检验：相对匹配同variant mAP至少+0.5个百分点且R1不降；同时报告相对独立global-only，以及native−semantic。门槛是项目推进规则，不是统计显著性。每端完整50轮，同一份mAP-best报告所有指标，201另报R5/R10；不跨轮拼列。完整15对全query报告包含首位修复/新增错误、身份宏平均AP及分布。

若无增量，否定本具体入口干预的收益主张，不能扩称所有度量学习无效；若只有车辆变好，不能分离L2和模态联合因素；若semantic提高而native无增量，不归于原生细节；若global职责保护和度量入口仍不足，不自动堆N2/N3救分。最终原创性、角色必要性、强参照及完整流程多种子仍待后续验证。

## 执行与存储

只2026物理GPU0/1，单进程原完整batch，CLIP前6层GPU1、后6层/heads GPU0，最大并发1。201 B64/K8；MSVR B64/K4；100 B128/K16；沿用现有FP32/noAMP配方。原环境warm reuse，无环境重建。无功率或温度查询、设置、门槛或watcher。

预计全部六端约6–7小时；依上轮实际相应训练时间估计节点，在节点附近观察，长任务180–300秒，不重复启动。顺序201 semantic/native→MSVR semantic/native→100 semantic/native。每端失败保留，不按官方结果调整入口或超参数重试。

01:58实际磁盘剩余5,352,148,992B，原队列保留六best+六M0直至最后的预算不足。新队列事先采用逐端清理合同：每端完成full50/首次strict且原verify（含真实M0二进制检查）通过后，封存该端全部正式文件和M0文字文件SHA、验证结果、探针路径/大小/SHA；只删除该端m0_reload_probe.pth并写一次退役journal。最终report按已验收不可变文件SHA验证，不调用依赖被退役探针的旧verify。失败端未满足合同不删除。每端正式只保留best_map.pth，所有历史正式控制和作者/初始化权重保留。

预算六best+最多一个活动M0，每个预留384MiB，加2GiB原reserve，共4,966,055,936B；距离/文字占reserve。不会直接降低原失败实验门槛。封存清理支持需要在首次正式端完成时接受实际验收，再允许其探针退役。

所有source、plan及CPU witness登记并四执行副本同步后才启动。数据/模型/距离保持远端，文字/CSV/SHA同步本地、Desktop、GitHub及2026。2025保持已确认I/O pending，不恢复或再探测。官方基准已消费、当前单seed42，任何结果不得提前标为SOTA或三模块有效。总Goal active/unmet。
