# Global身份学习与角色修正职责：唯一下一训练干预

2026-10-04，基于完整读取detach六端及其固定best全量诊断登记。状态：实现准备中，尚无本方案真实初始化/M0/正式成绩。

## 问题与唯一假设

当前作者head只监督h=g+gain*c；读取detach并未截断h到g的梯度。RGBNT201的主要缺口位于同模型global，RGBNT100的主要缺口位于有害修正，MSVR收益主要位于global。故不能只以“减少某条角色输入梯度”作为全局保护或角色互补的证明。

检验一个训练职责干预：shared视觉/camera/九个共享适配器及原作者head仅学习原作者global身份目标；roles/readout/gain仅学习使用当前同一身份头的fused身份目标。保持原三模态语义/native结构、参数/初始化、1536维部署、作者配方不变，不增加持久分类头或外部资源。

这是一项训练控制，不预称论文新模块。它同时检验角色能否在直接学习身份的global之上提供增量，以及融合头共同适应是否掩盖了修正的兼容性。一次完整干预并不能单独分离上述两个因素，亦不证明唯一梯度冲突原因。

## 实际梯度合同

- 使用上一方案的DetachedSemanticTriFusion/DetachedNativeTriFusion，角色读取stages/context/shared_global保持stop-gradient。
- global原作者head正常调用一次，计算L_g=原作者CE/soft-margin Triplet(g)。
- 融合训练特征为h=sg(g)+gain*c。原作者head的参数detach、BN buffers克隆后stateless调用，计算L_f=同一作者CE/soft-margin Triplet(h)。不将融合调用的BN运行统计写回原head，不让L_f修改head或shared参数。
- 一次backward处理L_g+L_f；每个参数组只接收其原任务量级的梯度，不扫描附加系数。原optimizer/LR/WD/scheduler/AMP/采样/增强不变。
- 原持久BN仅global调用更新，每8步M0仍须num_batches_tracked=8。融合stateless BN按当前batch训练计算，不积累第二套运行统计。
- 推理保持原f=Normalize(g+gain*c)，不用任务损失、stateless训练head、测试标签或更新。eval forward/state初值与匹配旧模型应一致。
- 该合同隔离直接损失梯度，未保证全50轮global与独立控制逐位一致；随机/数值执行及共享AMP仍需如实记录，不重启旧parity修复。

## 对照与必须报告的收益账

三数据集×semantic/native，共六端；每端自己prepare/8步真实M0通过后fresh50与首次独立严格评价。原公开CLIP、新camera/head、seed42、实际作者batch分别201 B64/K8、MSVR B64/K4、100 B128/K16。双卡完整batch分段不变，只2026物理GPU0/1，单队列。

直接控制为刚结束的相同variant读取detach六端；保留原V6相同variant和独立global-only三端作为历史强参照，均用固定best/原全query-gallery及过滤，不重训/重选/重跑已退役M0依赖的旧报告。控制仅绑定正式best、评价距离/回执、训练/初始化、batch顺序及固定诊断证据。

主要成功标准仍为相对匹配独立global-only mAP至少+0.5个百分点且R1不下降；同时报告相对读取detach同variant、原V6同variant、以及native−semantic的全部指标。恢复global基线本身不能记作角色新增贡献。每个模型一份mAP-best；201 R5/R10及车辆CMC跟随同权重，不能拼列。

完整六端所有query AP、首个合法正例排名、首位修复/新增错误、身份宏平均AP及分布、完整50轮曲线、真实global/fused两任务loss、参数变化、有效更新、显存/时间均保留。不根据首端官方成绩改loss/gain/seed/LR/batch；不加入N2/N3或其他读出。

## 执行顺序与边界

1. 源码自审和CPU结构witness先验证：state/参数量不变，融合head不改原BN统计，L_f不向shared/head反传，L_g不向roles/readout反传。witness不代表完整模型或检索性能。
2. 封存324项既有执行源码、新模块/入口/队列/报告/计划/验证入口及明确的正式控制artifact SHA；核对原科学source未变、当前控制不依赖已退役探针。发布后才能prepare/M0/full。
3. 每端真实8步M0检验原参数覆盖恰好一次、梯度有限、实际更新、author BN计数8、native14张量累计活动、完整state重载；不因诊断源梯度不同恢复parity门。
4. 六端顺序201 semantic/native、MSVR semantic/native、100 semantic/native；预计约6—7小时（原同规模队列约6.4小时），按180—300秒或预计里程碑观察，观察超时不重启。
5. 完成50轮/首次严格评价后原全量CPU报告仅一次；自身M0探针保留至其依赖闭合，再按明确路径/回执/SHA退役。只保留每端正式mAP-best及必要历史控制。

2025 `/data2/gb/Re-ID/Trifusion`在§822同步时返回真实I/O错误，文字镜像待恢复后补齐；不将其待补作为2026计算失败。当前只核对本地/Desktop/GitHub/2026，不冒称五份同步。不查询/设置功率和温度，不换环境或训练数据。

全量结果为已消费官方基准上的seed42探索；不代表完整流程多种子、来源独占容量控制、算子必要性或同协议SOTA。无稳定角色净增量时保留负结果，不把普通global身份目标包装为三个有效创新。Goal active/unmet。
