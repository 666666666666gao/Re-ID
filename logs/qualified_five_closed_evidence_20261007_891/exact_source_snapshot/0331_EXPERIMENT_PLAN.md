# 训练职责干预全六端闭合后的固定best分解候选

2026-10-04。本地准备，尚未登记到生产仓库、上传、启动或产生模型结果。先完成原六端及原一次CPU报告；不能根据已完成的四端先启动部分诊断或替换剩余训练。

## 明确问题

目前RGBNT201职责干预恢复了同variant检索，但native相对semantic仅增加0.0434mAP；MSVR310仅增加0.010120。独立global比较不能代替同一模型的global比较。本诊断只区分：全局表示是否恢复，以及固定部署修正对同一global究竟修复和破坏多少检索关系。

## 进入条件和证据封存

- 原`logs/global_task_role_v1_20261004_824`全部六端各自真实8步M0、fresh50和首次strict完成；原`results/global_task_role_v1_complete_20261004_824`唯一CPU报告及启动器exit0，完整300轮/12968步/18组配对文本接收并核验。
- 核对330项当前源代码、六端正式best/训练/首次评价/初始化/实际batch记录，以及独立global和必要历史参照的实际SHA。保留全部50轮，epoch只用原mAP-best。
- 新输入seal只绑定诊断需要的正式模型、距离、文本与源代码；不绑定已退休的旧M0文件，也不重新执行任何退休M0依赖的旧verify/report。
- 全报告证据接收后，原六份M0二进制才可按确切路径、实存SHA和原回执审计退役；保留M0文本。此处不执行清理。

## 最小执行

沿用现有固定best诊断，但源码复核确认原diagnosis.entry.configure会再次将foundation.condition覆盖为partitioned版本，而foundation.load严格要求当前objective条件一致。v2包装将诊断模块的entry明确指向run_global_task_role，并复用同一runner；每端都执行本轮真实configure，不修改任何严格断言。此修订只在本地准备，尚无模型执行证据。

每个正式semantic/native固定best，使用原eval loader，对每条完整query/gallery只执行一次原`forward_features`，同时保存g/c/h/f。按原顺序仅在26物理GPU0/1串行六端；无训练、优化器、测试更新、权重选择、parity修复或功温操作。不增新损失、倍率或外部资源。

保留现有模型state/buffer前后SHA相等、无参数梯度、初始化binding一致、原fused指标固定容差1e-5个百分点。逐端保存fused距离与原首次strict最大数值差；失败保留，不重试挑一次通过，不调容差。

全合法camera/scene过滤后报告：同模型L2(g)→f，独立global-only→同模型L2(g)，独立global-only→f。包括所有query AP、首个合法正例排名、首位修复/新增错误、身份宏平均及全部CMC。L2(c)只是诊断，不成为新部署输出。

native只沿用真实detail_reader和CNN首个output LayerNorm入口hook，记录实际detail相对加入前CNN-plus-anchor的能量；这是描述统计，不把槽位命名为真实部件。

图像、正式模型、特征和距离留远端；本地/GitHub只同步代码、合同、文本和SHA。复用核验过的conda环境，预计完整六端约8–12分钟；按实际开始时间安排180–300秒或接近预计终点的一次观察。

## 判断边界

相对独立global的差值不是同模型角色增量；聚合global loss/范数相等不是完整global状态轨迹一致。固定best诊断不能证明唯一梯度冲突原因、训练多种子稳定性或SOTA，也不能补充同容量原生来源控制。

依据全六端正式结果和全六端分解再登记唯一下一训练假设。N1未显出稳定增量时不自动叠加N2/N3，不靠weak baseline、训练配方收益或继承Signal能力凑三项贡献。
