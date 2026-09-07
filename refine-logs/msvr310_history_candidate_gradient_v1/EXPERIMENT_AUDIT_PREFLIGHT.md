# 固定状态历史候选梯度完整预检审计

日期：2026-09-08。审计人：独立上下文gpt-6-astra/max。总体 **WARN**，`review_independence: same-family`、`acceptance_status: provisional`。没有发现梯度公式错误、伪造身份标签或报告表格数值不符；不宣称跨模型家族接受。

## A–F结论

| 项目 | 判断 | 核心证据与边界 |
|---|---|---|
| A 标签来源 | PASS | 全1032训练记录与原标签文件、三模态文件名及三折源身份隔离一致；分类class0保留。没有新增heldout/official图像前向；既有Q1工件的绑定读取仍存在。 |
| B 指标与梯度 | PASS | 同一64anchor目标，当前peers正常反传，候选VJP按原B64/RNG重编码并正确累加；同角色参数块比较，原fused Triplet权重1.0。 |
| C 工件与数值 | WARN | 25原始文本805652B、72step/135角色行、全部36表行和CSV一致；远端模型/矩阵未由审计agent独立读取。 |
| D 执行与覆盖 | WARN | 原计数覆盖缺口已由新增13项NumPy补核关闭；直接模型对照仍只每状态一次单历史组，其余多组是运行及文本证据。 |
| E 范围 | PASS | 仅seed42、九固定状态短预检，参数更新0；完整来源和检索提升未声称。 |
| F 评价类别 | 分别标注 | T0为synthetic_proxy；训练身份统计为source-only real_gt；梯度数值为真实身份损失条件下的数值proxy；旋转例子为数学分析；均非新增官方检索。 |

## 全量确定性核对

初审19,782项检查通过，无失败；全部9行工程表和27行角色统计表、135行CSV一致。范数闭合最大相对误差6.026005765920966e-12。135行历史梯度非零且超过其记录的同图重复差异，余弦/均值/分位数均由所有文本行复算。它们仍是已记录参数梯度的算术核对。

补核审阅127项检查通过，无失败。新verify_msvr_history_gradient_all_statistics.py正确复现所有13项producer统计；实际远端CPU对72batch完成936项核对和625920矩阵元素读取，结果/脚本/原summary/原CPU/protocol/config均有SHA绑定。审计agent复核代码及收据，没有本地重算九个远端矩阵。原19782项与后127项不是独立样本数。

补核脚本SHA：bb9c078f7800c2d5255c5922df68ed5072e6e93931347cb185b7dcce44987026。
补核结果SHA：434beb27dc6349a2d62ea64dedfa0efa17b41cfd2db8b11b7383a47923fd509b。
初审确定性JSON SHA：81b4a93c05c81cc51e2450100bd1fd0970778bc2c652b8d89f93b589b775074e。
补核审阅JSON SHA：cb6f843fae8ae02618b42044f1e8635cdff4dc9e04bb82f4f17cc78f3db413a8。

## 已关闭与保留事项

原CPU只复算13项中的6项，新增只读入口已补齐全部13项，原运行合同和原证据保持。tracker已区分预检/预检CPU完成与来源阶段仍运行，历史状态保留为有日期的观察。原报告7978字节作为修订报告的完整前缀保留，全部原数值表未改。

仍保留：14项总loss未保存逐项分量，无法独立重组；梯度、像素/重编码一致和状态SHA为运行时见证；直接图对照只有step4单历史组；短预检没有覆盖超过8batch上限的过期、512容量淘汰或零upstream跳过。递归继承的五个本地CRLF文件与绑定远端LF原始字节不同，不能混称所有原字节匹配。完整来源2340batch尚未在本包中完结。

后续执行者对源进程/GPU/磁盘的观察会单独保存时间戳与命令行；审计没有实时远端访问，不将这些较新快照称为审计agent独立核实。

共同正交旋转导致两侧偏导抵消的数学说明成立，1/√18槽位尺度与实现相符；文档明确未识别项目旋转分量，因此不把它解释成实测失败原因。

## 完整审计证据

[初审全部A–F原文](../../evidence/msvr310_history_gradient_preflight_20260908/independent_audit_initial/reviewer_full_response.md)、[补核审阅原文](../../evidence/msvr310_history_gradient_preflight_20260908/independent_audit_followup/reviewer_full_response.md)。两阶段原报告/tracker快照、独立脚本/JSON与原始文本分别保留在同一证据目录。私有完整调用trace留在本地.aris/traces/experiment-audit/2026-09-08_history_gradient_preflight/，不推送。

下一步完成原全来源/原CPU、另行13项统计补核、全量分析及新的完整来源审计。当前没有新训练/Q1或检索收益结论，长期Goal ACTIVE/UNMET。
