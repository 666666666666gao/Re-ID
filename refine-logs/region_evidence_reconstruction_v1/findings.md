本轮主张复核：no；same-family/provisional。

**结论：claim_supported = no；confidence = high。** 候选重建 v1 不能继续写成已验证有效的主模块。本轮完整结果允许保留负结果/消融，以及 RGBNT100 上特定固定模型的 CMC 收益观察；它们未证明“patch 查询提供优于均值广播的有效局部身份证据，并在三数据集具有净收益”。

审阅归属为 `review_independence=same-family`、`acceptance_status=provisional`。请求路由为 `gpt-6-astra / max`；运行后端及实际 effort 无法在本审阅中独立核实，未声称已验证。高信心仅针对本轮主张不获支持的判断，不代表所有重建实现或局部信息永久无用。

**证据范围。** 六端均已完成 seed42 fresh50 和各自单一 mAP-best 的第一次 strict 验收，合计300轮、12968次正式更新。三数据集新两臂的初始完整 state 相同，重建模块均有105232个活动参数、15个张量，全部15配对的实际 batch 顺序相同。两臂都从原 semantic CNN patch 取输入并使用图像条件 anchor；mean 仍保留原空间 patch，只将重建残差的 query 广播。它不是独立 global、无局部输入模型或原生图像 CNN，旧 factory 的 `native` 名称不能改变这个定义。

每个新模型均按 mAP 选同一份权重报告所有 CMC；RGBNT201、MSVR310、RGBNT100 两臂的 best 分别同在 epoch8、38、26。[原完整终态报告](C:/Users/gb/.codex_tmp/independent_evidence_draft/region_reconstruction_completed_admin_intake858/received/results/region_reconstruction_v1_complete_20261005_858/SUMMARY.json)与[选定权重表](C:/Users/gb/.codex_tmp/independent_evidence_draft/region_reconstruction_complete_analysis859/selected_checkpoints.csv)提供如下数值：

| 数据集 | patch mAP / Rank-1 | mean mAP / Rank-1 |
|---|---:|---:|
| RGBNT201 | 74.339751 / 79.425836 | 74.678481 / 79.545456 |
| MSVR310 | 50.542936 / 67.851102 | 50.542135 / 67.851102 |
| RGBNT100 | 83.499203 / 96.618074 | 83.558437 / 95.743442 |

下表均为候选减控制，单位为百分点；首位修复/新增错误取自完整 query 配对。Rank-1 差值沿用保存距离上的配对计算，展示精度可能与两个浮点官方分数直接相减有末位差异。[全部15组配对](C:/Users/gb/.codex_tmp/independent_evidence_draft/region_reconstruction_complete_analysis859/fifteen_paired_comparisons.csv)：

| 数据集 | 比较 | ΔmAP | ΔRank-1 | 首位修复/新增错误 |
|---|---|---:|---:|---:|
| RGBNT201 | patch − mean | -0.338730 | -0.119617 | 5/6 |
| RGBNT201 | patch − raw_semantic | -0.035111 | +0.598086 | 7/2 |
| RGBNT201 | patch − raw_global_only | +0.043087 | +0.478469 | 4/0 |
| RGBNT201 | mean − raw_semantic | +0.303619 | +0.717703 | 7/1 |
| RGBNT201 | mean − raw_global_only | +0.381817 | +0.598086 | 8/3 |
| MSVR310 | patch − mean | +0.000801 | +0.000000 | 0/0 |
| MSVR310 | patch − raw_semantic | +0.000711 | +0.000000 | 0/0 |
| MSVR310 | patch − raw_global_only | +0.000807 | -0.169205 | 0/1 |
| MSVR310 | mean − raw_semantic | -0.000090 | +0.000000 | 0/0 |
| MSVR310 | mean − raw_global_only | +0.000006 | -0.169205 | 0/1 |
| RGBNT100 | patch − mean | -0.059234 | +0.874636 | 22/7 |
| RGBNT100 | patch − raw_semantic | -0.591102 | +0.758017 | 46/33 |
| RGBNT100 | patch − raw_global_only | -1.034582 | +0.000000 | 39/39 |
| RGBNT100 | mean − raw_semantic | -0.531868 | -0.116618 | 36/38 |
| RGBNT100 | mean − raw_global_only | -0.975347 | -0.874636 | 29/44 |

主要六个 patch−mean/patch−RAW-semantic 比较 **0/6** 通过；将同一推进门应用于全部15个报告配对，结果为 **0/15**。三组 patch−mean 为 **0/3**。门槛是 ΔmAP≥0.5 且 Rank-1 不下降，这是[预登记项目推进规则](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/region_evidence_reconstruction_v1/EXPERIMENT_PLAN.md)，不是统计显著性检验。

**这些数值怎样限制结论。** RGBNT201 的 patch 在 mAP、Rank-1/5/10 上均低于 mean；query AP 改善/恶化162/311、身份改善/恶化8/18，身份宏 AP 为−0.344105 pp。mean 相对 RAW-semantic/global 的小幅正增益高于 patch，因而不能把本轮任何正向变化归功于 patch 特定查询。

MSVR310 的 patch−mean 只有+0.000801 pp mAP，报告的 CMC 完全相同，首位修复与新增错误都是0；query AP 改善/恶化13/15。完整50轮的 Rank-1 曲线也相同，mAP 小差值随 epoch 变号。它支持“这次训练表现非常接近”的描述，不能证明统计等价、结构等价、模块没有活动或局部信息无用。

RGBNT100 的 patch−mean Rank-1/5/10 分别为+0.874636/+0.699708/+0.641399 pp，首位修复22个而新增错误7个，允许报告固定模型上的 CMC 收益。但 mAP−0.059234 pp、身份宏 AP−0.061192 pp，query AP 改善/恶化687/690。Rank-1 只要求至少一个正确匹配排到首位，mAP评价所有有效正匹配的整体排序，因此二者可以不同向；这些数字不能合并成总体检索净提升。相对独立 global，patch 的 mAP 还下降1.034582 pp，Rank-1 持平。相对 RAW-semantic 的 mAP下降0.591102 pp也不能用Rank-1提升抵消。

六条完整训练曲线均已阅读。RGBNT201后期部分 epoch 的 patch mAP高于 mean，但两臂都低于各自epoch8的best，不能事后改选last/某一局部时期救分。RGBNT100后段反复出现CMC较高、mAP较低，也只是同一次训练的相关观测。新增300轮及包括旧九控制的750条曲线记录都不等于独立训练种子；15配对共享模型与query，也不是15个独立重复。

**机制与不确定性。** [源码](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/region_evidence_reconstruction.py)确认128个patch查询16个带图像均值条件偏移的学习anchor，再作残差注意力变换；mean以重复均值查询同样的anchor。它没有独立部件标签或局部重建真值。局部query形式、M0活动参数、若干query修复都不足以证明真正部件对应或新增局部身份证据。两臂都有动态图像anchor，本轮也未识别其必要性。相对RAW-semantic/global的比较还包含容量改变及报告注明的可能随机数消耗差异，机制归因应优先依赖同初始化的patch−mean，而该核心对照未显示一致效用。

保存的身份宏AP bootstrap区间（patch−mean）为RGBNT201 [−0.721771, −0.043164]、MSVR310 [−0.000064, +0.003315]、RGBNT100 [−0.672576, +0.545656] pp。这些区间只在已经训练并经过mAP选择的固定模型上重采样身份；估计量也与query加权mAP不同。它们不覆盖训练seed、checkpoint选择、反复开发的不确定性。RGBNT201负区间可作为条件性的负向诊断，不能写成跨seed显著劣于；另两区间跨零不意味着等价。微小差值本身也不提供等价证据。官方基准已消费，不能称未触碰测试集或据此声称SOTA。

**执行链路与科学判断分开记录。** [终态collector](C:/Users/gb/.codex_tmp/independent_evidence_draft/region_reconstruction_completed_admin_intake858/stdout.json)记录164个文本SHA、354项科学源、187件旧控制及六份验收。审阅另读了原EXIT/磁盘断言、补完PLAN/campaign、6/6 accepted_matrix与六份验收，并核对本次指定九个输入的本地SHA。原父进程因最后RGBNT100 mean训练启动前的2GiB磁盘预留断言以EXIT1结束；原失败未覆盖。另立行政补完以EXIT0完成此前未启动的fresh50、首次strict和唯一原15配对报告，继承此前五端和六个M0，没有重训五端，也没有改种子、配方或科学门槛。旧manifest中的PENDING项属于保留的准备快照；完整结论由补完终态及验收支撑。行政完成不提升机制证据强度。

[预检查记录](C:/Users/gb/.codex_tmp/independent_evidence_draft/region_reconstruction_claim_review859/EVIDENCE_PRECHECK.json)如实标注ARIS canonical `evidence_check.py`未解析，未伪称它已经通过。完整collector验收属于已有证据链，本审阅未重跑collector、检索、推理、原CPU报告或bootstrap。指定根目录未提供独立 `EXPERIMENT_AUDIT.json`，因此 `integrity_status=unavailable`；没有将哈希存在、工程验收或行政EXIT0写成科学有效性的PASS。

**建议替换为以下主张：**

> 在已用于开发的三个基准上，我们以seed42、fresh50和各自单一mAP-best，比较了同活动参数、同初始化的patch query与图像均值query候选重建。patch−mean的mAP差值分别为−0.338730、+0.000801和−0.059234个百分点，三组均未达到预登记的ΔmAP≥0.5且Rank-1不下降的推进门。RGBNT100出现CMC收益，但mAP未改善；相对原RAW-semantic及独立global也未获得三数据集一致净收益。因此本轮未验证patch特定候选重建提供有效局部身份证据，将其保留为负结果/消融，不作为已验证有效的主模块。

**后续证据要求仅限条件性建议。** 关闭本轮主张不需要补跑。若另行决定重新打开功效主张，需要预先固定选择规则的独立seed配对重复与独立评估；若保留局部机制或anchor必要性主张，需要直接隔离相应因素的证据。这些是证据缺口，不是已经设计或授权启动的新方案；不以尺度搜索、额外堆叠、换epoch或放宽门槛救分。当前也没有证据把结果归因于学习率、零初始化或容量不足。

本审阅仅创建本目录的 `REVIEW.json` 与 `CLAIMS_FROM_RESULTS.md`，未SSH、执行模型/检索/原报告、改科学源或方案、删除权重或开展其他工作。



约束：本轮局部query不获净增益；不按已消费官方成绩调LR/gain/margin/seed；下一项不得用更多anchor或残差能量替代互补证据检验。
