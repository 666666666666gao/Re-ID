# 已有全source余弦诊断对应的固定0.3单位距离间隔

这是已有9个V24 source模型诊断的标量推导，非新训练或新检索运行；V24原Q1失败保持。
V8实际criterion先归一化embedding，再计算L2 batch-hard Triplet，margin固定0.3。
原诊断保存了每条source的最远非自身正例余弦p与最近异身份负例余弦n；因此推导
d_pos=sqrt(2-2p)，d_neg=sqrt(2-2n)，hinge=max(0,0.3+d_pos-d_neg)。
所有输入p/n实测严格在(-1,1)，无需clamp或fallback。结果来自序列化FP32余弦，非对原torch.cdist逐位重算。

| 模型 | source折内记录数 | 正hinge记录 | 比例 | 平均全source hinge |
|---|---:|---:|---:|---:|
| initial | 6252 | 400 | 6.397953% | 0.002149224 |
| ordinary_two_view | 6252 | 116 | 1.855406% | 0.000626920 |
| environment_identity_prototype | 6252 | 88 | 1.407550% | 0.000489391 |

每个模型条件6252是折内记录总数，同一真实fit图像在两个source fold出现；不是6252个独立样本。
全部9模型/18756条模型-记录行、每折94个身份统计均保存，没有只抽部分身份。

| fold | 模型 | 正hinge/全部 | 涉及source身份 | 最小单位距离间隔 |
|---|---|---:|---:|---:|
| 0 | initial | 158/2126 | 37/94 | 0.132645515 |
| 0 | ordinary_two_view | 35/2126 | 12/94 | 0.180486951 |
| 0 | environment_identity_prototype | 28/2126 | 10/94 | 0.191498453 |
| 1 | initial | 132/2075 | 27/94 | 0.115583185 |
| 1 | ordinary_two_view | 44/2075 | 13/94 | 0.148022841 |
| 1 | environment_identity_prototype | 32/2075 | 12/94 | 0.157890503 |
| 2 | initial | 110/2051 | 26/94 | 0.155227437 |
| 2 | ordinary_two_view | 37/2051 | 8/94 | 0.177782091 |
| 2 | environment_identity_prototype | 28/2051 | 7/94 | 0.196794600 |

原“source正负余弦间隔全部为正”仍成立，它只证明顺序；不等于0.3单位距离间隔处处满足。
另一方面，完整干净source的全局最难正/负约束也已经大部分满足，终点仅约1.4%–1.9%记录有正hinge。
原型终点进一步降低这些source违例，仍未通过heldout科学门；更好的source间隔不能替代泛化证据。

这使“直接扩为全source样本bank就会得到充足新监督”的假设需要更具体依据。XBM样本记忆仍不同于V24身份原型，
但不能用原型丢失局部难例这一事实，直接推导样本队列必然提升。本推导同时扩大正、负关系，不能将差别独归负例覆盖。
真实弱/强增强、持续更新后的缓存陈旧及未知身份未被本推导覆盖；没有证明这些训练视图的损失也为零或很小。
任何后继应先明确如何产生可保留身份信息的新环境变化，或证明其样本记忆在实际增强表示上仍有有效监督，
再固定一次完整配对训练；不扫描0.3 margin、重跑V24、访问dev/official或提前做结构消融。

相关源码：modeling/trifusion/signal_preserving_v8.py:696–711；modeling/trifusion/criterion.py:17–35；
tools/diagnose_v24_source_prototypes.py:82–124。原诊断SHAa66f17a450fb0eca2404fd23721545eed0dd7b061550230a7d51368da24fa271。
本次完整算术耗时0.2171912秒；0本地模型/张量/图片调用，0远端运行，0新增训练或检索。
实际原型与FIFO区别见docs/SOURCE_PROTOTYPE_MEMORY_RESEARCH_2026-09-05.md；本结果未登记新的训练版本。
