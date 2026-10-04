# 首项角色读取梯度边界对照：完整50轮与严格重载

| 条件 | best轮 | mAP | R1 | R5 | R10 |
|---|---:|---:|---:|---:|---:|
| 原独立global-only | 8 | 74.2967 | 78.9474 | 88.2775 | 91.8660 |
| 原semantic | 7 | 71.8981 | 74.4019 | 84.3301 | 90.1914 |
| 读取输入detach的semantic | 18 | 72.7798 | 76.9139 | 84.9282 | 89.2344 |

1. **对原semantic有窄范围正证据。** 同前向、容量、初始化、作者配方和实际批次顺序，仅改变角色读取的输入回传路径；mAP提高0.8817，R1提高2.5120，R5提高0.5981，R10下降0.9569。满足预登记mAP≥0.5/R1不降的推进线，不等于全部CMC改善、训练多种子显著性或SOTA。
2. **仍未超过独立全局适配。** 相对原global-only，mAP/R1/R5/R10全部为负。这份fused成绩不能说明global本身被保护；需要固定best的同模型g/c/h/f诊断分开其共同训练变化与实际修正效用。
3. **后期回落仍存在。** E18选中best；末轮mAP为70.0197，较best下降2.7601。保留完整50轮，而非按中途表现改日程。训练范数是step描述，不替代未知身份排序分析。
4. **只完成六项之一。** native自己的8次更新M0已通过，295/295非零有限梯度、BN8、重载差0，14个细节张量有真实参数变化；10:35:47已从匹配初始化进入fresh50。MSVR310和RGBNT100的四项尚待同队列执行。M0不等于检索有效，原九控制与封存parity失败保持不变。

Single seed paired intervention versus historical controls; no new seed uncertainty or causal gradient-conflict localization. First model improved mAP/R1 but Rank10 declined and all metrics remain below independent global-only. Same-model g/c decomposition pending; no claim that global was protected. Five endpoints and one final CPU report pending; scientific Goal active/unmet.
