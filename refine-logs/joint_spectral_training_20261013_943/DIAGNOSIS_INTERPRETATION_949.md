# §949：固定best用途与六项区域比较已完成

原三端50/首次strict/一次CPU配对报告已闭合。三个E8各自一次query/gallery固定权重提取完整通过：fused距离与首次official差值全0，before/after模型state SHA相同，0optimizer/grad；原metadata/order和作者camera评价一致。每端总耗时49.143/47.780/49.171秒，源代码/输入/特征数组SHA封存；没有改权重、重选epoch或第二次NN编码。

原NN remote三端均EXIT0，transport0；本地decoder遗漏base64导入造成工具终态1（be4a05）。随后只在本机解码第一次完整stdout并核SHA，8156dc EXIT0；新增SSH/NN/updates均0。原失败脚本与终态保留，不把source AST或单组件检查当全控制器运行通过，也不抹掉本地回收失败。

| 条件 | own global mAP/R1 | c单独mAP | fused mAP/R1 | gain*c/g范数比 | G→F角度 |
|---|---:|---:|---:|---:|---:|
| joint_delta | 74.2967/78.9474 | 3.8703 | 74.5331/79.1866 | 0.03425 | 1.951° |
| add_delta | 74.2967/78.9474 | 6.4034 | 74.4934/79.4258 | 0.04252 | 2.426° |
| joint_absolute | 74.2967/78.9474 | 4.5805 | 74.4006/79.1866 | 0.03578 | 2.038° |

三模型own-global的四项指标均与独立global-only相同；这证明本次选中best上的全局检索能力保持，不能把稀薄净增量归给已观测global退化。这里没有宣称所有权重/距离矩阵逐位相同或50轮全程都相同。修正单独很弱，但低独立mAP不是互补必然无效；本次实际净增量仅0.10—0.24点。不能再次靠增加gain期待收益，旧joint-L2的强修正也曾损害融合。

同一query集合c的公共均值能量/样本均方能量为81.867%、86.587%、79.127%；gallery描述相同。这仅是经验分解，不称公共内容全部噪声、已定位偏置根因或建议在测试集中心化。

按§947预登记，一次CPU调用完成三模式×处理后/reference共6项配对（12个局部ranking），新增NN/optimizer0。每项完全相同token与L2规范，完整gallery/camera过滤；比较全16×16cos均值与双向MaxSim均值，固定平均9role/modality组，无global融合系数/拟合/选择种子/重排子库。实际总CPU进程50.741秒，六次矩阵比较各6.1—6.3秒；逐query AP、首位修复新增错误、身份变化和6份距离SHA保存。

# 固定区域证据：均值与对称MaxSim比较

| 条件 | 来源 | mean mAP/R1 | MaxSim mAP/R1 | ΔmAP | 修复/新增错误 |
|---|---|---:|---:|---:|---:|
| joint_delta | processed_regions | 5.4876/2.7512 | 5.5352/3.3493 | +0.0476 | 21/16 |
| joint_delta | reference_regions | 7.9806/4.0670 | 9.9460/7.5359 | +1.9655 | 41/12 |
| add_delta | processed_regions | 11.0069/9.9282 | 11.5770/9.5694 | +0.5701 | 31/34 |
| add_delta | reference_regions | 11.2160/7.0574 | 16.7888/16.2679 | +5.5729 | 107/30 |
| joint_absolute | processed_regions | 6.8544/5.8612 | 6.5800/5.3828 | -0.2744 | 11/15 |
| joint_absolute | reference_regions | 8.0454/5.9809 | 7.9856/7.0574 | -0.0598 | 31/22 |

Mean of all pairwise cosine versus symmetric mean of row/column maxima on identical normalized tokens, within each of9 role/modality groups. Local-only scores; no global fusion coefficient, training, teacher, model selection, reranking subset or test centering. New diagnostic comparison rules use more stored vectors and pairwise computation; they are not original1536-vector formal results or complete ColBERT reproduction. Neither sign proves all late comparisons useful/useless or identifies compression/geometry as a causal failure.


最直接的判断：处理后区域最高MaxSim11.5770，语义reference最高16.7888，仍远低于own G74.2967。joint_delta处理后mean→MaxSim只有+.0476，joint_absolute为负；加法reference虽增加5.5729，也不是强global教师，更不能当完整模型涨5.57。当前未证明仅改变比较方式即可救回现有角色；也不能把这些局部规则的弱结果推广成全部late interaction无用。

因此不把MaxSim作为新主模块/教师，不调global+local倍率继续搜分。应将定位前移：原始CLIP patch在角色投影/读取前有什么可用身份信息，当前投影和加工是否保留这些内容。现有数据尚未回答原patch是否本来就弱，不能预先称单向量压缩、768→128投影或任何算子是唯一原因；§930全局敏感性遮蔽等已实现实验也不再作为未试新方案。

跨域来源核读：[ClearCLIP论文](https://arxiv.org/abs/2407.12442)及[作者仓库](https://github.com/mc-lan/ClearCLIP)区分CLIP的全局与dense特征，并改末层残差/FFN/self-self attention改善分割。它只启发检查局部特征形成路径，不能把原领域背景/残差结论追认成ReID原因；本次只读论文摘要/仓库，不曾运行或复制其实现。下一步先核本仓库真实CLIP输出与语义投影，再设计同一固定权重的有限诊断，保留global完整行为，不直接改主干或照搬整套分割框架。

所有原formal成绩不变，三端仍未超过全Signal。§948 exactD归档M0依赖及三个best保持；任何未来M0-dependent verifier需要先恢复probe。原3NN提取和CPU比较已终态，无新增训练/seed/gain/margin/rank/车辆/T-P/环境修改。三数据集超过Signal Goal ACTIVE_UNMET，后续训练须针对真实证据内容作实质改变，不重复封存的乘积、自由区域重建或旧选择性交互。
