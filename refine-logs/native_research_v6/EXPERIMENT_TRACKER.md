# 原生研究 V6 执行表

§41.809：九端完整50轮、各自真实8更新M0与首次严格独立评价全部收齐，450轮/19,452步；原全量CPU报告只执行一次并退出0。Goal active / unmet。

| 数据集 | 条件 | 同一mAP-best轮次 | mAP | R1 | R5 | R10 | 实际steps |
|---|---|---:|---:|---:|---:|---:|---:|
| RGBNT201 | global_only | 8 | 74.2967 | 78.9474 | 88.2775 | 91.8660 | 2649 |
| RGBNT201 | semantic | 7 | 71.8981 | 74.4019 | 84.3301 | 90.1914 | 2649 |
| RGBNT201 | native | 20 | 72.1273 | 75.1196 | 85.1675 | 89.3541 | 2649 |
| MSVR310 | global_only | 38 | 50.5421 | 68.0203 | 80.5415 | 85.4484 | 706 |
| MSVR310 | semantic | 49 | 50.9636 | 69.2047 | 80.7107 | 86.1252 | 706 |
| MSVR310 | native | 38 | 50.6755 | 68.6971 | 81.5567 | 85.9560 | 706 |
| RGBNT100 | global_only | 7 | 84.5338 | 96.6181 | 97.3178 | 97.9592 | 3129 |
| RGBNT100 | semantic | 5 | 83.4395 | 96.0933 | 96.6764 | 96.9679 | 3129 |
| RGBNT100 | native | 5 | 82.5610 | 95.5102 | 96.3848 | 96.6764 | 3129 |

| 数据集 | 配对差：candidate−control | ΔmAP | ΔR1 | 首位修复/新增错误 | 身份宏平均ΔAP | 原推进条件 |
|---|---|---:|---:|---:|---:|---|
| RGBNT201 | semantic−global_only | -2.3985 | -4.5455 | 46/84 | -2.1894 | 未满足 |
| RGBNT201 | native−semantic | +0.2291 | +0.7177 | 60/54 | +0.1012 | 未满足 |
| RGBNT201 | native−global_only | -2.1694 | -3.8278 | 45/77 | -2.0882 | 未满足 |
| MSVR310 | semantic−global_only | +0.4215 | +1.1844 | 12/5 | +0.3888 | 未满足 |
| MSVR310 | native−semantic | -0.2881 | -0.5076 | 4/7 | -0.2824 | 未满足 |
| MSVR310 | native−global_only | +0.1334 | +0.6768 | 9/5 | +0.1065 | 未满足 |
| RGBNT100 | semantic−global_only | -1.0943 | -0.5248 | 32/41 | -0.6606 | 未满足 |
| RGBNT100 | native−semantic | -0.8785 | -0.5831 | 31/41 | -0.1649 | 未满足 |
| RGBNT100 | native−global_only | -1.9727 | -1.1079 | 40/59 | -0.8254 | 未满足 |

全部CMC跟随各端一份mAP-best；每数据集三条件真实batch顺序SHA一致、共同初始化一致、source314未改。global-only含共享适配，native额外159,296参数，无同容量控制。阶段推进条件不等于显著性/种子稳定性/SOTA。

仅26GPU0/1，25文本同步；不管功温、不调参、不重跑/选新种子、不加N2/N3。旧反向parity FAIL仍封存。全部临时M0在本次发布时仍留待发布后精确清理；正式best/初始化/失败证据保留。完整文本与once-report见logs/native_research_complete809_20261004。固定best诊断仅提案，尚未启动。


本轮事前推进：native−semantic 0/3、semantic−独立global-only 0/3。保留201细节小幅正值/MSVR角色正值及其他负值，不称普遍无效或已有效创新。固定best只读诊断仅提案，尚未启动；不加N2/N3/新种子救分。


§41.810存储收尾：报告与§809同步核验后，9个精确无用M0探针已删，释放3196826712B。九份正式mAP-best/初始化/距离/文字证据均保留；旧M0依赖验证不可直接重放。检索结果不变，Goal active/unmet，未启动下一训练。


§41.811：固定六份正式best全量只读诊断已登记/输入实际SHA通过/源码复核，尚未启动。一个forward同时读g/c/h/f，实际gain，原native hook出口幅度；同模型global不冒充独立global-only。无新训练/参数调整/功温观察/旧M0依赖重放。完整科学Goal未完成。


§41.812六best全量只读诊断闭合，父子exit0/原fused复现/模型状态不变/原61依赖及316源码未变。所有原始query/AP/首正例、身份和幅度分布保留；二进制仅在远端，新文本归档logs/native_fixed_best_complete812_20261004。原九端0/3结论和M0退役不改。下一训练需基于完整诊断另行登记，当前没有新训练/调参/工程修复。科学Goal active/unmet。


§41.813原V6状态保持COMPLETE/九端0/3、原六best诊断COMPLETE。新训练家族role_input_detach_v1另行登记：六端只变读取梯度边界，同前向/容量/初始化/配方，尚未实际prepare/M0/full。原控制/旧M0退役不变，非parity修复。


§41.814原V6仍COMPLETE；新role_input_detach六项队列真实启动，201semantic自己M0通过、fresh50运行。原始证据logs/role_input_detach_launch814_20261004。正式0/6；不重跑旧M0、原nine报告或改变封存失败。
