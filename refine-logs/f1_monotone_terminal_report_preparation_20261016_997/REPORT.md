# 单调AP候选完整终端CPU报告准备

原定16:56新201端点和23:37原100节点尚未到。此前等待被goal续接中断，只读取本地登记与时钟，没有据此检查/重启远端NN。等待期间补齐一个只在完整终端执行的CPU报告器及其组件检查；这不是新的训练、消融、性能结果或额外中间NN查询。

新报告器仅消费status COMPLETE、完整50轮、同一mAP-best checkpoint的单调AP候选，及相同原F1权重/seed42/协议/完整预算的F1-R2控制；核验当前best和实际D SHA、连续50轮、原F1复现与监督，拒绝把M0/部分训练或旧完整Signal初始化当作匹配控制。输出同模型bank→fused、F1/Signal/匹配R2的完整四指标和修复/损伤/AP组，不新增推理、重排或距离插值。

对新增训练日志按65步hard Triplet、合法单调AP有支持/无支持分开汇总：三个角色encoder上的ranking/auxiliary范数与余弦、当前/历史余弦、完整ranking梯度与实际AdamW参数增量的内积和对应角度、零ranking范数、系数到1.6上限的次数。分类头和冻结F1不在这三个角色参数块统计中。原始范数不等于AdamW份额；正内积只说明当前候选集合的局部一阶上升方向，不保证有限步实际loss或测试损害。无可定义余弦保留不可用，不补成0。

每epoch首个有支持更新的分数导数样本单独列出，明确它不是全训练的关系梯度频率。新候选轨迹也不是原控制的反事实重放；不能单凭这些统计认证失败原因、原创性或泛化稳定性。

报告组件已实际本地CPU检查PASS：独立手算4query AP和首位修复/损伤、实际8步M0日志的2hard/6支持步/1分数样本，以及构造的参数步方向与零范数不可用案例。没有把8步当完整终端，也尚未执行full50报告CLI。CPU_COMPONENT_CHECKS.json记录真实成功输出及源码/输入SHA，不以重新测试制造科研结果。新分析器只用Python标准库，不导入网络/Torch或访问GPU。

完整终端通过后一次执行示例（不是已经运行）：

python -B tools/report_f1_monotone_terminal.py --run /data/gaob/Re-ID/Trifusion/trained-model/f1_monotone_ap_full50_20261016_995/RGBNT201 --control-receipt /data/gaob/Re-ID/Trifusion/trained-model/f1_initialized_r2_full50_20261015_988/RGBNT201/official_metrics.json --output /data/gaob/Re-ID/Trifusion/results/f1_monotone_terminal997/RGBNT201.json

MSVR/100使用相同新候选目录中的各自dataset，以及原f1_vehicle_prepare_continuation_20261015_990中同dataset完整F1/R2 official_metrics.json；只有原端COMPLETE才执行，不轮询或自动重训。分析源码独立于运行中的526份神经源码，部署前后逐项核验原526不变，学习实际版本仍9fe56905及原计划f7495552...。

科学账本保持：原F1 20177.2475/80.9809未达Signal，MSVR54.3826/72.7580四项已胜，原100最终端待23:37。新单调AP 20106:54 fresh50、原观察器34516只在16:56读取；MSVR随后、新100待原100后接续。没有提前检查本地原观察器/epoch/GPU或新增seed/参数候选；当前best、实际D和全部exact-D恢复义务保留。Goal ACTIVE_UNMET。
