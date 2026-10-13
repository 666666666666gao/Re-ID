# §945：当前三条件完成之后的固定best诊断

先说明状态：本次没有查询远端训练进度，没有新NN、optimizer、权重或正式分数。第一joint_delta已经真实M0通过进入fresh50；最后实际训练节点观察仍是23:54:44，1/50。唯一近第一full观察为原native23649/PID38852，到2026-10-14 00:42:58；原三端控制器保持，不能提前轮询、改配方或重复提交。

已有CPU报告在三端fresh50/首次strict全部完成后，分析保存距离的配对mAP、CMC、全query修复/新增错误与身份AP。新增tools/diagnose_joint_spectral_delta_best.py补上固定best的实际G/C/F及因子/reference/delta用途，防止分数失败后只能继续猜测容量或尺度。

执行先决条件：原parent campaign COMPLETE/3of3及原supervisor EXIT0；三端fresh50/首次strict accepted矩阵、各initializer、manifest、training receipt、M0依赖、唯一best、official及best-epoch距离均按原source/witness验收。CPU报告首次成功并保存SUMMARY/REPORT。之后才生成新诊断INPUT_SEAL，pin原488sources加本诊断源码/计划和复核，全部读取的正式artifact与independent global距离/receipt，以及原父campaign/EXIT/矩阵、CPU报告SHA。现在这些终态尚未存在，不能创建假seal或运行诊断。

原三端完成后，在同一26物理GPU0/1一次只读job顺序提取joint_delta、add_delta、joint_absolute的各自同一mAP-best。每端query/gallery按原loader/order各走一次，原CLIP在每batch只编码一次；输出真实shared_global、correction、raw_fused、fused以及actual/reference/delta descriptors和三模态因子。没有optimizer、head fitting、重选epoch、增加种子、换ranking/centering、修改因子/模式、额外训练、teacher或新比较器。

必须核对before/after模型state SHA相同、全部.grad仍None、同best初始化/epoch/指标，fused距离与原official保存矩阵最大差≤1e−5、mAP/R1/R5/R10复算一致、query/gallery完整metadata与independent global完全相同，并用作者原评价与当前camera scorer交叉核读。失败保留第一次日志、EXIT和数组，不重新挑checkpoint、重试挑通过或临时改容差。

输出G/C/F检索及same-model G→F、independent global→own G、independent global→F完整query/身份变化；记录gain、scaled-c/g范数、角度、actual/reference/delta和因子的样本均方能量、样本公共均值能量、中心化能量与实际rank weights，role顺序cnn/transformer/mamba、modal顺序RGB/NI/TI。不把公共能量直接叫噪声，也不把差分叫纯交互；LN因子并非严格三线性，slots没有真实部件对应，固定best诊断不是因果消融。

query/gallery特征、因子与诊断距离保存一次并SHA；预计三端合计约2–3分钟NN提取，按首端真实时间校正。原训练/严格验收有任何失败就先处理原失败，不能通过新诊断填正式分数。固定best诊断最多每端一次，无blind rerun；NN完成后CPU汇总与原配对报告一并更新交接/Git，再D盘归档核验无后续消费的重复M0，唯一best/数据/源码/唯一日志保留。

首次warm CPU import8cef58 EXIT0，只验证本脚本编译/import及所调用build/load、metadata/score/paired API存在；0model/0NN/0update/0diagnose，没有真实seal、checkpoint、特征、distance parity或检索结论。Root按code-review-excellence复核，不是独立agent或运行审计。查到实际runner.distance_matrix会L2规范G/C输入，官方extract无AMP，故诊断使用同一float32默认推理与原loader；load/condition/builder按producer.MODE准确设置目标mode，训练入口不改。

三数据集全面超过Signal的GoalACTIVE_UNMET。三个候选的有效性、新颖性、车辆适配与最终多seed仍需真实证据；当前不扩车辆或seed，不把准备工具当结果，不从旧对照的小增量推断新模块成功。
