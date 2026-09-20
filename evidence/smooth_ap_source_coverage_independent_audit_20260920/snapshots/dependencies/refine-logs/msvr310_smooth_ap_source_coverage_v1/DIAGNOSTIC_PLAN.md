# 固定终态来源候选覆盖诊断 v1

状态 REGISTERED_NOT_RUN。只有来源诊断，无新训练、无Q1或官方测试。主目标与原Smooth-AP失败门保持。

采用原Smooth-AP六个epoch20终态（3fold×control/smooth_ap）；每端clean与一次seed42增强，全部672/683/709来源记录，共12条件、8256记录前向。五个部署输出，模型eval，原Signal数值执行修复保留，无V27统计扰动。每条件前后完整state SHA不变、无参数梯度；配对像素与Signal特征要求逐位相同。先执行全部提取，再执行CPU全范围分析；不可因中间指标改范围。

每条件只有一份表示表。每个原260步batch的64个anchor位置，候选池由该步current与history记录索引并集组成，按全局record_index去重、排序；始终排除query自身记录。所有距离从该表读取，诊断使用Float64平方欧氏距离与稳定排序（并列时按全局来源记录序）；不改原Q1注册FP32指标。

分别统计all-identity与cross-scene。后者只排除same-ID AND same-scene，异身份干扰均保留。没有合法正例的anchor不计算AP，仍保留其图像作为其他身份的负例。保存所有全图库query行以及所有批次anchor位置的候选/全图库配对行；仅在共同合法集合上比较AP，并单列池内缺正例数及完整来源合法query。重复anchor是曝光，不是独立图像。

范围：120个输出/过滤条件、每个16640个anchor曝光，共1996800配对行。固定view、唯一记录池不同于训练时重复记录的随机view/dropout权重；AP差只描述候选覆盖效应，不能独自证明分数校准错误、训练参数梯度大小或身份外泛化。

工程检查：真实checkpoint及日志哈希，递归训练context原绑定，模型state与梯度，五输出有限/非零/形状，全部索引/source身份隔离、像素配对。最小语义测试覆盖self、same-scene排除、无正例身份的干扰保留和多正例AP。完整CPU独立核验需核对所有保存行和特征，结果未核验前不作机制结论。

运行：python -B -m tools.diagnose_msvr_smooth_ap_coverage --contract configs/MSVR310/TriFusion-smooth-ap-source-coverage-v1.json --root RUN_ROOT --mode extract；退出0后同参数 --mode analyze。使用持久wrapper、阶段日志及退出收据，观察超时不重启。仅复用tri_reid环境，W&B=false，无环境安装。

五输出数组预计811597824字节，完整关系文本预计约1GB；开始时输出卷至少2GiB，当前实查约8.17GiB。模型/数组/详细关系留远端，代码与汇总文本同步。本阶段不删除任何依赖权重。预计数分钟提取、数分钟CPU；真实首条件耗时后修正观察时间。参数梯度分解另行实现，此运行不宣称完成该测量。
