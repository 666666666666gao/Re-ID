# 源码自审：独立fused训练头

审阅来源：root self-review，same-family/provisional；没有子代理、跨家族复现或论文评分。

主干、原semantic角色读取/桥接、1536维推理与原RAW身份目标均不改。只有角色分类BN/线性头从原头deepcopy并独立学习；不加一个新的局部辅助loss。原global头继续只学global，角色g/stages仍detach。两任务梯度与buffer归属经一次CPUdirect/vehicle合成见证验证。

新增头的可训练容量与独立BN运行统计明确披露，不能称严格同容量或原创模块。它不保证未知身份检索改善，不能从源SIM车辆增益推断头就是唯一原因。

CPU检查一次通过：初值全部输出/logits一致，任务梯度分离，新2/6张量累计活动/实际变化及完整组件state重载通过。合成fixture为4训练类，因此7680参数是fixture预算，不能冒充生产数据集参数量。真实新增参数由initializer按1536×(生产训练类+1)核算。

新M0记录所有参数活动、原global与新role各BN8次、新头实际更新及完整重载；保留旧AMP/full-batch/first-strict合同。全模型初始化、M0、正式训练均尚未执行，旧parity/读取detach/joint-L2失败不追认通过。

队列在CPU报告启动前将COMPLETE/report1持久化；这避免重复已观测的旧报告状态写入时序问题，未修改原失败。新三端完成后只一次六配对报告，不重复旧控制或搜索官方超参。各端验收后只退役自己的M0，全部共享依赖保留。

五个新Python AST通过，CPU输出和实际四份清理journal归档。该自审仅确认登记范围内的接入准备；生产验证与检索有效性均未获得。
