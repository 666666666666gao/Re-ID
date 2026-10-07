# 同一图上的独立VJP autocast上下文对照

§881原六端未缩放独立活动门全部FAIL。§882实际新初始单图的scale1/256均六个Q/K非零，0优化且state恢复；因此仅调整尺度不能解释旧零值。

已核验源码的具体差异：run_foundation_recipe.py的loss_values位于autocast内，原run_incremental_role_objective.py在loss_values中调用autograd.grad，生产scaler.backward则在该autocast块外；此前两份单图诊断也在块外VJP。不能在没有控制测量时断言上下文就是旧八步唯一原因。

本项沿用一个RGBNT201 repair_keep公开初始状态、一批来源B64、一AMPforward和同一损失图。两次VJP固定原来的八个targets（g,c及六Q/K参数），固定scale1，不挂hook，不加入中间targets。仅将VJP的autocast上下文分别设为原检查的True和诊断/生产反向的False。分别保存unused/dtype/norm/max_abs，0优化、p.grad不累积、恢复BN等buffer后完整state相同。

成功执行标准只有真实单图、两VJP有限/global无梯度/精确state恢复，不规定应当测得哪种非零结果。若False活动而True为零，结论仅支持此图的上下文效应，后续可事前登记与实际生产一致的观测位置修订，保留原FAIL，不改模型或loss/scale/阈值；若无此差异，不继续押注该解释。没有训练、检索、新mAP或自动formal。

使用原48输入，新增脚本/本计划/两份独立源码review加到原402来源成为406。仅26物理GPU0/1/max1，25/GPU2/3/温度功率不操作。估计1分钟内，唯一observer首读3分钟后/后续240；异常保留不重启。复核通过、原诊断与observer结束并完整收集后发布启动。Goal ACTIVE_UNMET。
