# RGBNT100 Signal 工程修订R2合同

登记于2026-09-06T11:00:06.191514+08:00，READY_NOT_RUN。本修订只解决已捕获的数值失败，不是新TriFusion方法或科学负结果重跑。

## 已知证据与修正

原正式R1在60a3d0e首个epoch内AMP检查停止，成功更新数未持久化、无checkpoint/检索。
8b412d0独立source诊断第34步捕获异常，但其轨迹与旧M0不完全逐位相同，不能替原运行填入34步。
56f094f单batch完整重载复现原损失/153NaN梯度；原Gram半精度有3零det，AbsBackward0在sqrt(abs(det))报告NaN。
8034451只将Gram改FP32仍1零det/各512NaN，算子门失败，完整模型段未执行。
6c741b8真实输入回归将FP32 Gram与固定1e-12开方下限结合后通过：195梯度有限、4身份分量/Patch不变、3072D推理逐位一致。
本修订使用已通过的tools/signal_gram_stable.py；原Signal文件不改，在当前进程明确绑定useB.volume_computation3。

数值定义为sqrt(abs(det).clamp_min(1e-12))，下限取自作者Triplet现有开方保护值；
abs(det)低于下限时volume=1e-6、该det梯度0。这是明确的数值稳定化，与原无保护公式不完全等价。
不增加损失项、权重、网络容量、回退路径或可扫描超参数。所有配对候选若将来获准都必须使用同一数值基线定义。

## 最小工程变更及冻结内容

1. 原train_rgbnt100_signal_oof.py的new_model只安装该稳定化函数；helper无参数/缓存/随机数。
2. 每次尝试的步标量/实际source索引/AMPscale/是否更新先写steps.jsonl，再执行原AMP失败门。
   完整training.json与checkpoint仍在成功结束该fold后写出；失败记录不再因未结束epoch丢失。
3. M0由每fold8步改为每fold完整第1个source epoch，覆盖曾在34步出现的真实问题；
   这是工程检查覆盖范围变更，正式固定30epochs、仅seed42/B64K8、原noise-cosine/Adam及所有权重不变。
4. 初始CLIP/三折身份/全部8675内部query-gallery/原camera过滤/128×256拼图切片/增强/精确3072D检索不变。
   configure、数据函数、提取与排名函数AST逐一等同R1；旧runner原字节保存在run20 inputs中。

## T0与M0顺序、真实成本与停止

新runner/config改变了旧回执的源码绑定，因此对R2执行一次原全量T0以取得匹配本修订的真实回执，
不伪造或改写R1通过回执。T0重新核对8675文件/26025切片/全部query mask及2人工排序fixture，
17350图片解码、0模型/优化/真实检索；这是新工程源码绑定检查，数据协议没有重新选择。

随后新初始化三个fold各完整1个source epoch；原sampler实际完整batch数逐步记录，不用len(loader)猜更新数。
原195可训练梯度必须全有限、AMPscale不能下降、6个selector参数不变、严格重载8条源样本逐位相同。
没有100步新方法拟合门，因为这是Signal基线工程修订；不改变此前TriFusion新方法的门。
T0预计20–60秒，完整M0预计3–6分钟；首次检查安排在约4分钟，按实际终态决定后续，无盲目重启。
失败保留全部steps.jsonl/日志，停止定位；不会在本合同内改floor/精度/学习率/采样/seed或跳过异常batch。
模型/数据/张量仅在远端，Windows仅文本/JSON/AST与SHA计算。

## 后续完整基线边界

完整M0通过并核对全部实际步、checkpoint内容和源码后，才可单独登记/启动三个fresh模型的固定30epoch基线。
M0权重不用作正式初始化；无中间heldout评估/选best。完整8675query与gallery在各fold固定epoch30提取一次。
旧R1终态核验器的配置路径/源码绑定必须在正式启动前适配本R2，并保持真实全量算术复核。
独立审计按实际完成的工程/终态范围判定；本修订没有RGBNT100检索结果，不晋级主方法、dev或官方测试。

保留原失败日志、原8步M0、仅FP32失败、所有诊断成本及R1/R2的配置与来源差异，不覆盖历史。
