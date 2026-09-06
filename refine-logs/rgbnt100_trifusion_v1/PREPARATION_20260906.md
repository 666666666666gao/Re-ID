# RGBNT100 原完整三角色比较准备

准备时间 2026-09-06T12:27:04.515177+08:00。状态 PREPARED_SOURCE_ONLY_AWAIT_COMPLETE_B0；没有可执行正式配置，也没有启动M0或角色训练。

当前Signal R2三折固定30epoch基线在运行，必须先完成全8675内部query/gallery的唯一终点以及全部源文件、
权重内容、距离、排名和步骤算术核验，才能把实际完整summary与三checkpoint的SHA绑定进新配置。
本准备不访问当前未完成基线的heldout中间结果，不借用其他数据集或fold权重，也不改变其运行中的代码。

这是第三个新数据集上的整个原三角色主模型比较。MSVR310与RGBNT201既有科学失败保持；
不是重跑同一个已失败数据集版本，也不是单/双角色、容量或融合消融。
唯一待测问题是：在跨摄像头正关系更丰富的RGBNT100，原完整三角色是否稳定超过其同fold训练Signal。
本次R2 M0的相同身份正对57568对中，49473跨camera，85.938%（仅M0已采样关系，不能当作完整训练结果）。
结果若仍为负，将进一步限制“单靠增加跨摄像头监督覆盖即可解释/解决弱融合”的判断，不能预设相机是唯一原因。

## 已准备源码

tools/train_rgbnt100_trifusion_oof.py 从已执行MSVR310主模型入口直接移植，四个函数AST保持相同：
build_model、frozen_state_sha、output_mapping、engineering_checks。
比较公式及原五项支持条件不变，只把全量计数改为8675query、50heldout身份。
训练与模型本体、原七头ID/Triplet、AdamW3.5e-4/wd1e-4、20epochs/5warmup/seed42、B64K8/workers4保持。
输入128×256/grid8×16，三条路径均执行；baseline3072D/fullbranch4608D/fused7680D。

数据读入直接复用R2 RGBNT100 montage loader，索引来自单拼图path，去除MSVR310的三路径tuple假设。
评价使用RGBNT100 camera_scores和作者eval_func，严格删除同ID且同camera项；其他同camera身份留作负例。
五个完整输出保存全量features/distances/全部排序，排名使用无损gzip，避免把五份大型完整图库排序保留为未压缩文本。

推理复用未改动的tools/msvr310_exact_signal_inference.py。
这是针对同一PyTorch2.5.1、同一SIM投影在requires_grad不同情况下改变mm/bmm dispatch的既有实测问题。
只用detach后的functional_call视图恢复B0执行路径，不更改注册参数/冻结状态，不构建梯度图，没有fallback。
M0独立Signal调用也使用相同投影视图；正式全部baseline features及distance仍须逐元素等于B0保存数组。
这是一项尚待RGBNT100实际M0验证的工程移植，不将MSVR310 PASS冒充本数据集PASS。

新训练入口把每步全部损失、source索引、AMPscale/是否更新以及每个gradient finite结果落盘后再判断停止门。
这是保留本次真实早停现场的记录修复；正常有限步骤的数学更新不改变。
未完成步骤保留而停止，不跳过batch、不扫描数值下限或优化器。

## 后续固定顺序

1. 完整B0及原终态核验通过后，绑定真正summary/3checkpoint/arrays/协议/源码SHA，生成正式新配置和合同。
2. 每fold fresh角色/分类头，仅加载对应fold的固定epoch30 Signal；
   三折各8步真实容量和8条source的五输出严格重载，再新构造fold0固定一个增强batch恰好100步过拟合。
   原七头解析label-smoothing下界与final excess ratio<=0.1门保持；共124更新/7936训练记录前向。
   source parity每fold24条三角色+8条独立Signal前向，共72+24；heldout/dev/official为0。
3. 全部M0实际权重/步骤核验通过后，三折重新从M0同初始state开始各完整20epoch，不加载M0训练权重。
   预计总更新约5140，实际以完整sampler日志为准。按MSVR310实测约1.3秒/步粗估100–140分钟，
   以本数据集真实M0计时更新ETA；不得为缩短耗时减小epoch或只选部分身份。
4. 最终一次完整8675图库记录前向产生五输出；全部43375 query-output、50身份、三个fold及全部实际步骤核验。
   baseline与三角色不是总训练计算预算匹配；参数/FLOPs解释及结构消融仍在主结果成立后。
5. 五项支持条件仍是fused gain>=1pp、每fold非负、各fullbranch不低于Signal、
   50身份query加权bootstrap(seed42/10000/2.5%linear)下界>0、fused严格最好。负结果完整保留。
6. 独立审计当前遇服务额度限制，无完整新审计verdict；后续按实际可用性保留未闭合标记，不能由执行者冒写审计结论。

本准备没有添加新科学模块或损失，没有为已封存V23/V24/MSVR310修改条件，没有官方评估或SOTA成功声明。
