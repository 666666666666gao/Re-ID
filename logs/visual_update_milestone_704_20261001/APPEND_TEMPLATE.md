
## §41.704 RGBNT201首个完整视觉更新对照：更新边界有收益，角色额外增量仍薄

记录：{{RECORDED_AT}}。采用已登记十二端的顺序文本快照 {{SNAPSHOT_AT}}，目前父队列及子端严格验收均为 {{PARENT_COMPLETE}}/12。未完成端继续原50轮，不用中途best补正式结果。来源：`logs/visual_update_milestone_704_20261001/SNAPSHOT.json` 及其105份按字节/SHA封存的原始文本。

本轮新增完整端为RGBNT201 frozen_roles。它和low_lr_roles都采用本次统一FP32视觉存储、匹配初始化、同一新模块日程和完整图库；不是拿历史FP16冻结结果替代控制。每行四项来自各自同一mAP-best权重，完整50轮后严格重载、官方计分和worker验证均实际退出0。

{{NEW_RESULT_TABLE}}

{{NEW_C1_TABLE}}

这仅支持RGBNT201的roles条件下低学习率视觉更新有正收益。独立frozen_global_only尚未完成，无法据此判断更新是否对两种读出均有效，完整C1仍待六个配对及最终审核。同一low_lr边界下，已完成的roles减独立global_only为201 +0.1093 mAP/+0.3588 R1、MSVR +0.0673/+0.1692，两者仍低于原登记0.5 mAP要求；不改门槛，不从局部成绩宣布角色协作成功或SOTA。

新冻结端best在第2轮，第50轮为68.7972/69.8565/78.7081/83.7321，mAP从best下降3.7116。此前low_lr_roles和low_lr_global_only分别下降7.8972和8.4237。它们均是完整轨迹的观察，不能用不同best轮、单seed和目前不完整2×2确定唯一退化原因。下一步仍收齐全部训练边界控制，再决定共享全局/私有角色流的结构干预。

当前四卡继续三个RGBNT100端和MSVR310 frozen_roles；剩余三项frozen_global_only排队。既有240秒observer和父控制器继续，完整CPU报告调用次数{{REPORT_INVOCATIONS}}，只在全部十二端严格验收后执行一次。保存模型、图像、距离数组仍留远端；此快照空闲磁盘{{FREE_BYTES}}字节。222份活动源绑定、manifest、初始化witness、preflight和报告源均未改。

勘误：提交`02b22ba`的上一段误沿用了“§41.702”标题与702证据目录，应读作§41.703，实际证据目录为`logs/visual_update_milestone_703_20261001/`，91份原始文本；父队列4/12及四个完整成绩本身不变。该次私有staging回执也误写在702文件名，实际104个blob已另存703回执，未重新staging。历史正文保留，本段修正标签；本轮模板和证据目录独立704。原始UTF8严格解码通过，没有新增替换字符。归档源、真实运行回执与文档标签分别核对。

Goal继续ACTIVE/UNMET；普通视觉微调是训练控制，不作为方法新颖性。全十二端完成后再作固定C1/C2与成本、负翻转和身份收益分布的完整审核。
