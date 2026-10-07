# 修正观测上下文后完成原增量目标比较

已实际闭合的单图对照证明：相同公开初始201、相同AMP损失图、原八targets和scale1，VJP在autocast内六Q/K为0、unusedFalse，在autocast外六Q/K全部非零。原生产backward和诊断块外执行；原活动检查位于loss_values/AMP块内。这个结果支持修正观测的上下文，不能追认为旧八步唯一根因或新检索收益。

旧原入口/六端0/6/原失败与48更新全部封存，不编辑旧源码，不降低非零/有限/八步/BN8/严格重载门，也不放大loss/gain或改LR/margin/seed。新入口仅在M0的独立autograd.grad外加autocast(enabled=False)，使用相同未缩放损失、g/c/六QK targets。新增记录unused与上下文。prepare/train/evaluate复用原构造、原损失；training objective、生产AMP、GradScaler、模型参数和1536维推理完全不变。绑定增加观测协议与实际入口SHA。

新六端M0从相同公开CLIP/新camera/head/seed42重新初始化，每端8更新；逐端比较原rawsemantic initializer共有字段和完整首八batch。沿用原生产M0、全参数累计活动、BN8、重载≤1e-5、独立g无梯度/c非零/六QK累计有限非零要求，另明确sixQK非unused。接受后仅工程probe退役。该新观测版本是另一资格合同，不能把旧FAIL改PASS。

只有新六端全接受，才启动新的六端正式50轮队列；正式模型重新公开初始化，不继承M0权重，严格首次重载评价、同一mAP-best及全部CMC、完整query/gallery与camera/scene过滤。原两目标md_batch_ratio与repair_keep固定各三数据集，不新增模块、参数、数据、头、排序重写或测试时更新。控制为已封存rawsemantic三端与独立global-only三端，不重训旧控制。报告15个配对（各目标与两类控制，及repair_keep−MD），所有身份/首位修复和新增错误/曲线/成本保留，推进线仍沿原合同，不根据官方分数修改。

原六端正式阶段仍未执行、不自动续接其旧campaign；新完整阶段有自己的M0/campaign/版本与修订记录。工程修正不是论文创新，MD式batch目标不是完整MDReID复现，repair_keep公式与V26/R2/CIRC已有先例；P1/P2/P3尚未完整实现或证明。

资源仅26物理GPU0/1/max1分段模型，无25/GPU2/3/功率温度操作；NN活动期间不source/Git/server sync。M0原2GiB+512MiB、full原5,192,548,352字节门保持。启动前只能清理已完成全部消费者且不在当前输入/赢家的OWN权重，留下完整严格评价/距离/轨迹/SHA与二进制退役边界。每个长任务唯一observer按预计里程碑/180–300秒观察，不重复启动。先执行新M0，再依据真实接受结果、实际存储资格启动full；不因预期通过而宣称已完成。Goal ACTIVE_UNMET。


Source review found a concrete report-subprocess scope issue. The old prepared full-stage code has never executed and is outside all prior executed 393/398/402/406 source scopes. It now exposes REPORT_ENTRY; checked full points this to report_incremental_checked.py. That wrapper sets the original report panel.source_map before invoking unchanged main. Six rows, fifteen pair analyses, thresholds and report math remain unchanged. The earlier prepared full-stage source review is superseded by this review; old executed training/M0 source bytes remain intact.
