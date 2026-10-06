# 新增目标正式阶段：补充源码审查 878

日期：2026-10-07。最终结论：**SOURCE_ONLY PASS**。当前 BLOCKING：0；当前 NON-BLOCKING 缺陷：0；剩余所需源码修改：无。

审查者为前两轮独立源码审查的同一 Codex agent，`gpt-6-astra` / reasoning effort `max`；保留前次上下文，属于补充轮。`review_independence: same-family`，`acceptance_status: provisional`。审查未执行 SSH、GPU、模型、正式队列或 CPU 距离报告，也未修改实现。真实六端 M0 结果尚未在本审查中验收，不能由本结论视作已通过。

本轮审查当前 `tools/queue_incremental_role_objective_full.py`（122行）和最终修订后的 `tools/report_incremental_role_objective.py`（80行），依据原登记计划的正式阶段，并追踪实际 queue、M0 acceptance、official receipt 和全 query scorer。

## BLOCKING

无。未发现当前正式训练入口、已退役 M0 衔接、保存/评价接收或配对统计方面的具体错误。

## NON-BLOCKING

最终源码无待修复项。审查中发现共享 compare 的旧通用说明会把所有比较写成“Different capacity”，不适用于新目标之间及相对 raw semantic 的同参数量比较。执行者已仅在新报告 `50-51` 覆盖该说明：raw_global_only 明确省略 semantic roles；其余明确同 semantic 参数集、训练目标不同，并保留 post-selection 与 identity bootstrap 不是训练种子稳定性的边界。最终补丁已重读，问题关闭；共享 compare 和 M0 所用源码未因该修正改变。

## 核对结果

1. **正式启动门与已退役 M0 兼容。** `require_m0:25-38` 要求旧 campaign 是六端 `M0_ONLY_COMPLETE`，每个 job exit0/COMPLETE，磁盘 acceptance 与 job 内接受结果一致，状态是 `REAL_M0_AND_ISOLATED_INCREMENT_ACTIVITY_ACCEPTED`。它核对 c、六个 Q/K 的已验收正活动以及保留文本产物的实际 SHA，并要求工程 probe 已不存在；没有再次调用要求旧 probe 必须存在的旧 verify_m0。原 M0 consumer 在写 acceptance 前已检查八次更新、BN8、reload、全参数与独立梯度，acceptance 的 artifact map 正确排除被退役 probe。因此原验收责任保持，而退役不会令正式入口必然失败。

2. **沿用 initializer JSON，不继承 M0 权重。** 正式命令调用 `m0.command(..., args.m0_campaign, output)`，将六份原 initialization JSON 作为绑定见证，输出另建正式目录。既有 entry/foundation 的 train 路径重新从公共 CLIP/新 camera/head/角色 seed42 构造模型，严格比较 binding 后创建优化器；它不读取 M0 probe。evaluate 读取的是本正式目录训练出的 best_map.pth。mode 不进入模型初始 binding，因此 prepare/m0/train/evaluate 的既有绑定一致。

3. **source-map 与调用链正确。** 正式队列在第一次 run_logged 前将同一个 `queue_foundation_recipe.base.source_map` 设为正式 source_map；report 在调用 `require_sources` 前也安装同一函数。复用的 run_logged 因而核对正式 manifest、完整 source map 和六份 initializer。当前旧 M0 source scope 为393项，不包含这两份新正式文件；独立 FULL_SOURCE_SCOPE 在审查读取时尚未生成，应作为后续部署准备，而非已核验的运行输入。

4. **控制、容量、资源与顺序符合登记。** 当前 INPUT_SEAL 确为 semantic3+global_only3，共45个保留产物条目；队列启动和报告读取时均逐条核验这些条目。容量表达式确实是 `5,192,548,352` 字节，即六份384MiB权重预算、600MiB距离等预算及原2GiB预留；复用 run_logged 每阶段仍要求2GiB。按201→MSVR→100、每集两目标顺序串行 train→evaluate，只用物理0/1，没有重跑旧控制、并行第二模型或放宽磁盘门。

5. **正式 receipt 和同一 best 的接收正确。** `accepted_row:41-67` 的字段与 foundation 实际 official_metrics/training producer 相符：完整1..50 epoch，schema、initializer、condition/objective 一致；以 `(mAP, epoch)` 取最高值，匹配训练中的相等时后轮覆盖规则。全部 mAP/R1/R5/R10 与同一选中 history 行一致，并核验该权重、正式距离、训练 best 距离以及 upstream metric parity；没有跨 epoch 拼接指标。新完整训练 batch-order 文件逐字节等于该集原 raw semantic 文件，两个新目标因此也彼此配对。

6. **正式完成后才启动一次 CPU 报告。** 六个 job 必须依次完成 train、第一次 strict evaluate 和 accepted_row，才会写 accepted_matrix。queue 先落盘 `COMPLETE/report_invocations=1`，然后启动 report；不存在让子报告在 COMPLETE 尚未落盘时读取的旧顺序问题。报告进程 CUDA_VISIBLE_DEVICES 为空、距离加载到 CPU，仅做保存数组统计，不再启动模型。子报告退出码独立记回 campaign，不能把训练完成与报告成功混同。

7. **恰有15个登记配对。** 每个数据集：ratio 相对 raw semantic/global 两对，repair_keep 相对这两个控制及 ratio 三对，共5×3=15对。当前控制行的 status 都为 `VERIFIED_COMPLETE`，variant 前缀改名与 compare 的行选择协议一致。compare 对每对核验官方 receipt/距离，重新计算与原度量一致的分数，并检查 query/gallery 身份、camera、scene数组与距离形状逐项一致。

8. **全 query 与身份统计没有筛选正收益。** 现行 camera_scores/scene_scores 遍历每个 query，仅删除同 ID 且同 camera/scene 的 gallery 项，保留全部异 ID 负例，要求每个 query 有合法正例。report 为完整 query_ids 顺序写 AP 与 first-match rank；compare 已先保证两端 metadata 顺序相同，因此使用最后加载端的 query_ids 不会错配。repair/new-error 计数、query AP正负变化和每身份 AP变化均由同一完整数组计算；identity bootstrap 的对象与字段名一致，未冒充训练种子结论。MSVR 全部使用 scene 过滤。

9. **轨迹、末轮与推进条件保留。** report 逐端验证 training_steps 与 batch-order 数量及每条 epoch/batch 对齐，并带出完整50行 history，包含E50及各轮全部 CMC；累计正式步数来自实际日志，而非硬填预估12968。推进条件严格为 mAP增量≥0.5且R1不降，并分别保留全部15对结果。`training_and_epoch_eval_seconds` 来自该训练回执的起止区间，标签没有将其冒充包括构造和最后 strict 评价的完整 CLI 墙钟时间。Markdown是短表，完整轨迹/逐query/逐身份信息在 SUMMARY.json。

## 尚未被本审查接受的事项

这仍是部署前源码判定。实际六端 M0 验收、正式阶段可用空间、最终 FULL_SOURCE_SCOPE 的内容及其远端一致性、六端fresh50、strict评价、15对完整CPU统计均须由真实产物给出证据。本轮不查询这些运行状态，也不把 M0 尚未送达的结果、SOURCE_ONLY PASS 或预留容量常量当成执行成功。

最小后续动作：保持已审查实现，完成正式 source scope，并在六端真实 M0、控制文件和容量实际通过后运行登记正式阶段。没有新增模型修补、额外兼容层或科学参数搜索建议。
