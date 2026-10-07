# 固定五 best 诊断源码审查 · 第二轮

结论：**PASS（SOURCE_ONLY），blocking_count=0**。同家族、独立初始上下文、provisional；请求路由 Astra/max，实际运行模型/effort **UNATTESTED**。本轮未 SSH、未导入神经模块、未构造模型、未做 NN/训练/部署，也未修改生产源码或 Git。

本轮代码 SHA256：`3d8ca39b2418a3379224768be927ea28a96cd2d0b5253b054b2defb09b4785c3`。计划仍为 `83c81e09061729e1fd623641b4faad62586bb305f2f55a5130aa17475cf09732`。

**首轮 FAIL 保留**：原代码 `e1903440e9de0df3d928600c9e02032e969bd118c98b5b44cf84598cd7e19e1e` 的 1 项阻断与原计划、MD/JSON 已按原字节验证并封存于 [SNAPSHOT.json](C:/Users/gb/.codex_tmp/independent_evidence_draft/fixed_five_source_review892/initial_fail/SNAPSHOT.json)；[首轮报告](C:/Users/gb/.codex_tmp/independent_evidence_draft/fixed_five_source_review892/initial_fail/FIXED_FIVE_DIAGNOSIS_SOURCE_REVIEW_20261007_892.md)仍为 FAIL，不追认为通过。私有 trace 同时保存两个代码版本、差异和两轮完整请求/响应。

## 本轮修正与验证

B1 已在修订源码中解决。原 `main()` 的全协议环境表填充循环被放到 `read_protocol` 之后；MSVR310 仍读取 scene，其他数据集仍读取 camera，保留原同名键一致性断言，随后才调用 `entry.configure()`。[修订 57—64 行](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_incremental_fixed_best.py:57)、[原 main 106—112 行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0386_run_incremental_role_objective.py:106)。

AST 比较确认：仅增加这段字典填充，并移动 `configure()`；剔除这两项明确变化后，全部 AST 与首轮完全相同。因此没有新增 fallback/try、兼容层、loss、模型、gain、容差、配对、权重选择或队列逻辑。填表没有随机数或模型操作，故不改变原初始化构造顺序/随机状态；checked build/schema/condition 的最终绑定保持首轮核查的路径。

对五个任务分别建立独立环境表，执行实际源码提取的 record parser、dataset getter、collate、`_eval_batch` 及增量 `training_batch` 的标签查找路径。输入来自三个完整真实协议；图像读取/变换和张量/CUDA 仅为普通 Python 占位数据：

| 任务 | query/gallery | 完整批次数 | 结果 |
|---|---:|---:|---|
| RGBNT201 md_batch_ratio | 836/836 | 28 | PASS |
| RGBNT201 repair_keep | 836/836 | 28 | PASS |
| MSVR310 md_batch_ratio | 591/1055 | 27 | PASS |
| MSVR310 repair_keep | 591/1055 | 27 | PASS |
| RGBNT100 md_batch_ratio | 1715/8575 | 161 | PASS |

共 **16,926 条记录、271 个批次**：全部文件名 tag、身份/camera/view 元数据、作者文件名解析出的真实身份和环境均匹配；MSVR310/100 的全部环境查找成功，201 继续使用原 `image_batch` 路径。环境表大小分别为 4,787、2,087、17,250 个键，无同名环境冲突。代码仍通过 `core.forward_features()` 绕开仅训练时需要环境标签的外层辅助头，但之前漏掉的评价批处理依赖现在也已满足。

主要路径证据：[records_for/loader 8—39 行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0025_official_three_dataset_data.py:8)、[201 getter 98—107 行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0129_bases.py:98)、[MSVR getter 91—94 行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0033_aligned_data.py:91)、[100 getter 107—110 行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0021_train_rgbnt100_signal_oof.py:107)、[collate 142—182 行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0131_make_dataloader.py:142)、[_eval_batch 60—63 行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0009_run_correspondence_roles.py:60)、[环境查找 35—39 行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0386_run_incremental_role_objective.py:35)。完整数据检查保存在 [R2_DATA_CHECKS.json](C:/Users/gb/.codex_tmp/independent_evidence_draft/fixed_five_source_review892/.aris/traces/experiment-bridge/2026-10-07_fixed_five892/R2_DATA_CHECKS.json)。

## 沿用并确认未改变的核查结论

- **原模型与 strict best：** 五份初始化 JSON 与正式 accepted rows 完整 binding 相等；checked entry SHA 仍是 `70121fe68b6a89830ca98835e84ace604b04986e125881eb571744dc7549c17d`。最终类为 IncrementalObjectiveHeads/DetachedSemanticTriFusion，原 objective、schema、condition、cfg、head、公开 CLIP、protocol、state 和 first6 cuda:1/last6+head cuda:0 分区保留。foundation.build 比较完整 witness；load 核对 schema/dataset/condition/protocol 并 strict=True 加载完整 state。[新 build/load 65—69 行](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_incremental_fixed_best.py:65)、[foundation 103—107 行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0243_run_foundation_recipe.py:103)、[foundation 180—186 行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0243_run_foundation_recipe.py:180)。
- **状态和 metadata：** 明确 eval、inference_mode、无 AMP/optimizer；四项有限 CPU FP32 1536D 特征来自每记录一次 core 前向。完整参数/buffer state_dict SHA 与无梯度检查包围提取；六个元数据数组与原正式/独立 global 距离均要相等。[新 70—81 行](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_incremental_fixed_best.py:70)、[提取 120—139 行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0314_diagnose_native_research_best.py:120)、[head eval 42—47 行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0267_evidence_author_heads.py:42)。
- **比较与阈值：** 原四指标全部跟随同一选定 best，fused/独立 global 对原回执均 `<1e-5` 点；h/f 恒等式仍为原 `1e-6` atol/rtol。完整 gallery、真实 scene/camera 过滤、逐 query AP/首位修复与新增错误、身份宏平均均保留。[新 83—112 行](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_incremental_fixed_best.py:83)、[原 score/paired 45—94 行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0314_diagnose_native_research_best.py:45)。
- **五任务与首错停止：** 五个固定 JOBS、Popen/wait 顺序执行、首个非零退出立即标记 FAILED 并返回，整个协调器 AST 未变；失败100 repair_keep 无作业/成绩。物理 GPU0/1、仅 index/memory.used 查询、原诊断 `>2 GiB` 门及正式 `5,192,548,352 B` 门不变。预计张量 `624,523,020 B`（约0.582 GiB，另加序列化/文本）。[新 122—153 行](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_incremental_fixed_best.py:122)。
- **输入封存：** 首轮已实核423件执行快照、45件正式文本及五份初始化/回执，完整 SHA 随同名 JSON 保留。旧420源＋新代码/计划/本MD/JSON＝424；独立seal排除自身。输入SHA前后检查、原控制器EXIT0检查、旧权重/回执保存逻辑均未变。最终具体诊断seal尚未作为本轮输入；其完整成员与真实目标机检查仍是启动前置，不是本审查已完成的工作。[新32—40行](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_incremental_fixed_best.py:32)、[新116—117行](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_incremental_fixed_best.py:116)。

## 非阻断说明与边界

1. 作者 MSVR evaluator 仍会往 cwd 的 `re.txt` 追加日志；子进程 cwd 为 ROOT，两个 MSVR 作业四组评分共写4,728个query block到 ROOT/re.txt，位于各作业目录之外。没有所供证据表明它是受保护输入；这是继承的明确副作用。[作者71—78行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0213_metrics.py:71)、[cwd140行](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_incremental_fixed_best.py:140)。
2. 本地发布目录不是完整可运行目标树（107旧scope路径缺失、27文件仅CRLF/LF差异）；源码权威是已验SHA的实际执行快照。不要以本地整树等值替代服务器424源及输入封存验证。host、conda、GPU/OWN NN空闲、当前磁盘、唯一observer和二进制物理SHA并未在本轮查询。
3. 第一份R2数据fixture因注入Windows Path导致原Linux `split('/')` getter取名不符而退出1；原fixture和错误原样留存。仅将fixture的目标路径语义改为标准库PurePosixPath后，完整检查通过；没有为此修改生产源码，也不计作模型运行失败。
4. 本结论不是真实模型初始化、神经前向或科学有效性证据。同模型g仍是组件诊断，c仅在单独检索时归一化，所有配对仍是已选官方单种子的描述。历史FAIL/STOP和缺失100 repair_keep继续保留；没有新训练、因果梯度归因、三集稳定收益、创新或SOTA结论。

时间：2026-10-07T14:44:46.731928+08:00。本轮完整请求/响应及原失败保存在 `C:\Users\gb\.codex_tmp\independent_evidence_draft\fixed_five_source_review892\.aris\traces\experiment-bridge\2026-10-07_fixed_five892`。所有精确输入SHA、原五checkpoint/距离/receipt/initializer、独立global输入与45/48保护输入详见同名JSON。
