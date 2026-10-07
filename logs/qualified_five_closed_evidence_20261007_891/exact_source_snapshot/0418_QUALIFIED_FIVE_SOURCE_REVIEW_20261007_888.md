# 五端正式入口独立源码复审

结论：**PASS**，blocking **0**。审查时间：2026-10-07T09:38:22.802829+08:00。

审查者 `/root/review_qualified_five888`；`review_independence: same-family`，`acceptance_status: provisional`。实际模型后端与推理档位均为 **UNATTESTED**。这是源码结论，不是新正式训练或远端运行验收。

## 范围与证据

只读审查本仓库五端 queue/report、§888 计划、原六端 queue/report、checked M0/entry，以及必需的初始化、schema、训练、评价和统计链。使用本地标准库读取和 AST；没有 SSH、项目 import、NN、训练、队列或统计报告执行。仅写本审查 md/json。

原始证据为 `C:/Users/gb/.codex_tmp/independent_evidence_draft/checked_m0_terminal_failure887/REMOTE.json`，SHA256 `08addd190a6cd87f2955dab2eb9db815fe0945d784605b75d1f938211fcf4e16`。其 51 份嵌入文本逐份 SHA 通过。原 controller 于 `2026-10-07T09:24:19.637023+08:00` EXIT1；campaign 中 RUNNING 是遗留状态。

六端生产 M0 均完成 8 更新，281/281（RGBNT201）或 285/285（MSVR310、RGBNT100）trainable 参数活动，全部作者 BN 为 8，strict reload 差值均 0。五份 acceptance 与原 campaign、初始化 binding、training receipt、累计梯度和其 artifact SHA 一致。RGBNT100 repair_keep 没有 acceptance；第三 query/key 八步累计均 0，unused 均 false，c 与其余四个 Q/K 非零。该失败保持原义。

## 核对结果

| 检查 | 结论及位置 |
|---|---|
| 恰好五份资格 | PASS。`tools/queue_incremental_qualified_five.py:28` 要求原 EXIT1、六份原 job、准确五个被选 pair、五份 COMPLETE/exit0 acceptance、六个 Q/K 与 c 正值、原件 SHA 和 probe 已退役；缺失端无 acceptance 且第三 Q/K 累计 0。没有改写旧六端 campaign。 |
| checked binding 与新鲜初始化 | PASS。新 queue 第89行调用 `checked.configure()`，正式 train/evaluate 指到同一 checked entry，保留 initializer 的 entry SHA 和 VJP policy；完整 binding/schema 沿 configure 链传入 foundation。`run_foundation_recipe.py:103` 重新构造并精确对照 witness；`run_clean_clip_joint.py:37` 从公共 CLIP 构造 fresh camera/head，没有 M0 权重继承。 |
| 正式 loss 原样 | PASS。`run_incremental_role_objective_checked.py:22` 对非 M0 直接返回原 incremental loss。模型、两公式、margin/权重、raw 作者任务、优化器、AMP、seed42、50轮及1536部署没有改变。 |
| 父/子进程 source_map | PASS。新 queue 第103行与 report 第21行均设置 `base.source_map` 为五端 map；每个 run_logged 及 report 新进程调用 `require_sources`，校验 manifest source 和五份 initializer。新 scope 按约定在审查后冻结，当前缺失不判缺陷。 |
| 单一 best、首次严格评价 | PASS。新 queue 第110行每端只按 train→evaluate 执行；继承原单一 mAP-best 与相同 mAP 选较晚 epoch 规则，全部 CMC 来自同一权重。完整50轮、condition/schema/binding、seed、protocol、best/official距离 SHA、upstream equality 和原容差均保留。新旧 `accepted_row` AST 完全相同。 |
| 完整 batch 与曲线 | PASS。新 queue 第78行以原 raw semantic 全部训练 batch-order 字节对照；report 第35行保留 steps/batches/各轮更新总和与 epoch/batch 顺序校验，保留全 history/E50 和实际训练加逐轮评价耗时。 |
| 12 配对与明确缺失 | PASS。五端各对 raw semantic/global，201/MSVR repair_keep 另对同集 MD：共12。新旧 report 的完整逐行统计/配对循环 AST 相同。SUMMARY 第70行和 REPORT.md 文本第78行均明确 RGBNT100 repair_keep 缺失，没有补零或替代权重。 |
| GT 与原统计公式 | PASS。`analyze_correspondence_distances.py:28` 复核原件与指标、query/gallery身份和环境数组；以数据集 GT 算 ranking/AP，MSVR310 用 scene、其余用 camera 过滤。修复/新增错误、身份AP、固定模型身份bootstrap及 mAP≥0.5/R1不降门槛不变。 |
| 资源与失败边界 | PASS。新 queue 第18/85/98行保留 **5,192,548,352 B** 初始存储门、GPU0/1、max1；原 run_logged 的2 GiB逐阶段余量不变。未添加 retry、最后一端 rescue、清理或历史PASS重写。 |

合同数量为五端各50轮，**250轮**；沿匹配原 batch 流预期 **9,839更新**（2×2649 + 2×706 + 3129）。这些是待运行合同，不是本次观察到的新正式结果。

## 静态验证与一次补充复审

- 22 个相关 Python 源码 AST 解析通过。
- 20 个直接审查的受保护链源码 SHA 与 frozen415 一致；原 M0 manifest 的415项 map 与本地冻结表完全相同，INPUT_SEAL 与原 manifest 一致。
- 五端验收函数及完整统计循环 AST 分别与原六端版本相同。
- 初看 REPORT.md 只显示五行，虽 SUMMARY 已有 missing 字段，文字报告缺失原因未直接展示；已告知主代理。主代理补入明确缺失文本后，本次一次补充复审通过，统计循环未改，无遗留项。

## 范围限制

本审查未复跑远端415项、权重/数据或磁盘资格；不把本地静态通过当作部署/正式训练已完成。广域本地副本有已知27个历史CRLF字节差异与107个仅远端 comparator/protocol 文件，本次未修改或规范化；直接审查的20个保护链源码均与冻结摘要一致。父代理报告的远端全范围复核不冒称本审查独立重做。

`QUALIFIED_FIVE_SOURCE_SCOPE.json` 按本次合同在源码复审通过后冻结。运行时仍须通过既有 source/initializer/控制原件和存储断言。M0 活动通过不等于检索有效性；不主张三集完整 repair_keep、训练多种子、显著性、新颖公式、完整MDReID复现或SOTA。Broad goal 仍 ACTIVE_UNMET。

## 本次新增文件版本

- `tools/queue_incremental_qualified_five.py`: `ef14c868e0b15ae0e203ab3fd53bf03be66bb6fec56446edb7d03a0000ccea26`
- `tools/report_incremental_qualified_five.py`: `1c2c3259f3bf2a1dbc4650d2858b2611799213a32dd09bf805445f4eb08e1969`
- `QUALIFIED_FIVE_PLAN_20261007_888.md`: `ff0c66dd958fecfcc41e2a5fcdeb9d274edbd71f0a47746cd5136cd8e67b2d8f`
