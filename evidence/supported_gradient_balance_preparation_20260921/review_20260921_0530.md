审查结论：**PASS_WITH_LIMITS，静态实现审查通过；same-family / provisional。** 当前未发现需要阻止已登记 T0/M0 的 BLOCKING 缺陷。初版发现的 1 项 NON-BLOCKING 测试覆盖缺口，已由执行者补充；我重新读取补丁和配置后确认覆盖路径已经补齐，新增张量测试尚未运行。

最终审阅配置为 `configs/MSVR310/TriFusion-supported-gradient-balance-paired-v1.json`，SHA-256：`0b5ff0107e8dda6634e0c060a4b1a87e8e7e2fa117c7beb0cf5792b6ebedc15b`。审阅对象包括未提交的新文件；仓库 HEAD 为 `59db4bf760b524e1fbc9b2695d0c597287d3ccce`，因此不能仅用这个旧 HEAD 表示新实现已发布。

具体发现及修正：

- **BLOCKING：无。** 没有发现可由已读代码直接确定的错误梯度链、原门槛丢失、来源身份泄漏或必然失败的断言。这不是远端运行成功的证明。
- **NON-BLOCKING N1，已在源码层面修正：T0 原先未真正验证角色块置换和多张量整块范数。** 初版数学检查第 73 行仅反转名称并检查 `role_indices`，主要求导例子每角色只有一个张量，未覆盖计划要求的实际组合等变性。修订后的 [数学检查第 74 行起](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/check_msvr_supported_gradient_balance_math.py:74) 增加每角色两个张量，按 `[4,5,0,1,2,3]` 置换后实际调用 `combine`，比较全部参数梯度、角色统计与 EMA 状态，并用双精度平方和重算角色范数。补丁未改训练规则；测试文件 SHA 从 `3631b61f…` 变为 `b7ef9ee3…`，配置相应更新。状态为 **REVIEWED_NOT_EXECUTED**，没有把补丁存在写成测试通过。

以下关键路径按实际代码核对正确：

- **排名、辅助和历史梯度分解。** [训练器第 174 行](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/train_msvr_supported_gradient_balance.py:174) 计算当前实际排名项的 AMP-scaled encoder 梯度；普通总损失随后 backward；辅助量由当前总梯度减当前排名梯度得到。历史叶子导数使用同一个 cross-scene AP，并重放原角色入口 RNG，通过完整历史 VJP 加入排名。两端都执行这些测量。189 个 encoder 张量按三种前缀完整且互斥分组，14 个 neck/head 张量保存并逐位核对原当前总梯度。
- **组合及 AMP 语义。** [控制器第 65 行起](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/msvr_supported_gradient_balance.py:65) 先形成 `R_current + R_history`，在除以实际 scaler scale 后计算整角色范数。只有 balanced 的有支持步骤应用固定 EMA 和有界系数；control、预热及无支持步骤保留 `current_total + historical_rank` 原加法路径。无支持步骤不更新 EMA，并检查完整排名梯度为零。每步仍只有一次 `unscale_` 和 AdamW 更新；参数更新日志保存更新前/后范数及实际差值范数，没有把它解释为排名贡献份额。
- **M0 直接参考和原工程门保留。** [训练器第 178 行起](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/train_msvr_supported_gradient_balance.py:178) 在每个容量端首历史组计算独立 current/history/full-rank/auxiliary 参考；第 273 行起逐角色核对分解和最终组合，非零参考相对误差阈值仍为 0.005，零参考使用绝对误差 1e-8。原完整图总梯度检查、203/203 累计非零梯度、零 overflow、冻结路径、严格重载全部输出、三折两端 8 步和两端 100 步 overfit、校正损失比不高于 0.1 均保留，共 248 更新。来源批次文本重算显示三折第 3 步进入有支持的 AP，第 4 步首历史组均有支持，eligible anchors 分别为 16/24/32；这只证明分支在登记序列中可达。
- **数据、优化预算和科学门保留。** 初始化继续按折读取 source-only Signal 并以 seed42 新建角色；B64/K8、512 唯一历史记录、最大年龄 8、正式预热 65、20 epoch/260 更新、原 AdamW/LR/衰减均未改变。M0 只取来源记录，Q1 需要同配置 M0 与 CPU 凭证。完整 heldout gallery、真实身份/scene 过滤、五个输出、固定终点以及 paired 与 Signal 两组各五门由既有函数保留；[验证器第 294 行起](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/verify_msvr_supported_gradient_balance.py:294) 重新计算两组门及 seed42 的 10,000 次身份 bootstrap。未发现官方 query/gallery 读取路径。
- **T0 和 CPU 证据边界正确。** [T0 入口第 26 行](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/check_msvr_supported_gradient_balance.py:26) 实际调用新增数学检查，检查内部实际调用训练用 `scalar_gradients` 和 `combine`。CPU 验证器重算全部保存的 AP、掩码、支持、EMA、权重及标量组合恒等式，并检查 M0 直接参考账本。它没有独立恢复每步参数梯度，相关限制在 [统计验证器第 100 行](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/verify_msvr_supported_gradient_balance_stats.py:100) 和最终 scope 中明确保留。

资源与进程行为也已检查：[队列第 22 行起](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_msvr_supported_gradient_balance.py:22) 在启动前检查至少 4 GiB 空间和空闲 GPU，按 T0 → M0 → M0 CPU → Q1 → Q1 CPU 顺序运行，记录 PID、退出码和耗时，任何子进程非零退出都停止推进。`maximum_expected_additional_disk_bytes` 与预计耗时是预算声明；现有 runner 没有运行中的磁盘增长上限或超时终止。没有远端测量证据可据此声称实际新增空间不超过 3 GiB、显存够用或耗时符合预估；此项作为执行观察边界保留，不要求新增调度机制。

本次实际完成了 7 个新 Python 文件的 AST 解析、最终配置 9 项及前置六份配置 58 项本地 `project_file_sha256` 绑定核对、来源批次元数据文本重算和上述源码/差异审查，均通过。未导入 torch，未运行张量检查、模型、训练、优化器、GPU、远端命令或检索。远端权重、外部源码和运行环境仍应由已有 T0/M0 合同验证；本审查不提供性能、新颖性或科学晋级结论。

请求的审阅路由记录为 `gpt-6-astra` / `max` / `fork_turns: none`；独立性为 fresh context、same-family，接受状态为 provisional。后端型号/推理档位独立证明不可用，未将请求配置当作服务端 attestation。

审查原文、结构化报告和小回执已保存：

- [final_response.md](<D:/Program Files/UserCache/gb/codex/tmp/trifusion_supported_balance_code_review_20260921/final_response.md>)
- [report.json](<D:/Program Files/UserCache/gb/codex/tmp/trifusion_supported_balance_code_review_20260921/report.json>)
- [receipt.json](<D:/Program Files/UserCache/gb/codex/tmp/trifusion_supported_balance_code_review_20260921/receipt.json>)

我只写入上述 D: 审查目录。原缺口的源码修正来自执行者；审阅者未修改仓库。历史记忆仅用于定位项目范围与禁止把诊断升格为性能结论，当前判断依据本次直接读取的文件。

<oai-mem-citation>
<citation_entries>
MEMORY.md:452-467|note=[project scope and diagnostic evidence boundaries]
</citation_entries>
<rollout_ids>
01a06f22-cc0d-7da2-8cae-68a700778767
</rollout_ids>
</oai-mem-citation>
