审计已完成，终态为 **WARN / CLOSED_WITH_LIMITS**：M0 工程证据与可独立复算的确定性检查通过，**没有剩余 M0 工程或审计阻断**。该结论不授予检索效果或晋级结论。

| 检查 | 判定 | 结论 |
|---|---|---|
| A：真值来源 | PASS | 使用真实 TRAIN 身份/场景标签；独立复核三折身份隔离和完整来源队列。 |
| B：分数归一化 | PASS | 梯度比、参考误差和过拟合损失比均属工程指标，没有冒充检索成绩。 |
| C：结果存在与数值 | PASS | 全部八个运行、248 更新及原始文件齐全；严格重载措辞已修正并复读。 |
| D：实际执行路径 | PASS | 直接辅助求导、完整历史排名梯度、两端实际组合与 AdamW 路径一致，保存标量通过独立复算。 |
| E：覆盖范围 | WARN | 单数据集、seed42；真实 M0 无支持更新为零，逐步参数梯度未独立重生成。 |
| F：证据类型 | PASS | 已区分真实标签训练、T0 合成检查及工程一致性参考。 |

独立远端 CPU 核验耗时 **26.965 秒**，使用 2 个 CPU 线程和 1 个 interop 线程，CUDA 未初始化。覆盖全部 **248 步、4,945,920 个距离元素、8,280 项范数/余弦恒等式、780 个来源 batch**，并核对 **37 个 M0 文件、六个容量端检查点和三个 B0 检查点**。全部 90 项已保存参数参考检查满足原门槛，最大相对误差 **9.433501552915618e-05 < 0.005**；这些是原运行见证及其标量核验，不是重新生成梯度向量。

必须保留的边界是：实际 M0 为 **16 个预热步、232 个有支持步、0 个活动无支持步**。过拟合中另有 **557 个“有支持但排名梯度范数为零”的角色步**，不能计成无支持覆盖。严格重载仅覆盖六个容量端，两个过拟合运行没有终点检查点或重载。R1 的 **3 次已完成更新、失败的第 4 步，以及 0.006824872 > 0.005 的原失败**均保留。

归档还保留了审计脚本失败和修正。远端原始 `m0_cpu.json` 已重新按字节核验为 **16,016 字节、SHA-256 `910c5c8493059cded0adb02bcbdb5ab0b701b40567b645f3a1ca37014feb21b1`**；本地 `remote_text` 副本多出 343 个 CR 字节，JSON 内容相同。所有 154 份本地文本转换均已逐一核对，错误草稿和两种哈希分别归档。实验数据及所有门槛未变。

完整交付位于 [审计目录](<D:/Program Files/UserCache/gb/codex/tmp/trifusion_supported_balance_m0_independent_audit_20260921>)，其中包含 [A–F 报告](<D:/Program Files/UserCache/gb/codex/tmp/trifusion_supported_balance_m0_independent_audit_20260921/EXPERIMENT_AUDIT.md>)、[机器可读判定](<D:/Program Files/UserCache/gb/codex/tmp/trifusion_supported_balance_m0_independent_audit_20260921/EXPERIMENT_AUDIT.json>)、[全部文件清单](<D:/Program Files/UserCache/gb/codex/tmp/trifusion_supported_balance_m0_independent_audit_20260921/artifact_manifest.json>)及[哈希与执行回执](<D:/Program Files/UserCache/gb/codex/tmp/trifusion_supported_balance_m0_independent_audit_20260921/audit_receipt.json>)。脚本、精确请求、原始输出、失败记录、原文/修订快照、详细说明和本回复原文均已保留；报告 MD、JSON 与最终回复的 SHA-256 见回执。

请求的审阅路由为 **gpt-6-astra / max / fresh-none**，归属 **same-family / provisional**，后端独立认证不可用。审计未编辑仓库、未启动模型前向或更新、未下载模型/张量/数组，也未读取或干预正在运行的 Q1 指标。

<oai-mem-citation>
<citation_entries>
MEMORY.md:454-467|note=[preserve source-only audit scope and failed evidence]
</citation_entries>
<rollout_ids>
01a06f22-cc0d-7da2-8cae-68a700778767
</rollout_ids>
</oai-mem-citation>
