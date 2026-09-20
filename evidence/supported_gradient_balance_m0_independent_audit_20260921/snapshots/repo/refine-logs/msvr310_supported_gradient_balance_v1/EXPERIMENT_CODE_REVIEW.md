**最终裁决：PASS_WITH_LIMITS。R2 可以进入原登记的 T0/M0；没有剩余 BLOCKING 缺陷。** 这只是代码审查通过，R2 的实际张量测试、CUDA 数值误差及完整 M0 仍未验证。

裁决对象是 `implementation_revision=r2_direct_auxiliary`，最终配置 SHA-256 为 `9ce36299299efbc09ccfb72f5b1c4fed20fc1026bd9fee1bd55a591d6950ec7b`。本次检查时仓库 HEAD 仍是 R1 的 `92a75e4ac46cccb79bb6f78f913998c0d0039ce8`；这份裁决绑定读取到的 R2 文件与配置，不把旧 HEAD 当作 R2 部署证明。

直接读取修订源码后，确认以下关键点已正确处理：

- **实际 A 与独立参考分开计算。** [训练器第 175–181 行的起点](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/train_msvr_supported_gradient_balance.py:175) 使用原 components、原权重和同一 scaler scale，独立计算当前排名及其余 13 项的辅助导数。第 203–208 行在首历史组另行构造辅助目标并再次求导作为 M0 参考，没有复用实际 A 的列表或 tensor 作为参考。两者都在总 backward 释放图之前完成；仍只针对 189 个 encoder 张量，14 个 neck/head 梯度保持原总 backward。
- **control 与 candidate 的实际路径、参考路径分清。** [组合函数第 70 行起](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/msvr_supported_gradient_balance.py:70) 先形成完整 `R_current + R_history`。只有 balanced 且有支持时使用完整 R 与直接 A 的加权组合；control、预热、无支持步骤仍逐位使用 `current_total + history`。训练器第 260–288 行保留原完整图总梯度参考至逐角色核验结束；control 使用该参考，candidate 使用独立 full-rank/auxiliary 参考的加权组合，之后才删除 `direct`。没有提前释放或误用 scaled/unscaled 参考的问题。
- **真实 reference 门未放宽。** `check_reference` 继续使用相对误差 0.005，零参考使用绝对误差 1e-8；原全图/VJP、head 保持、RNG/buffer、单次 unscale/AdamW、203/203、零 overflow、严格重载及原 overfit 门均保留。原已失败的减法 A 被替换为直接 A，减法偏差通过 `subtraction_auxiliary_vs_direct` 留作真实诊断，没有把 R1 的 FAIL 改成 PASS。
- **CPU 不再强制 control 等于 direct R+A。** [统计验证器第 69 行起](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/verify_msvr_supported_gradient_balance_stats.py:69) 核对 direct-sum 自身范数与原梯度差异，只有 balanced 有支持步才按实际权重重算组合范数；其余步骤要求 `original_sum_vs_applied.difference_norm == 0`。新增 `current_auxiliary_backward_calls` 按每步 1 次、全程实际步数核对，容量阶段独立参考调用继续单列。
- **本轮 T0 缺口已补齐。** R2 初稿的差异 fixture 仍只覆盖 candidate；我指出后，执行者补充了 [数学检查第 102 行起](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/check_msvr_supported_gradient_balance_math.py:102)。现在同一个 `direct A != total-minus-rank` 固定小例实际调用有支持 control、有支持 balanced、无支持 balanced 三条组合路径；control/无支持断言原梯度逐位保持，并把三类合成行送入真实 `verify_balance`。无支持例明确令 R/H 为零，A 仍与 current 不同，EMA 保持 None；原有非空 EMA 不变例也保留。多张量角色范数、置换、AMP 缩放和头部检查仍在。这个测试覆盖修正已完成源码审阅，尚未运行张量。

固定计划已同步直接辅助求导、两端相同额外成本、独立参考、失效减法恒等式的诊断地位和 R1 失败记录。跨场景及更早前置源码保持绑定不变；M0 仍 248 更新，正式 Q1 仍 1560 更新、固定 seed42/epoch20、完整图库和 paired/Signal 两组原科学门。源码没有新增官方读取、选模、扫描或重启失败状态的路径。

需要保留一个解释边界：有限精度下，分别反传的 direct R+A 与原 combined backward 可以不同，即使系数为 1 也不保证逐位相等。计划末段已明确这一点。未来结果应归属于这个固定的 R2 实现与原控制端的比较，不能声称实际数值更新逐位只差两个乘数；这不构成当前实现缺陷，也不支持任何现阶段性能结论。

本次实际完成的检查为：7 个新文件 AST 解析、全部 5 个 `combine` 调用与新参数签名核对、当前配置 9 项及前置六份配置 58 项项目文件哈希核对、R1 封存的 8 份本地文本对既有 inventory 的长度和哈希核对，全部通过。另用 stdlib 标量模拟核对固定 fixture 的 float32 加乘误差落在现有 CPU 范数恒等式容差内；这不是实际 torch/T0 执行。未重新连接远端核对 inventory 中的远端声明。

R1 的真实 M0 失败和 3 次更新保持封存。R2 从原固定初始化进入新 T0/M0；Q1 继续受同配置 M0 与完整 CPU 凭证约束。实际 M0 仍可能揭示尚未观察的数值或容量问题，不能由静态审查预先排除。

裁决与回执已另存，未覆盖初次审查或 R2 救援意见：

- [final_response.md](<D:/Program Files/UserCache/gb/codex/tmp/trifusion_supported_balance_code_review_20260921/R2_CODE_REVIEW/final_response.md>)
- [report.json](<D:/Program Files/UserCache/gb/codex/tmp/trifusion_supported_balance_code_review_20260921/R2_CODE_REVIEW/report.json>)
- [receipt.json](<D:/Program Files/UserCache/gb/codex/tmp/trifusion_supported_balance_code_review_20260921/R2_CODE_REVIEW/receipt.json>)
- [static_checks.json](<D:/Program Files/UserCache/gb/codex/tmp/trifusion_supported_balance_code_review_20260921/R2_CODE_REVIEW/static_checks.json>)

审阅者只写入指定审查目录，没有修改仓库，也没有导入 torch、运行张量、模型、优化器或远端命令。R2 是原 fresh reviewer 的继续审查，same-family / provisional；请求路由仍记录为 gpt-6-astra/max，后端独立 attestation 不可用。

<oai-mem-citation>
<citation_entries>
MEMORY.md:452-467|note=[preserve failed evidence and bound engineering review claims]
</citation_entries>
<rollout_ids>
01a06f22-cc0d-7da2-8cae-68a700778767
</rollout_ids>
</oai-mem-citation>
