**赞成每步直接计算其余 13 项的辅助导数；这是针对本次已证实误差的最小实现修复。你提出的控制端与 CPU 核验边界也正确。** 这份意见认可修复方向，尚未审阅修订代码，更不能预先认定 R2 M0 会通过。

我已直接读取两份 05:39 原始观察 JSON。执行 `92a75e4` 的 T0 退出 0；fold0 control 保存了 3 个已完成更新，第 4 步在 `train_msvr_supported_gradient_balance.py:277` 的辅助梯度参考检查停止，M0 退出 1。失败发生在 `unscale_` 和第 4 次 AdamW step 之前。相对误差为 **0.006824872357540746 > 0.005**，参考范数 0.3899933414205923，重建范数 0.3900544174096418，差值范数 0.0026616547754863508。没有完整 M0、Q1 或 balanced 训练结果。顶层 M0 JSON 中残留的 `RUNNING` 不能覆盖 pipeline 的 `STOPPED_AT_M0` 与真实退出码。

证据直接证明的是：本次混合精度执行中，`grad(total) - grad(rank)` 未在登记误差限内还原 `grad(auxiliary)`。不同 backward 路径的舍入/累加与最后一次相减都可能贡献；现有标量证据不能把误差完全归因于最后一次 FP32 相减，也不能据此判定排名梯度本身有错。程序顺序表明旧全图总梯度检查以及首角色 current/history/full-rank 检查先于这次断言执行，不能把这些通过推广成全部角色的完整证据。

修订必须保留以下具体边界：

1. **直接 A 的构造。** 在普通总 loss backward 释放图之前，从原 `components` 与原权重构造其余 13 项的加权和，使用同一 scaler scale、同一当前前向图、同一 189 个 encoder 参数调用 `autograd.grad`。沿用当前容量参考中“复制 components、令 fused metric 项为图连接零”的构造即可；不要改成对 `loss - rank_loss` 的另一次取消运算。两端都多计算这一次辅助参数导数，14 个 neck/head 仍由原总目标 backward 产生。
2. **实际更新路径。** 只有 balanced 且有支持时使用 `wR * (R_current + R_history) + wA * A_direct`。control、预热、无支持步骤继续逐位保留原 `current_total + history`，不能因直接 A 已存在而替换这些步骤。EMA 仍只在有支持 AP 步更新；其辅助范数来源改为直接 A。优化器、BN/RNG、数据/预算、全部系数及科学门保持原定义。
3. **M0 参考不能自证。** 每步实际 A 与容量阶段的独立辅助参考必须来自独立求导调用，不能把同一列表或相同 tensor 别名同时传入 actual/reference 后宣布误差为零。current/history/full-rank 的独立参考继续保留；balanced 的 applied 参考使用独立 full-rank 与辅助参考的加权组合。control 的 applied 参考使用原 `full direct_loss` 导数，不能强制它等于分别求导的 R+A。现代码第 259 行提前删除 `direct`；若用于后面的逐角色 control 核验，必须保留到核验结束再释放。正常与参考求导仍在同一更新前参数状态进行。
4. **CPU 原恒等式必须同步改正。** `verify_msvr_supported_gradient_balance_stats.py:71-74` 目前无条件把实际梯度、原梯度与 R+A 关联。改用直接 A 后，只有 balanced 的有支持步骤可按实际加权 direct R/A 重算组合范数。`direct_sum_vs_original` 与 `subtraction_auxiliary_vs_direct` 应记录真实差异作为诊断；control/预热/无支持继续核对原加法路径。不能保留已失效等式使正确 control 再次停止，也不能删除所有实际组合检查只剩记录为 True 的字段。标量核验仍不等于独立恢复参数梯度。
5. **误差门和成本记录。** 真正独立梯度参考仍用相对 0.005、零参考绝对 1e-8；已被替换的减法 A 只保留误差诊断，不能重新解释原 FAIL。每步新增一次 auxiliary backward，两端相同；容量阶段四个独立分量参考调用及原总梯度参考的成本继续单列。计划第 58、72、82、95 行和相关日志/CPU计数需要如实反映该变化。

T0 的最小补充应覆盖本次真实问题：构造 direct A 与 total-minus-rank 有差异的固定小例，确认 candidate 使用 direct A、control 与无支持更新仍逐位保持原路径，且 CPU 不再错误强制 control = direct R+A。这是检查新路径和旧边界，不是证明真实 CUDA 数值误差已经消失。

保留 `92a75e4` 失败产物、3 次真实更新及第 4 步已付出的前向/反传成本。R2 应有明确的实现修订记录和新运行产物；不扫描权重、不放宽 0.005、不改变任何科学门。收到修订后，重点复核上述引用寿命、独立参考来源与 CPU 分支，再由真实 T0/M0 判断修复是否成立。

意见与小回执已保存到新的 R2 子目录，初次审查未覆盖：

- [final_response.md](<D:/Program Files/UserCache/gb/codex/tmp/trifusion_supported_balance_code_review_20260921/R2/final_response.md>)
- [report.json](<D:/Program Files/UserCache/gb/codex/tmp/trifusion_supported_balance_code_review_20260921/R2/report.json>)
- [receipt.json](<D:/Program Files/UserCache/gb/codex/tmp/trifusion_supported_balance_code_review_20260921/R2/receipt.json>)

本次仅阅读本地代码和失败观察文本；未运行张量、模型、优化器或远端命令，未修改仓库。R2 是原 fresh reviewer 的继续审查，same-family / provisional；不声称又启动了新的 fresh agent，后端独立 attestation 仍不可用。

<oai-mem-citation>
<citation_entries>
MEMORY.md:452-467|note=[preserve failed runs and distinguish engineering from scientific evidence]
</citation_entries>
<rollout_ids>
01a06f22-cc0d-7da2-8cae-68a700778767
</rollout_ids>
</oai-mem-citation>
