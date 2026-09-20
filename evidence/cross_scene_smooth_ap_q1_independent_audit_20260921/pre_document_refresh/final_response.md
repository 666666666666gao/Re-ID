完整 Q1 独立终态审计已完成。**完整性结论为 WARN / CLOSED_WITH_LIMITS，确定性核验通过；科学资格为 FAIL，原门槛保持不变。**

报告、独立脚本、原始输出、失败尝试和哈希清单全部保存在：

`D:\Program Files\UserCache\gb\codex\tmp\trifusion_cross_scene_q1_independent_audit_20260921`

- [完整 A–F 审计报告](<D:/Program Files/UserCache/gb/codex/tmp/trifusion_cross_scene_q1_independent_audit_20260921/EXPERIMENT_AUDIT_Q1.md>)
- [机器可读结论](<D:/Program Files/UserCache/gb/codex/tmp/trifusion_cross_scene_q1_independent_audit_20260921/EXPERIMENT_AUDIT_Q1.json>)
- [独立证据汇总](<D:/Program Files/UserCache/gb/codex/tmp/trifusion_cross_scene_q1_independent_audit_20260921/AUDIT_EVIDENCE.json>)
- [全部输入哈希](<D:/Program Files/UserCache/gb/codex/tmp/trifusion_cross_scene_q1_independent_audit_20260921/audited_input_hashes.json>)
- [审计失败尝试记录](<D:/Program Files/UserCache/gb/codex/tmp/trifusion_cross_scene_q1_independent_audit_20260921/AUDIT_ATTEMPTS.json>)

独立重算覆盖了全部三折、两个端点、五个输出：**1,560 个训练步骤、116,501,504 个训练距离元素、30 组检索输出、2,069,520 个检索距离及排名元素、6,000 条查询输出和全部身份指标**。六个终态 checkpoint 的完整状态、冻结 Signal、初始化绑定、全图库干扰身份和 MSVR310 scene 过滤均通过核验。119 个原运行文件在审计首尾完全一致，448 个输入哈希已记录。

| 输出 | control mAP | cross_scene mAP | 配对变化 pp |
|---|---:|---:|---:|
| baseline_only | 53.129381 | 53.129381 | 0.000000 |
| fused | 52.838309 | 53.405492 | +0.567183 |
| CNN | 50.608598 | 50.343302 | −0.265296 |
| Transformer | 51.555161 | 51.417266 | −0.137895 |
| Mamba | 51.403658 | 51.997190 | +0.593532 |

配对 fused 增益为 **+0.5671825364 pp**，三折增益均为正，但身份 bootstrap 下界为 **−0.0772393104 pp**。原配对门仅通过 **2/5**；候选对 Signal 的门仅通过 **1/5**。候选 fused 的 mAP 比 Signal 高 **0.276111 pp**，但 Rank-1 为 **62%**，低于 Signal 的 **63%**，因此没有晋级依据。

保留 WARN 的具体原因是：

- **梯度证据有明确上限。** 所有保存的范数、分解和运行断言一致，但未保存完整逐步参数梯度与原始前向状态，无法独立重建全程反传。Q1 没有执行 direct 全图比较；M0 的六个单历史组见证不能扩大为 Q1 全程证明。203/203 是累计非零覆盖。
- **配对预热并非逐位相同。** 尽管初始化、采样记录和输入像素哈希一致，前 65 步共同 hard 目标已有浮点差异，各折 total 最大差为 0.001278043 / 0.001622677 / 0.002021313，不能声称严格相同的训练轨迹。
- **科学范围有限。** 只有 seed42，且内部 OOF 协议已参与此前开发；固定身份 bootstrap 不代表多训练种子或未使用的外部验证。
- **状态文档落后。** 审计所见 tracker 与 AGENTS 仍写 Q1 RUNNING；原始 pipeline 已是 COMPLETE_VERIFIED_Q1_FAIL，应在归档审计后更新可变状态文档，保持冻结合同不变。

候选端实际出现的四个预热后零 eligible 批次全部保留，排名项和历史上游为零，其余监督与更新继续。活动 total loss 的定义在两端不同，不能直接用绝对值判断优劣，也不能由 current/history 梯度比推出因果权重。

执行方 v2 报告、全部五个 CSV 和 24 组来源阶段统计已逐项复核，57,813 个数值比较的最大差为 **4.44e-14**。执行方失败的 v1 报告脚本，以及审计辅助脚本的失败与修正均已保留。没有训练、模型前向、优化器更新、官方测试访问、安装环境或更改原证据。

本审计没有发现阻止归档的确定性完整性缺陷。**应封存本配置为 Q1_FAIL；审计不授权调参重训、官方测试或消融。** 审阅路由为 gpt-6-astra / max / fresh，同族 provisional，backend 身份没有独立证明。

<oai-mem-citation>
<citation_entries>
MEMORY.md:454-462|note=[prior protocol boundaries and complete fold reporting; current facts independently verified]
</citation_entries>
<rollout_ids>
01a06f22-cc0d-7da2-8cae-68a700778767
</rollout_ids>
</oai-mem-citation>
