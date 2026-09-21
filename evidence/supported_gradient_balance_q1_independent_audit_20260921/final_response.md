完整 Q1 独立审计已结束，结论是 **WARN / CLOSED_WITH_LIMITS**：可独立复算的完整证据检查通过，科学结论仍为 **Q1_FAIL**，没有剩余工程或审计计算阻断。复核没有改变门槛，也没有启动训练、模型前向、优化器更新或官方测试。

报告和全部脚本、失败记录、复算表格位于 [独立审计目录](<D:/Program Files/UserCache/gb/codex/tmp/trifusion_supported_balance_q1_independent_audit_20260921>)。主要交付物是 [EXPERIMENT_AUDIT.md](<D:/Program Files/UserCache/gb/codex/tmp/trifusion_supported_balance_q1_independent_audit_20260921/EXPERIMENT_AUDIT.md>)、[EXPERIMENT_AUDIT.json](<D:/Program Files/UserCache/gb/codex/tmp/trifusion_supported_balance_q1_independent_audit_20260921/EXPERIMENT_AUDIT.json>) 和 [failed_checks.json](<D:/Program Files/UserCache/gb/codex/tmp/trifusion_supported_balance_q1_independent_audit_20260921/failed_checks.json>)。

独立复核覆盖全部三折、两个端点、五个输出、600 个合法查询、60 个查询身份；95 个单场景干扰身份和 432 条仅图库记录全部保留。全部 1,560 步、4,680 个角色步骤、120 个训练 epoch、116,501,504 个训练距离元素和 2,069,520 个检索距离/排名位置通过检查。远端 CPU 还重建了六个最终模型状态及冻结子集，核对了三个原始 B0 检查点和 1,950 个 B0 来源训练步骤；耗时 26.16 秒，CUDA 未初始化，模型和数组没有下载。

| 指标 | 独立复算结果 |
|---|---:|
| Control fused mAP | 53.399383646 |
| Balanced fused mAP | 53.452649370 |
| 配对 fused 增益 | +0.053265725 pp |
| 三折 fused 增益 | +0.129966707 / −0.107213849 / +0.146774279 pp |
| 配对身份 bootstrap 下界 | −0.126141312 pp |
| 配对门 / 相对 Signal 门 | 1/5 / 1/5 |

原始流水线仍是 `STOPPED_AT_Q1_CPU`，最终完成来自单独保留的 `q1_cpu_arithmetic_recheck/verification.json` 和退出码为 0 的执行凭据。远端实际 Python 3.10.14 独立复现了六处 sqrt 与幂运算的 1 ULP 差异，以及 fold 0、balanced、step 71、Mamba 的原加权范数失败。按实际 FP32 系数表示重算后，全部 1,743 个有支持的 balanced 角色组合通过原有阈值；没有放宽任何工程或科学门。全部 43 个接收文件、159 条远端绑定和 98 个源码路径的两份执行 Git blob 核验通过。

需要保留的限制包括：

- Q1 没有独立直接梯度参考检查；保存的范数、余弦、RNG、buffer、head 和重编码断言不能恢复全部参数梯度或 AdamW 任务贡献。203/203 是每次运行的累计非零覆盖，不代表每个步骤全部张量非零。
- 两端初始化、来源记录及像素哈希一致，但三个 fold 都从预热第 2 步出现数值轨迹差异；有支持候选步骤还存在 direct R/A 与 combined backward 的有限精度差别。不能把这个很小的端点增益解释为逐位隔离的单一系数因果效应。
- 实际覆盖 390 个预热步骤、1,162 个有支持步骤和 8 个预热后的无支持步骤；无支持位置两端均保持 EMA、使用 1/1 系数并正常执行其余监督。历史记录始终是候选，历史 anchor 数为 0。
- 只有 seed 42，且这些内部身份折已用于多轮方法开发。身份 bootstrap 不覆盖训练种子方差和方法选择偏差，不支持官方成绩、稳健性或整体目标达成。
- 第一次事后 sqrt 修复曾因 stats 源码绑定不符失败。因此应写“最终原始绑定字节已恢复并核验一致”，不要写成所有事后尝试期间源码从未临时变化。原失败文件和修复链均保留。

A、B、C、D、F 在报告声明的证据范围内通过，E 为 WARN。最新结果表、全部训练表及梯度表均与原始记录一致。后续只需将文档中的审计进行中状态更新为此结论、附上审计，并保留源码恢复措辞；不应据此晋级或挽救这个已失败的固定版本。审阅归属是 **same-family / provisional，backend 未获独立证明**。

<oai-mem-citation>
<citation_entries>
MEMORY.md:454-467|note=[used prior TriFusion protocol and evidence boundary guidance; all reported Q1 facts freshly checked]
</citation_entries>
<rollout_ids>
01a06f22-cc0d-7da2-8cae-68a700778767
</rollout_ids>
</oai-mem-citation>
