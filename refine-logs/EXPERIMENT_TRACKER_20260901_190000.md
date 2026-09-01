# TriFusion V5 Baseline-First Experiment Tracker — 2026-09-01 19:00 +08:00

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| V3-R003 | M0 | 完整主实验 | task-anchor V3 | 141-fit/30-dev | best fused 42.8978/43.8788 | MUST | COMPLETE—FAIL | e14 best；低于 65 mAP 门 22.1022；fused 低于 Transformer 0.1190 mAP |
| V3-D001 | M0 | 冻结最佳分解 | V3 epoch14 | 30-dev | anchor/residual/fused、norm、entropy | MUST | COMPLETE—PASS | anchor 42.4787；residual 42.8225；router entropy≈0.9998；residual/anchor norm≈0.216；official0 |
| V3-C001 | M0 | 独立结果主张判断 | V3 result-to-claim | frozen receipts | claim_supported | MUST | COMPLETE—NO | 只支持负结果和结构瓶颈诊断；不支持互补融合/SOTA |
| V4-R000 | M1 | 公共接口 TDD | energy-calibrated utility-routed residual bank | synthetic + real CLIP build | shape、energy、utility target、zero-residual、gradient | MUST | COMPLETE—PASS | 6 项 V4 专项、23 项相邻回归、146 项内部全回归通过；外部 baseline runner 文件按用户约束排除 |
| V4-R001 | M1 | RTX3090 容量门 | V4 core | train-only B32/K4 8 steps | peak MiB、finite、gradient coverage、overflow | MUST | COMPLETE—PASS | 95,197,266 参数；6043.58 MiB allocated/6548 MiB reserved；366/366 梯度；AMP scale256 下 0 overflow；official0 |
| V4-R002 | M1 | 固定批学习能力门 | V4 core | same batch 100 steps | loss ratio≤0.10、official0 | MUST | COMPLETE—PASS | 14.91096→0.99563，ratio=0.0667716；0 overflow；366/366 梯度；official0 |
| V4-R003 | M2 | 完整 60-epoch dev 主实验 | V4 core | 141-fit/30-dev | fused/CNN/T/M mAP/R1 | MUST | COMPLETE—FAIL | e27 fused 43.4031/42.7879；Mamba 44.0659/43.5152；距65门21.5969；V4 anchor未测；official0 |
| V4-R004 | M3 | frozen all171 + official once | V4 final | all171→official test | mAP/R1/5/10 | CONDITIONAL MUST | BLOCKED—NOT AUTHORIZED | V4-R003 未通过，禁止 official |
| V4-R005+ | M4 | 主张消融与公平比较 | V4 controls | frozen protocols | effect sizes | CONDITIONAL | BLOCKED | V4 已失败；当前禁止消融和多种子 |
| V5-R000 | M0 | Signal 完整路径审计 | Signal cd1b0a6 | source/checkpoint/env | 3072D/SIM/SIE/provenance | MUST | IN PROGRESS | 上游80.3/85.2仍未本地复现；远端无Signal checkpoint/conda env |
| V5-R001 | M1 | baseline-only 同协议门 | exact Signal baseline | 141-fit/30-dev | baseline mAP/R1/5/10 | MUST | PENDING | seed42；不得使用 official test 选点 |
| V5-R002 | M1 | baseline-preserving TDD/capacity | V5 | synthetic+train-only | feature parity、梯度隔离、VRAM | MUST | PENDING | 同 checkpoint 输出 baseline-only/fused；baseline 路径冻结 |
| V5-R003 | M2 | 完整 60-epoch dev 主实验 | V5 | 141-fit/30-dev | baseline/fused/三专家 | MUST | BLOCKED | fused≥baseline、fused>best expert、fused≥65 才晋级 |
| V5-R004 | M3 | fixed all171 + official once | V5 | all171→official | mAP/R1/5/10 | CONDITIONAL | BLOCKED | 仅 V5-R003 全门通过；正式>85.3/>87.9 才解锁消融 |

硬约束：所有 GPU、数据、conda、训练和评估仅在远端 RTX3090；seed42 only；只做单一 Signal baseline floor，不做 baseline 矩阵；主结果超过冻结正式目标前不运行消融；未有同协议证据不得宣称 SOTA。
