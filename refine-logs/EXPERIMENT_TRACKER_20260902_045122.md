# TriFusion V9 Experiment Tracker — 2026-09-02 04:51 +08:00

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| V8-A | M0 | pretrained-tail 专家形成 | V8 Phase-A | 141-fit→final dev once | branch Oracle 64.7850；fixed fused 58.0972 | MUST | COMPLETE—PARTIAL | 互补存在；Oracle 非部署且低于65 |
| V8-B1 | M0 | fit-only continuous margin Router | V8 Phase-B | 3-fold identity OOF | learned/fixed margin 0.102034/0.101720 | MUST | COMPLETE—WEAK PASS | 400 Router steps；experts frozen；dev0/official0 |
| V8-B2 | M0 | frozen deployable evaluation | V8 Phase-B | 30-dev final once | fused 58.4050/59.3939 | MUST | COMPLETE—FAIL | 超 baseline/三专家；低65门6.5950；next phase false |
| V8-C | M0 | result-to-claim + integrity | V8 Phase-B | frozen receipts | partial/medium；WARN | MUST | COMPLETE | WARN 仅 remote-only artifact packaging |
| V9-R000 | M1 | 公共接缝 TDD | orthogonal triadic relay synthesis | synthetic/fake Signal | exact prefix、peer response、orthogonality、gradient/freeze | MUST | IN PROGRESS | 已确认接缝；严格 RED→GREEN |
| V9-R001 | M2 | RTX3090 capacity | V9 single config | train-only B64/K8 8 step | peak MiB、finite、grad、state SHA | MUST | BLOCKED BY V9-R000 | dev0/official0 |
| V9-R002 | M2 | 固定批学习能力 | V9 single config | same real B64/K8 100 step | excess-loss ratio≤0.10 | MUST | BLOCKED BY V9-R001 | dev0/official0 |
| V9-R003 | M3 | 完整主实验 | V9 final-only | 141-fit/30-dev | baseline/Phase-B/fused/CNN/T/M | MUST | BLOCKED BY READINESS | seed42、60 epoch；训练中dev0；final dev once |
| V9-R004 | M4 | all171 + official once | V9 fixed | all171→official | mAP/R1/5/10 | CONDITIONAL | BLOCKED | 仅 V9-R003 fused≥65且胜全部输出后解锁 |
| V9-R005+ | M5 | 论文消融 | frozen V9 controls | frozen protocols | effect sizes | CONDITIONAL | BLOCKED | 仅 official 超公开最佳后解锁 |

硬约束：全部 GPU/数据/conda/训练/评估只在云端 RTX3090；seed42 only；不复跑 baseline；不做超参扫描；主结果通过前不做消融或 official test；同协议未超过公开最佳不得宣称 SOTA。
