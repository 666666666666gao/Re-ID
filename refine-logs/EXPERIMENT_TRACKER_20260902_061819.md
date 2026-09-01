# TriFusion V10 Experiment Tracker — 2026-09-02 06:18 +08:00

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| V8-A | M0 | pretrained-tail 专家形成 | V8 Phase-A | 141-fit→final dev once | branch Oracle 64.7850；fixed fused 58.0972 | MUST | COMPLETE—PARTIAL | 互补存在；Oracle 非部署且低于65 |
| V8-B1 | M0 | fit-only continuous margin Router | V8 Phase-B | 3-fold identity OOF | learned/fixed margin 0.102034/0.101720 | MUST | COMPLETE—WEAK PASS | 400 Router steps；experts frozen；dev0/official0 |
| V8-B2 | M0 | frozen deployable evaluation | V8 Phase-B | 30-dev final once | fused 58.4050/59.3939 | MUST | COMPLETE—FAIL | 超 baseline/三专家；低65门6.5950；next phase false |
| V8-C | M0 | result-to-claim + integrity | V8 Phase-B | frozen receipts | partial/medium；WARN | MUST | COMPLETE | WARN 仅 remote-only artifact packaging |
| V9-R000 | M1 | 公共接缝 TDD | orthogonal triadic relay synthesis | synthetic/fake Signal | exact prefix、peer response、orthogonality、gradient/freeze | MUST | COMPLETE—PASS | RED→GREEN；V9+相邻V8共12 tests |
| V9-R001 | M2 | RTX3090 capacity | V9 single config | train-only B64/K8 8 step | peak MiB、finite、grad、state SHA | MUST | COMPLETE—PASS | 59/59 gradients；2020MiB reserved；overflow0；dev0/official0 |
| V9-R002 | M2 | 固定批学习能力 | V9 single config | same real B64/K8 100 step | excess-loss ratio≤0.10 | MUST | COMPLETE—PASS | loss 3.78850→0.61228；excess ratio0.000518；dev0/official0 |
| V9-R003 | M3 | 完整主实验 | V9 final-only | 141-fit/30-dev | baseline/Phase-B/fused/CNN/T/M | MUST | COMPLETE—FAIL | 60/60、2520 steps、overflow0；fused56.5339，低Phase-B1.8711，低65门8.4661；dev1/official0 |
| V9-R003-C | M3 | result-to-claim + integrity | V9 terminal receipts | frozen evidence | claim / integrity | MUST | COMPLETE | `no/high`；`WARN/warn/FAIL_TO_PROMOTE` |
| V9-R004 | M4 | all171 + official once | V9 fixed | all171→official | mAP/R1/5/10 | CONDITIONAL | CLOSED—GATE FAILED | V9-R003 未达65且未胜baseline/Phase-B；不得执行 |
| V9-R005+ | M5 | 论文消融 | frozen V9 controls | frozen protocols | effect sizes | CONDITIONAL | CLOSED—GATE FAILED | 不运行消融、多seed或任何事后扫描 |
| V10-Q0 | Q0 | DINOv2 冻结互补资格 | Phase-B / DINO / fixed equal-block concat | 21 fit IDs、571 query only | fixed retrieval、2-source Oracle、strict load | MUST | COMPLETE—FAIL | PhaseB/DINO/concat mAP=100/7.6284/92.2120；Oracle gain0；wins571/0；optimizer0/dev0/official0 |
| V10-Q0-C | Q0 | result-to-claim + integrity | terminal Q0 receipt | frozen evidence | claim / audit | MUST | COMPLETE | `no/high`；`WARN/warn/FAIL_TO_QUALIFY` |
| V10-Q1 | Q1 | Dual-foundation 三专家公共接缝/容量 | 条件式 V10 | synthetic + 141-fit B64/K8 | exact prefix、typed peer、gradient、VRAM | CONDITIONAL | CLOSED—Q0 FAILED | 不实现；不做DINO事后扫描 |
| V10-Q2 | Q2 | 三折 identity-OOF 表示效用 | 条件式 V10 | fit IDs disjoint folds | combined held-out fused - Phase-B ≥1.0 mAP | CONDITIONAL | CLOSED—Q0 FAILED | 不训练、不访问dev |
| V10-R001 | M3 | 完整主实验 | 条件式 V10 final-only | 141-fit/30-dev | baseline/Phase-B/fused/CNN/T/M | CONDITIONAL | CLOSED—Q0 FAILED | 未执行 |
| NEXT | — | 非饱和身份隔离新假设 | 未预注册 | train-only gate first | positive retrieval utility and harm suppression | MUST BEFORE GPU | PENDING HYPOTHESIS | 未授权V11或新GPU运行 |

硬约束：全部 GPU/数据/conda/训练/评估只在云端 RTX3090；seed42 only；不复跑 baseline；不做超参扫描；主结果通过前不做消融或 official test；同协议未超过公开最佳不得宣称 SOTA。V10-Q0 是零训练 fit-only 资格门，不是消融或主结果。
