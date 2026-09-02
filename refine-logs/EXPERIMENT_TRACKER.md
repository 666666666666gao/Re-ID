# TriFusion V12 Experiment Tracker — 2026-09-02 07:09 +08:00

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| V8-A | historical | pretrained-tail expert formation | V8 Phase-A | 141-fit→dev once | branch Oracle 64.7850；fixed fused 58.0972 | MUST | COMPLETE—PARTIAL | 互补存在，非部署 Oracle |
| V8-B | historical | OOF margin Router + dev | V8 Phase-B | contaminated fit OOF→30-dev | fused 58.4050/59.3939 | MUST | COMPLETE—FAIL | 胜基线/三专家但低65；教师底层含 all-fit Signal |
| V9 | historical | orthogonal triadic relay | final-only dev | 141-fit→30-dev | fused 56.5339 | MUST | COMPLETE—FAIL | 已封存 |
| V10-Q0 | historical | DINO complement | fit-only | 571 query | Phase-B/DINO/concat=100/7.6284/92.2120 | MUST | COMPLETE—FAIL | Signal 前缀饱和 |
| V11-Q0 | historical | fold residual/DINO | adapter-OOF only | 3 fold / 571 query | bank/DINO/concat=100/14.1323/95.8582 | MUST | COMPLETE—FAIL | all-fit Signal token 泄漏，已封存 |
| V12-P0 | M0 | preregistration | complete-path identity-OOF teacher | plan/tracker | fixed claims/gates/cost | MUST | COMPLETE | 不复跑 baseline；fold Signal 仅为内部教师 |
| V12-T0 | M0 | public-seam RED→GREEN | fold registry + fail-closed gate + CLI receipt | synthetic | 4次真实RED→GREEN；相邻8 tests PASS | MUST | COMPLETE | 未改科学门 |
| V12-P1 | M1 | real one-step preflight | fold0 Signal internal teacher | fold0 fit B64/K8 | 1 step、191 gradients、0 overflow、10701.82MiB allocated | MUST | COMPLETE—PASS | dev0/official0 |
| V12-Q0 | M2 | complete-path OOF target generation | 3 fold Signal50 + experts20 | 141-fit only / 571 query | fixed expert best86.9549；Oracle92.2679；gain5.3130 | MUST | COMPLETE—PASS | overlap0；三专家/三模态多样性门PASS；dev0/official0 |
| V12-Q0-C | M2 | result-to-claim + integrity | terminal Q0/Q1 receipts | frozen files | partial/high；WARN/warn | MUST | COMPLETE | Q0窄资格支持；Q1不晋级；remote-only二进制包装WARN |
| V12-Q1 | M3 | train hierarchical Router | existing V8 Phase-A + V12 cache | fit-only OOF | margin -0.117330<-0.099975；Top1 12.2592%<16.8126% | CONDITIONAL | COMPLETE—FAIL | quality/missing PASS；combined checkpoint null；dev0/official0 |
| V12-R001 | M4 | unique frozen dev | V12 combined checkpoint | 30-dev once | 未执行 | CONDITIONAL | CLOSED—Q1 FAILED | 未生成combined checkpoint，禁止dev |
| V12-OFFICIAL | M5 | official test once | fixed promoted V12 | all171→official | 未执行 | CONDITIONAL | CLOSED—NO DEV PROMOTION | official access0 |
| V12-ABLATION | M6 | paper claim isolation | fixed promoted V12 | frozen protocols | 未执行 | CONDITIONAL | CLOSED—MAIN TARGET NOT MET | 用户要求主实验先超过目标 |

硬约束：全部 GPU/数据/conda/训练/评估只在云端 RTX3090；seed42 only；fold Signal 不作为 baseline 结果，不访问 dev/official；不做超参扫描；主结果通过前不做消融或 official；同协议未超过公开最佳不得宣称 SOTA。
