# TriFusion V16 SATR Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics / Gate | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| V16-T0 | T0 | public contracts RED→GREEN | hard-pair / SATR loss / activity / Q1 gate | synthetic + adjacent tests | unit tests | MUST | PASS | 相邻V8/V15/V16 23/23 PASS |
| V16-M0A | M0a | exact、activity、paired draw | V16 initial SATR vs no-SATR | V12 fold source train | exact SHA、coverage、paired hashes、dev0/official0 | MUST | FAIL | 配对全PASS；T三fold coverage=0，活动门FAIL |
| V16-M0B | M0b | real capacity | SATR | V12 fold0 source train | B64/K8 8-step、all gradients、0 overflow、<24GiB | MUST | PASS | 203/203 gradients；reserved5962MiB |
| V16-M0C | M0c | optimization sanity | SATR | fixed fold0 B64 | 100-step floor-aware ratio<=0.1 | MUST | PASS | excess ratio=0.064479 |
| V16-Q1-0S | Q1 | fold0 method endpoint | SATR | 94 train / 47 heldout IDs | final mAP/R1/AP vectors | MUST | NOT_RUN_M0_FAIL | seed42,20e |
| V16-Q1-0N | Q1 | fold0 matched comparator | no-SATR | same | paired metrics | MUST | NOT_RUN_M0_FAIL | same draws/scope |
| V16-Q1-1S | Q1 | fold1 method endpoint | SATR | same protocol | final metrics | MUST | NOT_RUN_M0_FAIL | seed42,20e |
| V16-Q1-1N | Q1 | fold1 matched comparator | no-SATR | same | paired metrics | MUST | NOT_RUN_M0_FAIL | same draws/scope |
| V16-Q1-2S | Q1 | fold2 method endpoint | SATR | same protocol | final metrics | MUST | NOT_RUN_M0_FAIL | seed42,20e |
| V16-Q1-2N | Q1 | fold2 matched comparator | no-SATR | same | paired metrics | MUST | NOT_RUN_M0_FAIL | same draws/scope |
| V16-Q1-G | Q1 | aggregate scientific gate | SATR-noSATR | 571 queries / 21 IDs | fused≥+1, LB>0, all branch aggregate>0 | MUST | NOT_RUN_M0_FAIL | V16封存 |
| V16-D1 | D1 | unique deployable main | SATR all-fit | 141-fit/30-dev | fused≥65且严格胜出 | CONDITIONAL | NOT_AUTHORIZED | dev0/official0 |
| V16-A1 | P1 | minimal ablation | noSATR/single-peer/no-protect | frozen dev protocol | mechanism isolation | LOCKED | NOT_AUTHORIZED | 主实验未通过 |
| V16-O1 | P1 | official/SOTA | frozen all171 endpoint | official test | same-track mAP/R1 | LOCKED | NOT_AUTHORIZED | official0 |
