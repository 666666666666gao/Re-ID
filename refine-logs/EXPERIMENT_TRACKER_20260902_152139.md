# TriFusion V16 SATR Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics / Gate | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| V16-T0 | T0 | public contracts RED→GREEN | hard-pair / SATR loss / activity / Q1 gate | synthetic + adjacent tests | unit tests | MUST | TODO | 不改V15语义 |
| V16-M0A | M0a | exact、activity、paired draw | V16 initial SATR vs no-SATR | V12 fold source train | exact SHA、coverage、paired hashes、dev0/official0 | MUST | BLOCKED_BY_T0 | 三fold前8 batch |
| V16-M0B | M0b | real capacity | SATR | V12 fold0 source train | B64/K8 8-step、all gradients、0 overflow、<24GiB | MUST | BLOCKED_BY_M0A | 单3090 |
| V16-M0C | M0c | optimization sanity | SATR | fixed fold0 B64 | 100-step floor-aware ratio<=0.1 | MUST | BLOCKED_BY_M0B | optimizer0 test不算训练结果 |
| V16-Q1-0S | Q1 | fold0 method endpoint | SATR | 94 train / 47 heldout IDs | final mAP/R1/AP vectors | MUST | BLOCKED_BY_M0 | seed42,20e |
| V16-Q1-0N | Q1 | fold0 matched comparator | no-SATR | same | paired metrics | MUST | BLOCKED_BY_M0 | same draws/scope |
| V16-Q1-1S | Q1 | fold1 method endpoint | SATR | same protocol | final metrics | MUST | BLOCKED_BY_M0 | seed42,20e |
| V16-Q1-1N | Q1 | fold1 matched comparator | no-SATR | same | paired metrics | MUST | BLOCKED_BY_M0 | same draws/scope |
| V16-Q1-2S | Q1 | fold2 method endpoint | SATR | same protocol | final metrics | MUST | BLOCKED_BY_M0 | seed42,20e |
| V16-Q1-2N | Q1 | fold2 matched comparator | no-SATR | same | paired metrics | MUST | BLOCKED_BY_M0 | same draws/scope |
| V16-Q1-G | Q1 | aggregate scientific gate | SATR-noSATR | 571 queries / 21 IDs | fused≥+1, LB>0, all branch aggregate>0 | MUST | BLOCKED_BY_Q1 | FAIL则封存 |
| V16-D1 | D1 | unique deployable main | SATR all-fit | 141-fit/30-dev | fused≥65且严格胜出 | CONDITIONAL | BLOCKED_BY_Q1_GATE | train dev0, final dev1 |
| V16-A1 | P1 | minimal ablation | noSATR/single-peer/no-protect | frozen dev protocol | mechanism isolation | LOCKED | BLOCKED_BY_D1 | 主成功后 |
| V16-O1 | P1 | official/SOTA | frozen all171 endpoint | official test | same-track mAP/R1 | LOCKED | BLOCKED_BY_D1 | test once |

