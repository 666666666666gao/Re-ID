# V14 Experiment Tracker

| Run ID | Milestone | Purpose | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|
| V14-M0 | M0 | fold-bound risk/public seam TDD | synthetic only | exact risk, binding, gradients, parity | MUST | TODO | 无数据/GPU claim |
| V14-Q0 | Q0 | real-cache zero-step qualification | 571 fit queries / 3 folds | risk support, grad coverage, SHA, access | MUST | BLOCKED_BY_M0 | optimizer steps=0 |
| V14-Q1-S42 | Q1 | sole OOF Router qualification | 2-fold train / 1-fold heldout ×3 | risk/AP/margin gains + bootstrap + safety | MUST | BLOCKED_BY_Q0 | failure seals V14 |
| V14-D1-S42 | D1 | final refit + one dev | 141-fit / 30-dev | five-output mAP/R1, reload parity | CONDITIONAL | BLOCKED_BY_Q1 | official0; requires fused≥65 and strict wins |

