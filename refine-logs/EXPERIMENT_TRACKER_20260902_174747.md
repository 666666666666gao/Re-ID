# V17 DTRED Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| V17-T0 | T0 | public seams + adjacent regression | DTRED source/tests | code | tests | MUST | TODO | RED->GREEN；不加额外模块 |
| V17-M0A | M0A | exact/frozen/source-only/paired preflight | DTRED + weight0 receipts | V12 3-fold source batches | hashes/pair coverage | MUST | BLOCKED_BY_T0 | dev0/official0 |
| V17-M0B | M0B | physical capacity | DTRED | fold0 B64/K8 | gradients/overflow/VRAM | MUST | BLOCKED_BY_M0A | 不预先降batch |
| V17-M0C | M0C | fixed-batch optimization | DTRED | fold0 100 steps | excess-loss ratio | MUST | BLOCKED_BY_M0B | <=0.10 |
| V17-Q1 | Q1 | identity-OOF mechanism qualification | DTRED vs matched weight0 | 3 folds x20 epochs | mAP gain/bootstrap/branch/envelope | MUST | BLOCKED_BY_M0 | 全门conjunction |
| V17-D1 | D1 | no-reranking main dev | all-fit DTRED | 141-fit/30-dev | mAP/CMC | CONDITIONAL | NOT_AUTHORIZED | 仅Q1 pass；>=65且strict wins |
| V17-OFFICIAL | post-success | official/SOTA comparison | frozen successful D1 | RGBNT201 official | mAP/CMC | CONDITIONAL | NOT_AUTHORIZED | D1成功后另行冻结 |
| V17-ABL | post-success | minimum paper ablations | frozen successful D1 | later | claim isolation | CONDITIONAL | NOT_AUTHORIZED | 用户要求主实验SOTA后再做 |

## Frozen Boundaries

- seed42 only；remote RTX3090 only；physical B64/K8。
- `RERANKING=false`；no runtime fallback；no baseline rerun。
- Q1是train-only资格结果，不能写成dev性能。
- Q1失败即封存V17；不扫描width/loss/LR/epoch/checkpoint。
- D1失败则无SOTA/部署claim，official与消融继续不授权。
