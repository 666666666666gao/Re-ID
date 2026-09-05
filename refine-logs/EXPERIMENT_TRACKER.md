# V17 DTRED Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| V17-T0 | T0 | public seams + adjacent regression | DTRED source/tests | code | tests | MUST | DONE_PASS | 源码535ef2f；2026-09-05远端12项相关测试通过 |
| V17-M0A | M0A | exact/frozen/source-only/paired preflight | DTRED + weight0 receipts | V12 3-fold source batches | hashes/pair coverage | MUST | DONE_PASS | 三折配对/合法pair/冻结状态通过；dev0/official0 |
| V17-M0B | M0B | physical capacity | DTRED | fold0 B64/K8 | gradients/overflow/VRAM | MUST | DONE_PASS | 8步，22/22梯度，overflow0，1808MiB |
| V17-M0C | M0C | fixed-batch optimization | DTRED | fold0 100 steps | excess-loss ratio | MUST | DONE_PASS | 100步excess ratio0.000693508 |
| V17-Q1 | Q1 | identity-OOF mechanism qualification | DTRED vs matched weight0 | 3 folds x20 epochs | mAP gain/bootstrap/branch/envelope | MUST | DONE_FAIL | 全3360步完成；fused aggregate -0.338635；不授权D1 |
| V17-D1 | D1 | no-reranking main dev | all-fit DTRED | 141-fit/30-dev | mAP/CMC | CONDITIONAL | NOT_AUTHORIZED | 仅Q1 pass；>=65且strict wins |
| V17-OFFICIAL | post-success | official/SOTA comparison | frozen successful D1 | RGBNT201 official | mAP/CMC | CONDITIONAL | NOT_AUTHORIZED | D1成功后另行冻结 |
| V17-ABL | post-success | minimum paper ablations | frozen successful D1 | later | claim isolation | CONDITIONAL | NOT_AUTHORIZED | 用户要求主实验SOTA后再做 |

## Frozen Boundaries

- seed42 only；remote RTX3090 only；physical B64/K8。
- `RERANKING=false`；no runtime fallback；no baseline rerun。
- Q1是train-only资格结果，不能写成dev性能。
- Q1失败即封存V17；不扫描width/loss/LR/epoch/checkpoint。
- D1失败则无SOTA/部署claim，official与消融继续不授权。

## 2026-09-05 接续核查

M0/Q1原始回执已纳入evidence；Q1完整结束且失败，不存在待续训进程。
完整留出gallery只读补评由`tools/audit_v17_full_gallery.py`执行；它不改变原始
Q1终态，也不打开D1。参见`results/TRIFUSION_RGBNT201_V17_DTRED_2026-09-05.md`。
