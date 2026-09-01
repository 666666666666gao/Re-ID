# TriFusion V2 Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| V2-R000 | M0 | APSD/SURE/QIPF 行为契约 | `shared_semantic_cascade_v2` | synthetic/CPU | anchor equality、shape、stage refresh、mask、gradient | MUST | IN PROGRESS | 不访问数据/test |
| V2-R001 | M0 | 真实容量门 | V2 B32/K4 AMP | RGBNT201 train-only | VRAM、finite、gradient coverage、params | MUST | TODO | RTX3090 free≥22000 MiB 后启动 |
| V2-R002 | M0 | 学习门 | V2 100-batch overfit | RGBNT201 train-only | loss decrease、nonfinite | MUST | TODO | 不评估 dev/test |
| V2-R003 | M1 | 强语义主生成器 | V2 HFER-uniform | 141-fit/30-dev | fused/branch mAP、R1、projection cosine | MUST | TODO | ≥65 mAP 且 fused>best branch 才晋级 |
| V2-R004 | M2 | 新效用 target | V2 OOF CIRC | train folds only | overlap、hash、effect/rank coverage | MUST | TODO | 必须由 V2 生成器重建，禁止复用 V1 cache |
| V2-R005 | M2 | 完整主方法 dev | V2 CIRC+SURE+QIPF | 141-fit/30-dev | retrieval、AUROC、Brier/ECE、router std | MUST | TODO | ≥70 mAP、领先分支≥1.0、AUROC≥0.65、std≥0.03 |
| V2-R006 | M3 | 正式冻结 | exact V2 candidate | no test | SHA、config、cache、selection receipt | MUST | BLOCKED | 等 V2-R005 全部门 |
| V2-R007 | M3 | 新正式主实验 | seed42 fixed endpoint | full171→official once | mAP、R1/5/10 | MUST | BLOCKED | 只有 >85.3/87.9 后才允许消融 |
