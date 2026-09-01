# TriFusion V2 Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| V2-R000 | M0 | APSD/SURE/QIPF 行为契约 | `shared_semantic_cascade_v2` | synthetic/CPU + real CLIP | anchor equality、shape、stage refresh、mask、gradient | MUST | PASS | V2专项9 PASS；联合回归37 PASS |
| V2-R001 | M0 | 真实容量门 | V2 B32/K4 AMP | RGBNT201 train-only | VRAM、finite、gradient coverage、params | MUST | PASS | 93,965,138参数；峰值allocated/reserved 5739.87/6248 MiB；coverage=1.0 |
| V2-R002 | M0 | 学习门 | V2 100-step overfit | RGBNT201 train-only | loss decrease、nonfinite | MUST | PASS | 22.9879→2.0428；ratio=0.08886；finite/coverage PASS |
| V2-R003 | M1 | 强语义主生成器 | V2 HFER-uniform | 141-fit/30-dev | fused/branch mAP、R1、projection cosine | MUST | READY | M0 全过；等待冻结提交后启动，≥65 mAP 且 fused>best branch 才晋级 |
| V2-R004 | M2 | 新效用 target | V2 OOF CIRC | train folds only | overlap、hash、effect/rank coverage | MUST | TODO | 必须由 V2 生成器重建，禁止复用 V1 cache |
| V2-R005 | M2 | 完整主方法 dev | V2 CIRC+SURE+QIPF | 141-fit/30-dev | retrieval、AUROC、Brier/ECE、router std | MUST | TODO | ≥70 mAP、领先分支≥1.0、AUROC≥0.65、std≥0.03 |
| V2-R006 | M3 | 正式冻结 | exact V2 candidate | no test | SHA、config、cache、selection receipt | MUST | BLOCKED | 等 V2-R005 全部门 |
| V2-R007 | M3 | 新正式主实验 | seed42 fixed endpoint | full171→official once | mAP、R1/5/10 | MUST | BLOCKED | 只有 >85.3/87.9 后才允许消融 |

## 2026-09-01 13:29 +08:00 execution evidence

- V2专项：`tests/test_trifusion_cascade_v2.py`，9 PASS；真实 CLIP checkpoint 已参与 builder contract。
- 联合回归：cascade/builder/collaboration/criterion/runner/interventions，37 PASS。
- Preflight：`artifacts/trifusion_cascade_v2_hfer_uniform_seed42_preflight/preflight.json`，RTX3090 free 24572 MiB，数据/CLIP/protocol 哈希通过，official-test access=0。
- Capacity：`artifacts/trifusion_cascade_v2_hfer_uniform_seed42_capacity/capacity.json`，B32/K4、8步、无 AMP overflow、无非有限梯度、official-test access=0、dev iterations=0。
- Overfit：`artifacts/trifusion_cascade_v2_hfer_uniform_seed42_overfit/overfit.json`，固定批100步，loss ratio 0.08886、official-test access=0、dev iterations=0。
