# Closed Patch-memory panel: global versus additive role readout

CPU diagnosis completed at **2026-10-01 05:13:26 CST**, actual process exit 0. This uses the six previously accepted 50-epoch runs; it does not read the current FP32 competition experiment. Every query and complete gallery is retained, with original camera/time-period filtering and the same selected checkpoint. All 72 rescored metrics match the archived receipts within **0.000002738 percentage points**. Weights and distance arrays remain remote.

The comparison is between three outputs of **one checkpoint**. `shared_global` includes the trained M1 adaptation; it is not the original frozen baseline. `joint_local` is the normalized role correction. The final vector is the normalized sum of global and scaled correction, not an average of their distance matrices.

| Dataset / memory | Global mAP | Joint-local mAP | Fused mAP | Fused − global mAP | Rank-1 repairs / new errors | Joint-local correct, fused wrong |
|---|---:|---:|---:|---:|---:|---:|
| RGBNT201 / local | 72.4155 | 39.7552 | 72.5442 | +0.1287 | 2 / 1 | 31 |
| RGBNT201 / full | 72.4552 | 33.5180 | 72.5335 | +0.0784 | 2 / 1 | 17 |
| RGBNT100 / local | 85.0754 | 54.0293 | 85.1030 | +0.0276 | 2 / 1 | 19 |
| RGBNT100 / full | 85.0723 | 56.6576 | 85.1048 | +0.0325 | 2 / 1 | 19 |
| MSVR310 / local | 52.3949 | 11.9087 | 52.3622 | −0.0326 | 5 / 10 | 6 |
| MSVR310 / full | 52.6639 | 6.6501 | 52.2919 | −0.3720 | 4 / 10 | 5 |

There are 836 / 1715 / 591 legal queries respectively. Rows are repeated outputs on those same queries, not independent training seeds. “Repair” and “new error” compare fused with its own global output. The final column uses true labels for explanation; it is not a deployable selector or a promised gain.

1. **Small average changes also mean small first-rank changes on RGBNT201/100.** Both versions repair two first-rank errors and introduce one. The evidence supports a thin net role contribution at these selected checkpoints, rather than a claim that every local representation contains no useful information.
2. **MSVR310 shows a clear unfavorable balance within each checkpoint.** Ten new first-rank errors outweigh four or five repairs. Identity-equal AP changes are also negative: −0.2629 / −0.4884 points for local/full memory. This is broader than a query-weighted mean artifact, but does not identify a visual cause or prove all identities deteriorate.
3. **Standalone local correctness and additive usefulness are different.** Some queries remain fused-wrong despite joint-local being correct. Conversely, on RGBNT100 the local descriptor is wrong on both fused repairs; on MSVR310 it is wrong on every fused repair. Vector addition can change ranking without either constituent independently having the right first match. Local-correct counts cannot be converted into an ensemble score or an optimal mixing guarantee.
4. **The intervention cannot be uniquely blamed on the readout.** Local evidence formation, projection and joint training are coupled. This diagnostic does not replace an independently trained global-only control or a controlled readout intervention. Earlier global-only/context/auxiliary panels remain the relevant training controls.

The current registered six-end FP32 independent/competitive-slot panel continues unchanged. Its full outcome will test whether slot competition improves evidence formation and actual retrieval. No additional training hypothesis, selector, coefficient search, checkpoint choice or advancement threshold is introduced here.

Source: [diagnostic helper](../tools/diagnose_patch_memory_readout.py), SHA256 `d54b59a233ac4ae072c52c83f10d854542e0a16582d5346209abbbd9c1ddf978`. Full per-query AP/ranks and identity changes: `logs/closed_panel_readout_diagnosis_20261001.json`, SHA256 `b42358ac396029719bd3d37a286f17c93f26d63a477d126a0e435a346f1c11e9`. Original accepted matrix SHA256 `35067bce6690e86e43a08f963ddc259ade704549d899f8b2ce58cfb1dcb3aa56`; original scientific gate remains FAIL. One development seed and official epoch selection limit generalization claims. The overall research goal remains unmet.
