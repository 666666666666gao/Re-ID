# Fixed same-evidence comparison: terminal results904

The original903 invocation completed all three endpoints on server26 GPU0/1, with zero optimizer updates. Original coordinator and endpoint exits are0. The saved original first-observer receipt is retained. The original remote terminal state was collected without restarting the diagnostic. Executor continuity notes are separate from verified result artifacts.

All seven predeclared full-gallery comparisons are reported. Each uses the same selected raw-semantic824 best (201 E8, MSVR E38,100 E5); global is from that same model, not an independently retrained ablation. Source/reference comparisons have different training history. No seed, coefficient, candidate-gallery, checkpoint or parameter selection was performed.

| Frozen comparison (mAP / R1, %) | RGBNT201 | MSVR310 | RGBNT100 |
|---|---:|---:|---:|
| Same-model global | 74.296664 / 78.947368 | 50.542129 / 68.020305 | 84.369219 / 96.093294 |
| Original additive1536 | 74.374862 / 78.827751 | 50.542225 / 67.851100 | 84.090303 / 95.860058 |
| Residual alone1536 | 11.578064 / 9.090909 | 6.820775 / 16.243655 | 36.625906 / 57.084548 |
| Same-residual block3072 | 74.338753 / 78.947368 | 50.538969 / 68.020305 | 84.077846 / 95.976676 |
| Indexed region mean | 14.237502 / 13.995215 | 8.769176 / 21.319797 | 41.684449 / 62.448980 |
| Symmetric region MaxSim | 14.160263 / 14.354067 | 8.773044 / 21.319797 | 44.634323 / 68.163265 |
| Fixed-energy regional block | 74.326469 / 78.947368 | 50.540551 / 68.020305 | 84.457925 / 95.918367 |

| Dataset / contrast vs same-model global | ΔmAP | ΔR1 | Repairs / new errors | Worst5% mean ΔAP |
|---|---:|---:|---:|---:|
| RGBNT201 / additive | +0.078198 | -0.119617 | 3 / 4 | -3.122669 |
| RGBNT201 / same_residual_block | +0.042089 | +0.000000 | 0 / 0 | -0.469897 |
| RGBNT201 / fixed_energy_region_block | +0.029805 | +0.000000 | 0 / 0 | -0.360292 |
| MSVR310 / additive | +0.000096 | -0.169205 | 0 / 1 | -0.571278 |
| MSVR310 / same_residual_block | -0.003160 | +0.000000 | 0 / 0 | -0.068783 |
| MSVR310 / fixed_energy_region_block | -0.001577 | +0.000000 | 0 / 0 | -0.032128 |
| RGBNT100 / additive | -0.278915 | -0.233236 | 20 / 24 | -15.299264 |
| RGBNT100 / same_residual_block | -0.291373 | -0.116618 | 8 / 10 | -12.008198 |
| RGBNT100 / fixed_energy_region_block | +0.088707 | -0.174927 | 3 / 6 | -4.437161 |

## What this closes

Exact same-residual block comparison does not improve mAP over the additive comparator on any dataset. It jointly removes cross-branch dot products and changes normalization, so this is not a causal intervention isolating the cross terms. No general benefit from separating the two vectors is established.

RGBNT100 final-region MaxSim gives44.634323mAP versus41.684449 for indexed region comparison and36.625906 for the residual. With original gain/norm energy, the regional block gives84.457925: +0.088707mAP but -0.174927R1 versus its own global;3 first-rank repairs and6 new errors. Versus original additive it is +0.367622mAP, but the comparison changes retained representation and scoring. It does not establish an acceptable overall advance, a stronger teacher, or a1536 static method. The other datasets have no corresponding meaningful benefit.

Final-slot mean off-diagonal cosine is0.920373/0.575988/0.836426 for query201/MSVR/100. These are actual final C/T/M inputs consumed by readout. They indicate directional similarity within groups, not identified parts, physical visibility, or proof of semantic collapse. Higher local-only scores do not imply complementary retrieval.

The next mechanism must learn the evidence actually consumed by identity comparison. Do not add a learned matcher, visibility gate, gain search or distillation teacher solely from these results. A future score-consistent training probe must be registered separately and disclose its retained-region/pairwise deployment cost; this report authorizes no automatic training rollout.

## Execution and evidence limits

Original additive metrics reproduced all four retained metrics within1e-5. Maximum fused-distance differences are7.15e-7/7.15e-7/1.19e-6 (201/MSVR/100), not bitwise historical distance identity. Model state hashes unchanged, parameter gradients absent, no training. Endpoint elapsed60.673/67.226/155.923s, local-score computation0.998/0.891/20.326s; these are this implementation’s wall-clock measurements, not a complete production inference benchmark.

Regional caches are retained on26 as current inputs to a future evidence-learning diagnosis. This text report does not load or authenticate their tensor content. Official camera/scene filters and complete galleries are checked in the actual source and result audit; image-byte provenance remains the sealed historical record. All official sets have been used for development. Fixed-model identity summaries/worst tails are descriptive, not full-training seed uncertainty.

All-three-datasets-over-full-Signal goal remains ACTIVE_UNMET. The user-resume record contains a historical full-Signal reference80.3029; it is not a newly reauthenticated comparator in this diagnostic. No new trained checkpoint or claimed module contribution was created; the full goal is not proved by these fixed-model results.
