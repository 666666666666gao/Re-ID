# Archived four-end trajectory diagnosis, 2026-10-01

Only four accepted endpoints are covered. RGBNT100 remains pending; this does not alter formal best checkpoints or the original six-end gate. All source receipts and step logs are SHA-bound in the accompanying JSON.

| Dataset / mode | Best epoch | Best → final mAP | ΔmAP pp | Best → final R1 | ΔR1 pp | Positive Triplet steps after best |
|---|---:|---:|---:|---:|---:|---:|
| RGBNT201 / local_memory | 2 | 72.5442 → 69.1778 | -3.3664 | 73.9234 → 70.2153 | -3.7081 | 1061/2543 |
| MSVR310 / local_memory | 10 | 52.3622 → 48.1347 | -4.2276 | 67.8511 → 64.2978 | -3.5533 | 771/800 |
| RGBNT201 / full_memory | 2 | 72.5335 → 70.0578 | -2.4758 | 73.9234 → 71.5311 | -2.3923 | 1012/2543 |
| MSVR310 / full_memory | 10 | 52.2919 → 48.5888 | -3.7031 | 67.8511 → 65.8206 | -2.0305 | 766/800 |

Training loss decreases from best epoch to final epoch in all four runs. Triplet remains positive in some final-epoch steps; MSVR310 has positive Triplet in most post-best steps. Thus complete absence of hinge supervision is not an explanation supported by these two MSVR runs. Positive loss is not an AdamW update share or proof of why unknown-identity retrieval declines.

The official selected best results remain unchanged. Epoch selection consumed official data; these trajectories are descriptive observations of one development seed, not independent significance or a basis for test-specific retuning. No current training progress was read for this analysis.

Evidence: [bound source analysis](PATCH_MEMORY_ARCHIVED_TRAJECTORY_2026-10-01.json), original50-epoch training receipts and every original training step.
