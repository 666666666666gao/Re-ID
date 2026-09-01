# TriFusion Signal-preserving V7 seed-42 dev result — 2026-09-02

V7 completed its only authorized 60-epoch run on the fixed RGBNT201
141-fit/30-dev protocol. Execution passed, but the preregistered result gate
failed. This is a negative held-out-dev result, not an official-test or SOTA
result.

## Reloaded selected checkpoint

The selection metric was fused dev mAP. Epoch 1 was selected and then strictly
reloaded from checkpoint SHA-256
`8bcdf3583e121dd7a7b0071743b8fd34f93a82cc710bd07e26b53c3693609a2b`.

| Output | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| baseline_only | 58.0109 | 57.4545 | 69.9394 | 76.6061 |
| fused | 58.3293 | 57.9394 | 70.1818 | 76.7273 |
| CNN | 58.2773 | 57.4545 | 69.9394 | 76.9697 |
| Transformer | 58.3028 | 58.0606 | 70.1818 | 76.6061 |
| Mamba | **58.3476** | 57.8182 | **70.3030** | **76.9697** |

Fused improves the exact Signal baseline by 0.3184 mAP, but trails Mamba by
0.0183 and misses the 65 mAP development threshold by 6.6707. It therefore
fails the requirement to beat every fixed branch and the absolute gate.

## Completion and integrity

- 60/60 epochs and 2,520 optimizer steps completed in 2,419.6 seconds.
- Real B64/K8, zero AMP overflow; peak allocated/reserved memory was
  11,176.8/12,908 MiB on the RTX 3090.
- Reloaded metrics match selected metrics for every output and metric.
- Frozen Signal state SHA-256 is identical before training, after training and
  after strict reload.
- Official-test access count is zero.
- An independent result-to-claim review returned `claim_supported=no` with
  high confidence. A V7-specific independent integrity audit remains missing,
  so the integrity label is provisional/warn.

## Why this is not undertraining

The best router-warmup result occurs at epoch 1. The best joint-phase fused
result is only 57.9804 mAP at epoch 11, and epoch 60 ends at 57.7550 even as
training loss reaches 0.74763. More training did not improve held-out retrieval.

The selected epoch-1 checkpoint has updated reliability and fusion parameters,
but CNN, Transformer, Mamba, HFER and retrieval heads are still unchanged from
the V6 initialization. Once those modules enter joint training, dev quality
decreases.

## Read-only diagnosis

The paired diagnostic performs zero optimizer steps and no official-test
access. On all 825 dev queries it finds:

- joint-router normalized entropy 0.99791 and modality entropy 0.99994;
- alpha 0.198947 with standard deviation 0.0000015;
- only 14.0625% top-slot agreement between predicted routing and per-slot
  marginal identity utility on one deterministic B64 fit batch;
- fused/baseline distance correlation 0.999786 and 99.6364% Top-10 overlap;
- fused wins 273 query APs versus Mamba, loses 248 and ties 304, but produces
  only two Rank-1 repairs and one break.

The residual-only experts remain meaningfully complementary: fixed CNN,
Transformer and Mamba residuals reach 59.1317, 54.8594 and 57.8991 mAP, while a
ground-truth Oracle reaches 62.7435 mAP, 3.6118 above the strongest fixed
residual. All three leave-one-out expert margins are positive. Oracle uses
query ground truth and is diagnostic only, never a deployable result.

## Claim boundary

V7 supports only the narrow statement that exact Signal preservation plus the
new residual path improves the fixed held-out baseline by 0.3184 mAP. It does
not support learned collaborative superiority, the 65 mAP gate, official-test
performance, innovation effectiveness or SOTA. Official testing, ablations and
multiple seeds remain closed.

Evidence:

```text
evidence/trifusion_signal_preserving_v7_dev_terminal_seed42.json
evidence/trifusion_signal_preserving_v7_diagnostic_seed42.json
```
