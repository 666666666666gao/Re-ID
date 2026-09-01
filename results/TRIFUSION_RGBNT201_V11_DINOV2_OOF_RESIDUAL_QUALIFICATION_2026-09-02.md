# TriFusion V11-Q0 DINOv2 OOF Residual Qualification

- **Date**: 2026-09-02
- **Hardware**: one remote NVIDIA GeForce RTX 3090
- **Code**: `a29692a59dc1458d3dc2b8ffe1b7cb0c64244644`
- **Evaluation type**: `real_gt_fit_identity_oof_residual_only`

## Outcome

V11-Q0 completed, but every scientific qualification condition failed.
V11-Q1, V11-Q2 and dev evaluation are not authorized.

| Fixed output | mAP | Rank-1 |
|---|---:|---:|
| CNN residual | 98.5115 | 98.4238 |
| Transformer residual | 100.0000 | 100.0000 |
| Mamba residual | 99.9416 | 100.0000 |
| Three-expert residual bank | **100.0000** | **100.0000** |
| Frozen DINOv2 | 14.1323 | 9.4571 |
| Fixed equal-block concat | 95.8582 | 96.4974 |

The fixed concat is `-4.1418 mAP` below the stronger source. The two-source
hard Oracle remains `100/100`, so Oracle gain is zero. Unique AP wins are
residual-bank/DINO=`570/0`.

## Fold receipts

| Held-out fold | Queries | Residual bank mAP | DINO mAP | Concat mAP |
|---:|---:|---:|---:|---:|
| 0 | 190 | 100.0000 | 19.6346 | 99.1974 |
| 1 | 179 | 100.0000 | 8.9340 | 94.8485 |
| 2 | 202 | 100.0000 | 13.5632 | 93.6120 |

All distances were computed within the held-out fold. The reported total is a
query-weighted aggregation of per-fold AP and Rank-1; embeddings from different
fold checkpoints were never placed in one distance matrix.

## Why the intended non-saturation claim failed

The exact Signal and Phase-B embeddings were excluded from the qualification
metric, and the three expert adapters were trained without their held-out fold.
However, every expert still consumes the frozen Signal token field loaded from
`Signalbest.pth`, which was trained on all 141 fit identities. Consequently,
adapter training is identity-OOF but the complete feature path is not.

The 100 mAP residual result is therefore evidence of fit-identity saturation,
not a deployable gain and not identity-unseen generalization. The independent
integrity audit found no self-normalized metric or fake ground truth.

## Engineering integrity

- Three folds and 571 real-GT queries completed.
- The three fold checkpoint SHA-256 values matched their source receipts.
- DINOv2 used the same fixed 252×126 input, final CLS + patch mean embedding,
  and strict state load as V10; only `mask_token` was removed.
- Frozen expert/DINO state hashes were unchanged.
- `optimizer_steps=0`, `training_executed=false`, `dev_access_count=0`, and
  `official_test_access_count=0`.
- Peak CUDA allocated/reserved memory was 2820.82/3878 MiB.

## Claim boundary and action

The result supports only this narrow conclusion: the fixed V11 representation
does not qualify DINOv2 as a complementary foundation source. It does not show
that DINOv2 is generally unsuitable for RGBNT201 ReID.

V11 is sealed. Do not scan DINO modality subsets, input resolution,
intermediate blocks, token pooling, fusion weights or training heads. Do not
implement V11-Q1/Q2, access dev or official test, run ablations, or add seeds.
Any successor must be a new preregistered hypothesis whose entire measured
feature path is identity-unseen and whose train-only metric is demonstrably
non-saturated.

## Evidence

- Raw result: `evidence/trifusion_v11_dinov2_oof_residual_complement_seed42.json`
- Provenance wrapper: `evidence/trifusion_v11_dinov2_oof_residual_complement_seed42_provenance.json`
- Independent claim gate: `.aris/traces/result-to-claim/2026-09-02_run06/`
- Independent integrity audit: `EXPERIMENT_AUDIT_V11_Q0.md`
