# TriFusion V10-Q0 frozen DINOv2 fit-only qualification

Date: 2026-09-02

Hardware: one remote NVIDIA RTX 3090 24 GB

Scope: 141-fit cross-camera identities only; 21 identities / 571 queries

Training / dev / official access: false / 0 / 0

## Outcome

The probe completed, but the preregistered qualification gate failed. V10 is
not authorized for model implementation, capacity testing, training or dev
access.

| Frozen representation | mAP | Rank-1 |
|---|---:|---:|
| V8 Phase-B | **100.0000** | **100.0000** |
| DINOv2 ViT-B/14 | 7.6284 | 6.1296 |
| fixed equal-block `[L2(Phase-B), L2(DINO)]` | 92.2120 | 95.9720 |

The fixed concat loses `7.7880 mAP` relative to Phase-B. The Phase-B/DINO
query-wise hard Oracle is still `100/100`, for zero gain over the strongest
fixed representation. Unique AP wins are Phase-B/DINO=`571/0`.

All scientific gates fail:

- required concat gain: `≥1.0 mAP`; observed `-7.7880`;
- required Oracle gain: `≥2.0 mAP`; observed `0.0`;
- required Phase-B unique AP wins: observed `571`;
- required DINO unique AP wins: observed `0`.

## Engineering integrity

- DINOv2 checkpoint SHA-256:
  `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`.
- The feature model strict-loads after removing exactly the pretraining-only
  `mask_token`; no `strict=False` path exists.
- Existing `256×128` tensors are deterministically converted to ImageNet
  normalization and `252×126`; the observed DINO token shape is `163×768`
  (one CLS plus an 18×9 patch grid).
- Phase-A, Router and DINO state hashes are identical before and after the
  probe. Optimizer steps are zero.
- Runtime is 27.79 s; peak allocated/reserved VRAM is 2802.84/3872 MiB.
- `status=PASS` in the JSON means only that the probe program completed.
  Scientific qualification is represented by `qualification_gate.passed=false`.

Before the terminal run, one command used a nonexistent checkpoint path and
stopped before model/data access. A second engineering attempt exposed the
loader's public `RGB/NI/TI` key contract and stopped on the first fit batch
before DINO features, metrics or a result file existed. The only result-bearing
run is the fixed `bfec5eb` execution above.

## Claim boundary

Independent result-to-claim is `no/high`. The evidence supports only that this
frozen DINO representation supplies no usable complement under this saturated
fit-only protocol and that the one fixed concat is harmful. It does not support
a general claim that DINOv2 is unsuitable for RGBNT ReID.

Independent audit is `WARN / warn / FAIL_TO_QUALIFY—STOP_V10_Q0`. GT/protocol,
normalization, executed paths, scope and evaluation-type checks pass. WARN is
limited to the result JSON being untracked at audit time and the Phase-B/DINO
binary weights remaining remote-only, so their bytes cannot be rehashed from a
fresh local clone.

## Decision

Seal V10-Q0. Do not scan DINO modality subsets, resolution, intermediate
blocks, token pooling, concat weights, training heads or dev results. Any later
use of DINO must be a separately preregistered hypothesis with a non-saturated,
identity-isolated train-only qualification protocol; it cannot be called a V10
continuation.

Evidence: `evidence/trifusion_v10_dinov2_fit_qualification_seed42.json`.
