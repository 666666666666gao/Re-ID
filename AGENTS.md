# TriFusion-ReID workspace

## Experiment environment

- Runtime: remote Linux GPU server; local WSL2 is SSH transport only and must
  not store the project or run training.
- GPU: one NVIDIA GeForce RTX 3090, 24 GB
- Conda: `/root/miniconda3/bin/conda`; project environment `tri_reid` and
  comparator environment `signal`. The latter was receipted on 2026-09-01 at
  Python 3.10.13, PyTorch 2.1.1+cu118, and CUDA 11.8. The remote host does not
  contain separate `peft_boa` or `mfrnet` environments; either must be created
  and receipted before use.
- Code directory: `/root/autodl-tmp/trifusion-v2/TriFusion-ReID`
- Dataset root: `/root/autodl-tmp/trifusion-v2/data`
- Artifact root: `/root/autodl-tmp/trifusion-v2/artifacts`
- W&B: false

## Research provenance

- Implementation base: official DeMo repository, AAAI 2025, MIT licensed.
- High-metric licensed baseline: Signal commit `cd1b0a6`, MIT licensed;
  the local 141-fit/30-dev floor is 58.0109/57.4545 from the exact 3072D
  path, while 80.3/85.2 is an upstream official-test log only and must not be
  called locally reproduced.
- Measurable checkpoint anchor: official MDReID RGBNT201 checkpoint. Its pinned
  repository has no
  repository-level license, so keep it isolated and do not copy its code.
- Source-visible training comparator: PEFT-BoA commit `d2b198b`; its pinned
  repository has no repository-level license. Use only the isolated `peft_boa`
  environment and fixed-epoch protocol; do not call it open source or copy its
  code into TriFusion.
- Supplemental checkpoint comparator: MFRNet commit `ec54a13`; use the
  isolated `mfrnet` environment and label its released best as test-selected.
  Its pinned repository likewise has no repository-level license.
- Preserve the official baseline path and metrics; put new architecture behind explicit config flags.
- The latest user direction supersedes the earlier no-baseline-reproduction
  constraint: first establish a remote-only Signal baseline floor. Keep the
  upstream `80.3/85.2` label until a local checkpoint and same-protocol result
  have actually been verified.
- Do not report SOTA unless the same dataset split, input resources, inference protocol, and metrics have been reproduced and audited.

## TriFusion implementation status

- The exact `接缝同意` reply is receipted before core implementation.
- The v2 build runs one complete shared 12-layer CLIP visual trunk and then
  three 768-wide CNN, Transformer, and Mamba semantic-residual experts.
- Reliability-conditioned HFER performs two stage-wise bidirectional exchanges;
  the final quality posterior is refreshed from the collaboratively enhanced
  expert features before identity fusion.
- The real RTX 3090 B32/K4 capacity gate passes: 92,682,578 parameters, AMP
  initial scale 512, zero overflow events, finite gradients for 366/366 trainable
  tensors, and 6,444 MiB peak reserved memory.
- The full CIRC/URGC development and post-freeze configuration chain is defined.
  The CIRC protocol must be re-anchored to the final committed v2 source.
- Capacity and overfit gates are train-only engineering evidence, not retrieval
  metrics, and never support a SOTA claim.
- The exact Signal seed42 B64/K8 run completed 50/50 epochs on the frozen
  141-fit/30-dev protocol. The deterministically reloaded best checkpoint is
  `1f5c200c...66c3` with 58.0109 mAP / 57.4545 Rank-1 and official-test access
  zero. This is the V5 development floor, not an official-test reproduction.
- Signal-preserving V5 readiness passes on the remote RTX3090: exact 825/825
  baseline feature/metric parity, real B32/K4 8-step capacity with 213/213
  trainable gradient tensors and 3,542 MiB peak reserved memory, and a
  100-step fixed-batch loss ratio of 0.02102. Official-test access is zero.
- Signal-preserving V5 completed its only seed-42 60-epoch held-out-dev run.
  Selected epoch 51 gives baseline/fused/CNN mAP 58.0109/58.0168/58.0181;
  fused fails both the best-expert and 65 mAP gates. The frozen Signal state
  stayed unchanged and official-test access stayed zero. Do not launch V5 on
  the official test and do not start ablations.
- The V5 read-only diagnostic shows a 0.02747 fused-suffix/baseline norm ratio,
  distance correlation 1.0 and 99.9879% Top-10 overlap versus baseline. The
  next main version must materially change retrieval geometry while preserving
  the exact Signal output; do not scan hyperparameters or residual multipliers.
- V6 is the single diagnostic-driven main-only correction: it removes the V5
  learned residual scale, deterministically matches the routed residual-bank
  norm to the exact Signal norm per sample, and trains/routes from residual-only
  identity embeddings so the frozen baseline cannot satisfy expert objectives.
  It is not an ablation or multiplier scan. It must pass preflight, capacity and
  overfit before one seed-42 held-out-dev run; official test remains forbidden.

## Remote experiment constraints

- Check the remote RTX 3090 before every launch. The `rtx3090_b32k4` gate
  requires at least 22,000 MiB free before a new training process starts.
- Use real B32/K4 without gradient accumulation so batch-hard losses see eight
  identities. AMP and activation checkpointing are required.
- Run only seed 42. Signal baseline work is now explicitly authorized, but only
  on the remote GPU and without hyperparameter or epoch selection on the
  official test. Preserve its full 3072D direct-plus-SIM retrieval feature,
  camera SIE and baseline-only output; do not call the 1536D projected-CLS
  anchor a Signal reproduction.
- Any fused successor must emit baseline-only and fused embeddings from the
  same checkpoint. Fused is promoted only if it is not worse than baseline on
  the frozen development protocol; otherwise fused is rejected and no
  fusion-gain claim is allowed. Do not add runtime fallback logic unless a
  later explicit requirement and test justify it.
- Start with RGBNT201. Do not run ablations until the frozen same-protocol main
  target (85.3 mAP / 87.9 Rank-1) has been exceeded.
- Mamba CUDA extensions are source-locked to the receipted SM86 builds; do not
  replace them with unverified wheels.
