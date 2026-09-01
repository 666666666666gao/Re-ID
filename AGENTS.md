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
  80.3/85.2 is an upstream fixed-path log only and must not be called locally
  reproduced.
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
