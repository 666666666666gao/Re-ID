# TriFusion-ReID workspace

## Experiment environment

- Runtime: local WSL2, Ubuntu 20.04
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB
- Conda: `/root/miniconda3/bin/conda`; project environment `tri_reid`;
  upstream PEFT-BoA comparator environment `peft_boa`; upstream MFRNet
  checkpoint comparator environment `mfrnet`; upstream Signal environment
  `signal`
- Code directory: `/root/mmreid-trifusion/TriFusion-ReID`
- Dataset root: `/root/mmreid-trifusion/data`
- Artifact root: `/root/mmreid-trifusion/artifacts`
- W&B: false

## Research provenance

- Implementation base: official DeMo repository, AAAI 2025, MIT licensed.
- High-metric licensed baseline: Signal commit `cd1b0a6`, MIT licensed;
  80.3/85.2 is currently an upstream fixed-path log only, so locally retrain a
  fixed epoch-50 endpoint before calling it reproduced.
- Measurable checkpoint anchor: official MDReID RGBNT201 checkpoint; local
  parity is required before comparison. Its pinned repository has no
  repository-level license, so keep it isolated and do not copy its code.
- Source-visible training comparator: PEFT-BoA commit `d2b198b`; its pinned
  repository has no repository-level license. Use only the isolated `peft_boa`
  environment and fixed-epoch protocol; do not call it open source or copy its
  code into TriFusion.
- Supplemental checkpoint comparator: MFRNet commit `ec54a13`; use the
  isolated `mfrnet` environment and label its released best as test-selected.
  Its pinned repository likewise has no repository-level license.
- Preserve the official baseline path and metrics; put new architecture behind explicit config flags.
- Do not report SOTA unless the same dataset split, input resources, inference protocol, and metrics have been reproduced and audited.

## TriFusion implementation status

- The exact `接缝同意` reply is receipted before core implementation.
- All six frozen public seams, the official evaluator, HFER/CIRC/URGC, optional
  RDPT, named criterion and real CLIP builder exist under `modeling/trifusion`.
- The full default build has 12 Transformer, 9 CNN and 9 Mamba blocks,
  95,893,482 parameters, one shared patch projection, and one stage-1 posterior
  object reused at both relays and fusion.
- The hash-bound test run is 39/39 passing. This is implementation evidence,
  not retrieval-metric or SOTA evidence; real TriFusion CUDA/overfit/training is
  still pending the GPU launch gate.
- The accepted PEFT-BoA fixed120 and MFRNet official128 runners are implemented;
  their combined CLI contract suite passes 6/6 and the full project suite is
  45/45. Both real preflights remain blocked by the same `<500 MiB` launch gate,
  so neither produced a new local metric.
- The train-only TriFusion loader, pre-CIRC criterion and public
  `preflight|capacity|overfit|dev` runner are implemented. The current project
  suite is 51/51; its latest real capacity preflight was blocked before model
  construction at 1,035 MiB used GPU memory. No TriFusion metric exists yet.

## Local constraints

- Check free GPU memory before every training launch; launch only when used
  memory is strictly below 500 MiB, and fail closed otherwise.
- The 8 GB GPU requires AMP and conservative batches. Gradient accumulation and activation checkpointing are conditional: accumulation must not be used to pretend that batch-hard metric losses saw a larger identity batch. The DeMo calibration therefore uses B32/K4 (8 identities) without accumulation.
- Start model selection and training with RGBNT201. Vehicle-dataset download and integrity audits may run in parallel, but vehicle model experiments remain gated on RGBNT201 data and metric parity.
- Every proposed paper contribution needs an isolated ablation and a no-extra-capacity control.
- Mamba CUDA extensions are built from the pinned official sources for SM89; do not replace them with incompatible manylinux wheels on Ubuntu 20.04.
