# TriFusion-ReID workspace

## Experiment environment

- Runtime: local WSL2, Ubuntu 20.04
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB
- Conda: `/root/miniconda3/bin/conda`; project environment `tri_reid`;
  upstream PEFT-BoA comparator environment `peft_boa`; upstream MFRNet
  checkpoint comparator environment `mfrnet`
- Code directory: `/root/mmreid-trifusion/TriFusion-ReID`
- Dataset root: `/root/mmreid-trifusion/data`
- Artifact root: `/root/mmreid-trifusion/artifacts`
- W&B: false

## Research provenance

- Implementation base: official DeMo repository, AAAI 2025, MIT licensed.
- Measurable checkpoint baseline: official MDReID RGBNT201 checkpoint; local parity is required before comparison.
- Open-training-code comparator: PEFT-BoA commit `d2b198b`; use only the
  isolated `peft_boa` environment and the fixed-epoch protocol.
- Supplemental checkpoint comparator: MFRNet commit `ec54a13`; use the
  isolated `mfrnet` environment and label its released best as test-selected.
- Preserve the official baseline path and metrics; put new architecture behind explicit config flags.
- Do not report SOTA unless the same dataset split, input resources, inference protocol, and metrics have been reproduced and audited.

## Local constraints

- Check free GPU memory before every training launch; launch only when used
  memory is strictly below 500 MiB, and fail closed otherwise.
- The 8 GB GPU requires AMP and conservative batches. Gradient accumulation and activation checkpointing are conditional: accumulation must not be used to pretend that batch-hard metric losses saw a larger identity batch. The DeMo calibration therefore uses B32/K4 (8 identities) without accumulation.
- Start model selection and training with RGBNT201. Vehicle-dataset download and integrity audits may run in parallel, but vehicle model experiments remain gated on RGBNT201 data and metric parity.
- Every proposed paper contribution needs an isolated ablation and a no-extra-capacity control.
- Mamba CUDA extensions are built from the pinned official sources for SM89; do not replace them with incompatible manylinux wheels on Ubuntu 20.04.
