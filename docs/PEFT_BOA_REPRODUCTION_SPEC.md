# PEFT-BoA RGBNT201 reproduction specification

**Status:** user accepted the frozen seam with exact reply `接缝同意`; the
runner and CLI contract are implemented and TDD-verified. The real B64/K4
capacity run remains blocked by the `<500 MiB` GPU-idle gate, so fixed120 has
not started and no local PEFT metric exists.

## Purpose and reporting roles

PEFT-BoA is the strongest static CLIP method in the current scan that exposes
training code but no checkpoint. The official paper number is 82.7 mAP / 86.1
Rank-1. The pinned repository log proves this is the epoch-80 test-best result
from a run that evaluates the official test every epoch. The same run's fixed
epoch-120 endpoint is 82.2 / 85.8.

The project therefore freezes two labels:

- `released-test-selected/e80`: published-protocol calibration only;
- `fixed/e120`: primary fair locally reproduced PEFT-BoA baseline.

Neither value is a TriFusion result or SOTA evidence.

## Frozen source and optimization contract

- source: `fffunly/PEFT-BoA` commit
  `d2b198be634ac4f9f5744eebf6e0a6604e490deb`;
- RGBNT201 `train_171`, full test query/gallery, same-ID/same-camera exclusion;
- OpenAI CLIP ViT-B/16 SHA-256
  `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`;
- 256×128, B64/K4, AdamW, frozen CLIP, 120 epochs, seed 1111;
- no reranking, text, mask, TTT, gradient accumulation or official-test early
  stopping;
- upstream-compatible environment is torch 2.1.1+cu118. A torch
  2.5.1+cu121 run is a compatibility variant until it passes an independently
  labeled parity comparison.

The reproducible upstream environment is defined by
`environment/peft_boa_environment.yml` and
`environment/peft_boa_requirements-lock.txt`. The installed `peft_boa`
environment passes `pip check`, CPU imports/tensor math, and a real seed-42
B32/K4 RGBNT201 loader batch under CUDA masking. That proves dependency and
loader compatibility only. It does not construct the CUDA model, establish
B64/K4 capacity, reproduce a metric, or make training ready; those boundaries
are recorded in `evidence/peft_boa_environment_smoke_20260831.json`.

If B64/K4 cannot pass the 8 GiB capacity gate, it is recorded as an exact
reproduction failure. B32/K4 may be run as a hardware-matched diagnostic but
must not inherit the exact-reproduction label.

## Public runner contract

The implemented public boundary is:

```text
tools/run_peft_boa_resumable.py --output-dir DIR --mode capacity|fixed120
```

`capacity` must execute eight real B64/K4 AMP optimization steps without
opening the official test loader. It records peak allocated/reserved VRAM,
finite losses/gradients, trainable-parameter coverage and environment/source
hashes.

`fixed120` must:

1. reject a non-empty output directory unless it contains a valid recovery
   manifest;
2. record an epoch-0 full-state boundary before the first batch;
3. train without iterating the official test loader in epochs 1–119;
4. atomically retain current and previous complete epoch-boundary generations;
5. export `BoA_80_preregistered.pth` and `BoA_120_fixed.pth` immediately after
   their training epochs and before any official-test evaluation;
6. evaluate only `BoA_120_fixed.pth` once in the primary mode, then emit a
   machine-readable fixed-result receipt;
7. make a resumed epoch-120 `post_train` boundary run evaluation only, while a
   durable `complete` boundary performs no further work.

The public state contains model, AdamW, cosine scheduler, AMP scaler,
center-criterion/optimizer state where constructed, Python/NumPy/CPU/CUDA RNG,
epoch and phase. Its run identity binds source/config/runner/runtime hashes,
CLIP and dataset manifests, seed, real batch/K, worker count and device.

## Required observable receipts

- `capacity.json`: no test access, B64/K4, eight steps, finite gradient and
  memory gates;
- `.resume/latest.json`: atomic current/previous full-state generations;
- `fixed_checkpoint_receipt.json`: checkpoint saved before official-test
  access, byte count and SHA-256;
- `fixed_eval.json`: exact mAP/R1/R5/R10, query/gallery counts, evaluator hash,
  reranking=false, test-evaluation count=1;
- `run_summary.json`: source/runtime identity, resume history, fatal/nonfinite
  scan, output hashes and explicit `fixed` versus `test-selected` labels.

The official epoch-80 result may be evaluated locally only as a separately
labeled, preregistered published-protocol calibration after the fixed120 result
is sealed. It never replaces the fixed primary row.

## Decision gates

- capacity failure: do not start the 120-epoch run;
- corrupt/foreign/incomplete recovery state: fail closed, never restart into
  the same output directory;
- fixed120 metric below the upstream fixed log: diagnose environment,
  evaluator and optimization drift; retain the negative result;
- only a locally completed fixed120 result may be used as the measured PEFT
  baseline in the fair comparison table.
