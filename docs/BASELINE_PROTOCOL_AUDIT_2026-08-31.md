# DeMo RGBNT201 checkpoint-selection protocol audit

**Status:** verified against the clean official DeMo checkout at commit
`b4f323a430b32e3a1637c3e7acb25868cb52e9cd`.

## Direct source evidence

| Fact | Upstream source | Consequence |
|---|---|---|
| Both `query_dir` and `gallery_dir` are RGBNT201 `test` | `data/datasets/RGBNT201.py:27-28` | Retrieval metrics are test-set metrics, not development metrics. |
| `val_loader` concatenates `dataset.query + dataset.gallery` | `data/datasets/make_dataloader.py:249-259` | The loader called “val” is the official test query/gallery population. |
| Training evaluates whenever `epoch % EVAL_PERIOD == 0` | `engine/processor.py:120-131` | The released configuration reads test metrics every epoch. |
| `DeMobest.pth` is overwritten when joint test mAP improves | `engine/processor.py:132-143` | The “best” checkpoint is selected on test mAP. |
| Periodic `DeMo_<epoch>.pth` is saved before that epoch's evaluation | `engine/processor.py:111-120` | A pre-registered fixed-epoch checkpoint is not selected by the metric subsequently observed that epoch. |

The source hashes and machine-readable findings are frozen in
`evidence/demo_rgbnt201_protocol_audit_20260831.json`.

## Interpretation

This behavior reproduces the released implementation but is not a leakage-free
model-selection protocol. The current R012 run is therefore an
**official-protocol implementation calibration**. Its `DeMobest.pth` may be
reported only with an explicit `test-selected` label; it cannot establish that
our model-selection procedure is clean.

The optimization trajectory itself is not changed by evaluation: evaluation is
under `torch.no_grad()`, and periodic checkpoints are written before the
evaluation block. Because R012's 50-epoch schedule and seed were frozen before
launch, `DeMo_50.pth` will be the primary fixed-epoch matched baseline. We will
report it separately from the official-protocol test-best checkpoint.

## Binding reporting policy

1. **Official reproduction column:** report the best joint test result and mark
   it `test-selected (released protocol)`.
2. **Matched fair baseline column:** report the pre-registered `DeMo_50.pth`,
   regardless of whether an earlier test epoch is better.
3. **TriFusion selection:** use only the frozen 141-fit/30-dev protocol for
   architecture and checkpoint decisions, then retrain the promoted
   configuration on all 171 training identities and evaluate the official test
   once.
4. **No SOTA claim from R012:** neither an interim epoch nor its test-best
   checkpoint is sufficient for a SOTA claim. Same-resource public comparisons,
   final test evaluation, ablations and multi-seed evidence remain mandatory.

This policy is stricter than merely copying the released training loop and is
the protocol used for all paper claims.
