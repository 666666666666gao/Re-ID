# RGBNT201 baseline checkpoint-selection protocol audit

**Status:** verified against clean official checkouts for DeMo commit
`b4f323a430b32e3a1637c3e7acb25868cb52e9cd`, PEFT-BoA commit
`d2b198be634ac4f9f5744eebf6e0a6604e490deb`, and Signal commit
`cd1b0a672d1fe642e7608731cb4899a19dda7d51`, plus MFRNet commit
`ec54a1302321cda4b5fad9ca1c0878dabf0b46b6`.

## DeMo

### Direct source evidence

| Fact | Upstream source | Consequence |
|---|---|---|
| Both `query_dir` and `gallery_dir` are RGBNT201 `test` | `data/datasets/RGBNT201.py:27-28` | Retrieval metrics are test-set metrics, not development metrics. |
| `val_loader` concatenates `dataset.query + dataset.gallery` | `data/datasets/make_dataloader.py:249-259` | The loader called “val” is the official test query/gallery population. |
| Training evaluates whenever `epoch % EVAL_PERIOD == 0` | `engine/processor.py:120-131` | The released configuration reads test metrics every epoch. |
| `DeMobest.pth` is overwritten when joint test mAP improves | `engine/processor.py:132-143` | The “best” checkpoint is selected on test mAP. |
| Periodic `DeMo_<epoch>.pth` is saved before that epoch's evaluation | `engine/processor.py:111-120` | A pre-registered fixed-epoch checkpoint is not selected by the metric subsequently observed that epoch. |

The source hashes and machine-readable findings are frozen in
`evidence/demo_rgbnt201_protocol_audit_20260831.json`.

### Interpretation

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

This ordering has now been observed at epoch 10: `DeMo_10.pth` was closed at
04:42:09.475 local time and the epoch-10 test evaluation began at 04:42:09.479.
The checkpoint receipt records its byte size and SHA-256 independently of the
subsequent test metrics.

## PEFT-BoA

### Direct source and released-log evidence

| Fact | Upstream evidence | Consequence |
|---|---|---|
| Published config resolves to B64/K4, AdamW, seed 1111, 120 epochs and `EVAL_PERIOD=1` | `configs/RGBNT201/train_log.txt:95-118` | The released trajectory reads official-test metrics after every epoch. |
| Training evaluates at every configured period | `engine/processor.py:112-153` | With `EVAL_PERIOD=1`, all 120 epochs consume the official test. |
| `best.pth` is overwritten when test mAP improves | `engine/processor.py:154-160` | The released best is test-selected, not a fixed endpoint. |
| Epoch 80 is 82.7 / 86.1 / 92.3 / 94.7 | `configs/RGBNT201/train_log.txt:1661-1670` | This exactly matches the paper's main RGBNT201 row. |
| Epoch 120 is 82.2 / 85.8 / 91.5 / 93.5 while logged best remains 82.7 / 86.1 | `configs/RGBNT201/train_log.txt:2261-2269` | The paper number and fixed 120-epoch endpoint are different results. |

The released log is 166,691 bytes with SHA-256
`e2443e4fa14c250f055271ab22a3db0181c1095a755d1ddd6005c701a3139f78`.
The machine-readable interpretation is frozen in
`evidence/peft_boa_protocol_audit_20260831.json`.

### Interpretation

PEFT-BoA 82.7 / 86.1 must be labeled
`test-selected epoch80 (released protocol)`. It is valid as a paper-reported
calibration target but not as the clean fixed endpoint. The primary local fair
reproduction is pre-registered as epoch120, evaluated once only after
`BoA_120_fixed.pth` is durable. The upstream log's corresponding reference is
82.2 / 85.8; neither log row is a local reproduction.

The full implementation contract is in
`docs/PEFT_BOA_REPRODUCTION_SPEC.md`. No released PEFT checkpoint exists, so a
measured comparison still requires a from-scratch run.

## Signal

### Direct source and released-log evidence

| Fact | Upstream evidence | Consequence |
|---|---|---|
| `query_dir` and `gallery_dir` both point to the complete RGBNT201 `test` list | `data/datasets/RGBNT201.py:26-34` | The loader called validation is official-test evaluation. |
| Released B64/K8 configuration uses 50 epochs, seed 1234, no reranking and `EVAL_PERIOD=1` | `configs/RGBNT201/Signal.yml`; `config/defaults.py:140-147` | Training reads official-test metrics after every epoch. |
| `Signalbest.pth` is overwritten when official-test mAP improves | `engine/processor.py:172-187` | The upstream best checkpoint is test-selected. |
| Periodic `Signal_50.pth` is written before epoch-50 evaluation | `engine/processor.py:162-172`; checkpoint period 50 | This generated checkpoint is a fixed endpoint rather than a test-best selection. |
| Released `test.py` hard-codes an author path named `signal_50.pth` | `test.py:42-52` | The public test entry targets a fixed-name file, but does not expose its bytes or hash. |
| Released test log reports 80.3 / 85.2 / 91.4 / 93.7 | `test_RNT201/test_log.txt:146-152` | These are upstream log values, not local checkpoint reproduction. |

The released log is 3,390 bytes with SHA-256
`b200abf85f3f3c39dd315fc95c5858dbb8ca18d7a040aa6d9bb4e4f88793c83e`.
The public Baidu share resolved but the RGBNT201 checkpoint was not downloaded,
so the log-to-checkpoint relationship remains unverified. The CPU loader probe
does not close that gap. The full machine-readable boundary is frozen in
`evidence/signal_source_protocol_audit_20260831.json`.

### Interpretation

Signal 80.3 / 85.2 may be cited only as an upstream fixed-path test-log result.
It must not be labeled locally reproduced until the exact checkpoint bytes are
hashed, strictly loaded and evaluated by the audited protocol. If future work
reports `Signalbest.pth`, that result must instead carry a `test-selected`
label. The periodic epoch-50 artifact and test-selected best belong in separate
columns even if their rounded metrics happen to match.

## MFRNet

### Direct source, checkpoint and loader evidence

| Fact | Upstream or local evidence | Consequence |
|---|---|---|
| Released RGBNT201 config is B64/K8, seed 1111, 45 epochs, no reranking and `EVAL_PERIOD=1` | `configs/RGBNT201/MFRNet.yml`; `config/defaults.py:138-146` | Training reads official-test metrics after every epoch. |
| Query and gallery both use the complete RGBNT201 `test` list | `data/datasets/RGBNT201.py:26-34`; real loader probe | Its `val_loader` is official-test evaluation. |
| `MFRNetbest.pth` is overwritten when official-test mAP improves | `engine/processor.py:119-137` | The released best checkpoint is test-selected. |
| Official checkpoint has 407,297,967 bytes and SHA-256 `f0c2df33…711126e` | Direct official Google Drive download | The exact local artifact is durable and identifiable. |
| The isolated upstream model and checkpoint match 297/297 tensors under `strict=True` | CPU construction with Tutel v0.3.2 and verified CLIP | Checkpoint architecture compatibility is proven without claiming metrics. |

The source/environment/checkpoint receipt is
`evidence/mfrnet_rgbnt201_checkpoint_audit_20260831.json`. It also records that
the repository has no explicit license and its released requirements contain
author-local wheel URLs.

### Interpretation

MFRNet 80.7 / 83.6 is still an upstream test-selected result. Once the local GPU
gate permits evaluation, the downloaded checkpoint can be used for released-
protocol parity, but that local result must retain the `test-selected released
checkpoint` label. It is not the clean model-selection comparator and cannot
replace the frozen dev-selection/final-once policy for TriFusion.

The released evaluation batch of 128 is also protocol-relevant rather than a
free memory knob. MFRNet's Tutel top-1 MoE uses capacity factor 1.0 and
batch-prioritized routing. A deterministic source-runtime example shows that
processing the same four routing scores in one batch versus two batches changes
the accepted token set from `[0,1]` to `[0,2]`. This proves that the mechanism
is not generally batch-partition invariant, although it does not prove an
RGBNT201 metric change for this checkpoint. Local parity must therefore retain
B128; an 8 GB OOM is an infeasibility result, not permission to silently lower
the batch. The receipt is
`evidence/mfrnet_eval_batch_semantics_audit_20260831.json`.

## Binding reporting policy

1. **DeMo official reproduction column:** report the best joint test result and
   mark it `test-selected (released protocol)`.
2. **PEFT paper-reference column:** label 82.7 / 86.1 as
   `test-selected epoch80 (released log)`, never as fixed120 or locally
   reproduced.
3. **Signal reference column:** label 80.3 / 85.2 as an upstream fixed-path
   test-log value until the exact weight is hashed and evaluated locally;
   always separate any `Signalbest.pth` test-selected result. Signal is the
   selected high-metric MIT baseline, so the fair primary row is a locally
   trained seed-1234 fixed `Signal_50.pth`, not the unavailable public-link
   bytes or a test-selected best.
4. **MFRNet checkpoint column:** label the downloaded official checkpoint and
   any parity result as `test-selected (released protocol)`; strict loading is
   not metric reproduction.
5. **Matched fair baseline column:** report the pre-registered `DeMo_50.pth`,
   locally trained `Signal_50.pth`, and locally trained
   `BoA_120_fixed.pth`, regardless of whether an earlier test epoch is better.
6. **TriFusion selection:** use only the frozen 141-fit/30-dev protocol for
   architecture and checkpoint decisions, then retrain the promoted
   configuration on all 171 training identities and evaluate the official test
   once.
7. **No SOTA claim from a baseline calibration:** neither an interim epoch nor
   a test-best checkpoint is sufficient for a SOTA claim. Same-resource public
   comparisons, final test evaluation, ablations and multi-seed evidence remain
   mandatory.

This policy is stricter than merely copying the released training loop and is
the protocol used for all paper claims. Repository-level licensing and source
reuse boundaries are separately binding in
`docs/BASELINE_SELECTION_AND_LICENSE_AUDIT_2026-08-31.md`.
