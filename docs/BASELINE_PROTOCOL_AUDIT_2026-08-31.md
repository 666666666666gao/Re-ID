# RGBNT201 baseline checkpoint-selection protocol audit

**Status:** verified against clean official checkouts for DeMo commit
`b4f323a430b32e3a1637c3e7acb25868cb52e9cd`, PEFT-BoA commit
`d2b198be634ac4f9f5744eebf6e0a6604e490deb`, and Signal commit
`cd1b0a672d1fe642e7608731cb4899a19dda7d51`.

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

## Binding reporting policy

1. **DeMo official reproduction column:** report the best joint test result and
   mark it `test-selected (released protocol)`.
2. **PEFT paper-reference column:** label 82.7 / 86.1 as
   `test-selected epoch80 (released log)`, never as fixed120 or locally
   reproduced.
3. **Signal reference column:** label 80.3 / 85.2 as an upstream fixed-path
   test-log value until the exact weight is hashed and evaluated locally;
   always separate any `Signalbest.pth` test-selected result.
4. **Matched fair baseline column:** report the pre-registered `DeMo_50.pth`
   and locally trained `BoA_120_fixed.pth`, regardless of whether an earlier
   test epoch is better.
5. **TriFusion selection:** use only the frozen 141-fit/30-dev protocol for
   architecture and checkpoint decisions, then retrain the promoted
   configuration on all 171 training identities and evaluate the official test
   once.
6. **No SOTA claim from a baseline calibration:** neither an interim epoch nor
   a test-best checkpoint is sufficient for a SOTA claim. Same-resource public
   comparisons, final test evaluation, ablations and multi-seed evidence remain
   mandatory.

This policy is stricter than merely copying the released training loop and is
the protocol used for all paper claims.
