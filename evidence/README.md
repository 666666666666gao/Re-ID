# Versioned evidence bundle

This directory contains byte-for-byte snapshots of the external JSON artifacts
that support the current data, environment and baseline claims. The canonical
runtime copies remain under `/root/mmreid-trifusion/artifacts`; these mirrors
make the evidence travel with the source revision being reviewed.

Verify every snapshot from the repository root with:

```bash
sha256sum -c evidence/SHA256SUMS
```

## Scope

- Dataset integrity: RGBNT201, MSVR310 and RGBNT100 audits plus the frozen
  RGBNT201 train-only development protocol.
- Runtime: conda/PyTorch/CUDA and source-built Mamba forward/backward smoke
  evidence.
- Numeric baseline: strict MDReID checkpoint loading and RGBNT201 evaluator
  parity at 82.0868% mAP / 85.1675% Rank-1.
- Comparator audits: PEFT-BoA and Signal pinned-source, released-protocol,
  environment/loader and claim-boundary receipts. Signal's public log is
  explicitly not treated as a local checkpoint result.
- Claim-gate preregistration: the quantitative effect, clustered-statistics,
  calibration, robustness and SOTA thresholds were hash-frozen before any
  R020+ implementation or method result existed.
- MFRNet checkpoint comparator: direct official weight download, isolated
  Python 3.8/torch 1.12/Tutel environment, real loader and 297/297 strict CPU
  state match, plus a separate batch-routing semantics receipt. No GPU metric
  is implied by these receipts.
- DeMo implementation base: synthetic and real-data training gates, the
  excluded TB128 paging-pressure run, the accepted TB64 live gate, a bounded
  post-launch provenance observation, and the terminal R012 incident receipt.
  The protocol audit additionally binds the source-level fact that released
  `DeMobest.pth` is selected on test mAP.

The `demo_rgbnt201_seed42_b32k4_tb64_gate2_20260831.json` snapshot is only an
interim systems gate. It contains two fully completed three-mode evaluation
epochs (and a later training-epoch marker captured while evaluation was still
in progress); it is not a final 50-epoch result and must not support an accuracy
or SOTA claim.

R012 started before commit/dirty and GPU-occupancy guards were added to the
launcher. Its live provenance receipt therefore sets
`launch_attestation=false`: it verifies the exact command, effective overrides,
and clean frozen DeMo checkout at observation time without pretending the
later guards ran at launch.

At epoch 10, `DeMo_10.pth` was written before that epoch's test evaluation. Its
receipt binds SHA-256 `b2ab79f056d73d6b827c52fd27ec0607aeae1a10cd756db5c0cc62f3ab4631c0`
and the train/save/eval ordering. The accompanying epoch-10 summary contains 30
complete evaluation records and both checkpoint hashes. It deliberately has
`valid=false`: epoch-7 and epoch-9 `ori` evaluation latencies were 329.741s and
327.717s, exceeding the pre-registered 300s systems limit. Fatal/nonfinite,
record-completeness and uniqueness checks still pass. The threshold was not
raised after seeing the breach.

At epoch 20, `DeMo_20.pth` was likewise saved after the training epoch closed
and before that epoch's test evaluation began. Its receipt binds SHA-256
`5d61a4cfc8d1796f6e9dccc0341dee8f9da5fa43b61a85b2e7a45364bc8e7b2e`
and records the three feature-mode metrics. It is a fixed implementation
milestone, not a model selected on test. The pre-registered fair endpoint
remains `DeMo_50.pth`; epoch-17 `DeMobest.pth` is labeled only as
released-protocol calibration.

At epoch 30, `DeMo_30.pth` was again saved after training and before the
epoch's test evaluation. Its receipt binds SHA-256
`d5e375fa2c4bab08f753fc7b0a17b698537db2398250770b130aceac1b274ce5`,
the 30-epoch/90-evaluation fail-closed summary state, and all three feature-mode
metrics. The joint result is 76.3% mAP / 81.5% Rank-1. This remains a fixed
implementation milestone, not a test-selected or SOTA result; the fair
endpoint is still `DeMo_50.pth`.

R012 subsequently completed epoch-31 training and entered `ori` evaluation,
then terminated at `feat.cpu()` with `CUDA error: unknown error`. Seven Windows
`nvlddmkm` event-153 records and WSL DXG failures are temporally correlated
with the exception; they support, but do not uniquely prove, a host/WSL GPU
reset. No epoch-31 evaluation record or fair `DeMo_50.pth` exists, so R012 is
classified `INCOMPLETE`. Its last fixed checkpoint contains only 417 model
tensors and no optimizer, scheduler, scaler, epoch or RNG state; resetting Adam
from that file would not be an exact continuation. The incident receipt binds
the terminal boundary, raw-log hashes and this recoverability audit. The fair
replacement must therefore start from epoch 0 with atomic full-state recovery
and pass the hash-bound epoch-10 exact-tensor parity gate.

The summarizer fail-closed receipt binds the corrected source SHA and freezes
the epoch-12 34/35/36-record live boundary together with seven one-off public
CLI contract probes. Empty, gapped, out-of-range, duplicate and partial logs
all exit nonzero; a complete contiguous epoch exits zero under the diagnostic
latency limit. Under the registered 300-second limit, the complete live
snapshot remains invalid because its latency breach is preserved. No
persistent regression test is claimed before the TDD seam agreement.

The MDReID metric JSON predates the strengthened audit-gate fields. Its numeric
checkpoint result, exact official commit and referenced dataset-audit bytes are
retained as valid historical evidence; the patched driver must be rerun after
the local GPU satisfies the registered idle-memory gate before claiming
code-to-artifact parity for that revision.

The Signal receipt binds the clean official commit, fourteen source/log hashes,
the source B64/K8 real loader shapes and the released 80.3% / 85.2% test log.
It also records the missing checkpoint bytes, hard-coded author paths and
per-epoch official-test selection behavior. Consequently it supports protocol
and portability claims only; it does not support a local metric comparison.

The MFRNet receipt binds the 407,297,967-byte official checkpoint at SHA-256
`f0c2df33…711126e`, the clean source commit, pinned two-phase environment,
source B64/K8 loader and exact 297-tensor model/checkpoint match. It records
that official-test mAP selects the released best and that no GPU forward or
metric evaluation has run; 80.7% / 83.6% remains an upstream value.

The MFRNet batch-semantics receipt binds a deterministic CPU call to the pinned
Tutel v0.3.2 routing implementation. Its four-token counterexample proves that
capacity-factor-1.0 batch-prioritized routing can retain a different token set
after batch splitting. It supports preserving released evaluation B128 and
labeling any lower-batch run non-comparable; it does not claim the real
checkpoint necessarily changes metrics across batches.

The claim-gate preregistration receipt binds `claim_gates_v1`, the byte-identical
timestamped/fixed v1.2 plan and tracker, the absence of the TriFusion core and
training/evaluation entry points, and 50 R020+ rows with zero completed method
results at freeze time. It proves ordering and hashes only; it does not prove
that the thresholds are optimal or that any future method claim passes them.

Absolute paths inside JSON records identify the WSL2 machine on which the
evidence was produced. File hashes, source commits and protocol fields are the
portable provenance anchors.
