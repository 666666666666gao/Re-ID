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
- DeMo implementation base: synthetic and real-data training gates, the
  excluded TB128 paging-pressure run, the accepted TB64 live gate, and a
  bounded post-launch provenance observation of the current process. The
  protocol audit additionally binds the source-level fact that released
  `DeMobest.pth` is selected on test mAP.

The `demo_rgbnt201_seed42_b32k4_tb64_gate2_20260831.json` snapshot is only an
interim systems gate. It contains two fully completed three-mode evaluation
epochs (and a later training-epoch marker captured while evaluation was still
in progress); it is not a final 50-epoch result and must not support an accuracy
or SOTA claim.

The current R012 process started before commit/dirty and GPU-occupancy guards
were added to the launcher. Its live provenance receipt therefore sets
`launch_attestation=false`: it verifies the exact command, effective overrides,
and clean frozen DeMo checkout at observation time without pretending the later
guards ran at launch.

At epoch 10, `DeMo_10.pth` was written before that epoch's test evaluation. Its
receipt binds SHA-256 `b2ab79f056d73d6b827c52fd27ec0607aeae1a10cd756db5c0cc62f3ab4631c0`
and the train/save/eval ordering. The accompanying epoch-10 summary contains 30
complete evaluation records and both checkpoint hashes. It deliberately has
`valid=false`: epoch-7 and epoch-9 `ori` evaluation latencies were 329.741s and
327.717s, exceeding the pre-registered 300s systems limit. Fatal/nonfinite,
record-completeness and uniqueness checks still pass. The threshold was not
raised after seeing the breach.

The MDReID metric JSON predates the strengthened audit-gate fields. Its numeric
checkpoint result, exact official commit and referenced dataset-audit bytes are
retained as valid historical evidence; the patched driver must be rerun after
R012 releases the GPU before claiming code-to-artifact parity for that revision.

Absolute paths inside JSON records identify the WSL2 machine on which the
evidence was produced. File hashes, source commits and protocol fields are the
portable provenance anchors.
