# MFRNet RGBNT201 checkpoint reproduction specification

**Status:** proposed public seam; no runner or runner test is authorized until
the user replies exactly `接缝同意`.

## Purpose

Reproduce the released MFRNet RGBNT201 checkpoint without modifying the pinned
upstream checkout, silently changing sparse-MoE routing semantics, or promoting
an upstream test-selected number into a clean model-selection result.

The proposed public boundary is:

```text
tools/run_mfrnet_checkpoint_eval.py \
  --mode preflight|official128 \
  --output-dir DIR
```

Only behavior observable through this CLI and its emitted receipt belongs to
the seam. Private subprocess, parsing and monitoring helpers are not test
interfaces.

## Immutable inputs

| Input | Required value |
|---|---|
| Source | `/root/mmreid-trifusion/baselines/MFRNet`, commit `ec54a1302321cda4b5fad9ca1c0878dabf0b46b6`, clean worktree |
| Python | `/root/miniconda3/envs/mfrnet/bin/python`, with `PYTHONNOUSERSITE=1` |
| Checkpoint | `/root/mmreid-trifusion/checkpoints/MFRNet/RGBNT201_MFRNetbest.pth`, SHA-256 `f0c2df33f3901738051a917e728c73d9b494113e1e327361bc8f1acf4711126e` |
| CLIP | `/root/mmreid-trifusion/pretrained/ViT-B-16.pt`, SHA-256 `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f` |
| Dataset | `/root/mmreid-trifusion/data/RGBNT201`, bound to the versioned dataset audit |
| Config | `configs/RGBNT201/MFRNet.yml`, SHA-256 `fecef5e40461930b84f4313aca86f33617be25c37fa3459c39367e6b18caa43e` |
| Test entry point | upstream `test_net.py`, SHA-256 `d0136fbe91dd7d7e9b4044637798ed0e26db44038b290a6beffbd3cefd1b77b3` |

The runner may override only machine-local path, worker, output-directory and
visible-device fields. It must reject any drift in the following scientific
fields before importing the model:

- RGBNT201 complete `test` list is concatenated as 836 query plus 836 gallery;
- `INPUT.SIZE_TEST=[256,128]`;
- `TEST.IMS_PER_BATCH=128`;
- `TEST.RE_RANKING='no'`;
- `TEST.NECK_FEAT='before'`;
- `TEST.FEAT_NORM='yes'`;
- `TEST.MISS='nothing'`;
- inference `return_pattern=3`;
- evaluator removes same-identity/same-camera gallery entries.

## Why `official128` cannot silently microbatch

MFRNet enables a Tutel cosine top-1 MoE with `capacity_factor=1.0` and
batch-prioritized routing. Tutel flattens the active batch/token dimensions,
computes capacity from the current sample count, and ranks tokens against all
other tokens in that invocation. Therefore changing batch partition can change
which over-capacity tokens are dispatched.

The deterministic CPU worked example in
`evidence/mfrnet_eval_batch_semantics_audit_20260831.json` proves the mechanism:
four tokens routed together retain global indices `[0,1]`, while the same four
scores split 2+2 retain `[0,2]`. This does not prove that the released MFRNet
checkpoint overflows an expert or changes RGBNT201 metrics; it proves that
batch-size invariance cannot be assumed.

Consequently:

- `official128` must keep the released batch size of 128;
- an OOM is recorded as local-hardware infeasibility, not retried at a lower
  batch under the same label;
- a future lower-batch diagnostic must use a separate non-comparable label and
  cannot supply the primary parity number;
- changing `CAP_FACTOR`, routing order, expert count or overflow behavior is a
  model/protocol change and is forbidden in parity mode.

## Mode contracts

### `preflight`

This mode performs no CUDA import or model construction. It must:

1. verify all immutable hashes and the clean source commit;
2. verify the exact `mfrnet` package lock with user-site imports disabled;
3. verify dataset/pretrain/checkpoint existence, size and hashes;
4. resolve the upstream command without running it;
5. query `nvidia-smi` and set `launch_allowed=true` only when
   `memory.used < 500 MiB`;
6. atomically emit a receipt whose status is `READY` or `BLOCKED`, never a
   metric result.

### `official128`

This mode first repeats `preflight` and fails closed unless it is `READY`. It
then launches exactly one upstream evaluation with:

```text
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
/root/miniconda3/envs/mfrnet/bin/python test_net.py \
  --config_file configs/RGBNT201/MFRNet.yml \
  --model_path /root/mmreid-trifusion/checkpoints/MFRNet/RGBNT201_MFRNetbest.pth \
  MODEL.DEVICE_ID 0 \
  MODEL.PRETRAIN_PATH_T /root/mmreid-trifusion/pretrained/ViT-B-16.pt \
  DATASETS.ROOT_DIR /root/mmreid-trifusion/data \
  DATALOADER.NUM_WORKERS 0 \
  OUTPUT_DIR DIR/upstream
```

The runner must preserve combined stdout/stderr and the upstream
`test_log.txt`, monitor peak GPU memory, and atomically write `receipt.json`.
It must never patch or copy edited code into the upstream checkout.

## Receipt and result rules

The receipt binds source/config/runtime/data/pretrain/checkpoint/runner hashes,
the resolved command, environment isolation, preflight GPU state, exit status,
peak GPU memory, log hashes, batch count, query/gallery counts, and parsed
mAP/Rank-1/Rank-5/Rank-10.

Possible terminal states are:

- `PASS`: the complete official-B128 evaluation exited normally and emitted
  all four finite metrics;
- `PARITY_MISMATCH`: evaluation completed but its rounded mAP or Rank-1 differs
  from the released 80.7 / 83.6;
- `CUDA_INCOMPATIBLE`, `OOM`, `DRIVER_RESET`, or `FAILED`: no metric claim;
- `BLOCKED`: preflight GPU memory was not strictly below 500 MiB.

Even a `PASS` result is reported as **local parity of a released,
test-selected checkpoint**. It does not become TriFusion's clean
model-selection baseline and does not support a SOTA claim by itself.

## Proposed TDD observations

After seam acceptance, vertical slices may test only these CLI-visible facts:

1. `preflight` rejects a changed source/config/checkpoint hash and emits no
   launch command as executed;
2. exactly 500 MiB is blocked while 499 MiB is launch-eligible;
3. the resolved `official128` plan contains B128/no-rerank/normalized complete
   modality evaluation and rejects a lower batch override;
4. a synthetic upstream log is classified into the documented terminal states
   and atomically bound by hash;
5. an integration smoke with a fake executable proves argv/environment/log
   preservation without importing the MFRNet model.

No private helper or real-model output value is a unit-test seam.
