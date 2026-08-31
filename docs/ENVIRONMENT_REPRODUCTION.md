# WSL2 environment and baseline reproduction

## Verified local stack

- WSL2 Ubuntu 20.04, glibc 2.31
- NVIDIA GeForce RTX 4060 Laptop GPU, compute capability 8.9
- Python 3.10.14
- PyTorch 2.5.1+cu121 / torchvision 0.20.1+cu121
- causal-conv1d 1.6.0 and mamba-ssm 2.2.6.post3, compiled for SM89
- Transformers 4.45.2

The upstream prebuilt Mamba wheels require a newer glibc than Ubuntu 20.04. The
environment therefore pins the official source commits and applies auditable
SM89-only build patches. Transformers is fixed to 4.45.2 because Mamba 2.2.6's
generation imports are incompatible with Transformers 5.x.

## Recreate the environment

Run from this repository so the relative pip lock resolves correctly:

```bash
cd /root/mmreid-trifusion/TriFusion-ReID
/root/miniconda3/bin/conda env create -f environment.yml
/root/miniconda3/bin/conda activate tri_reid
bash scripts/build_mamba_sm89.sh
```

The build script verifies these upstream commits before applying patches:

- causal-conv1d v1.6.0: `da6dbaa9fd5a919967f14d3fd031da1288ad5025`
- mamba-ssm v2.2.6.post3: `10b5d6358f27966f6a40e4bf0baa17a460688128`

It finishes by running a real CUDA forward/backward smoke and writes
`/root/mmreid-trifusion/artifacts/mamba_cuda_smoke_20260831.json`.

## Reproduce the measurable baseline

```bash
conda activate tri_reid
CUDA_VISIBLE_DEVICES=0 python tools/reproduce_mdreid.py
```

The official MDReID `test_net.py` hard-codes an author checkpoint and calls an
undefined visualization function. Its `engine/processor.py` also contains an
invalid UTF-8 byte under Python 3.10. `tools/reproduce_mdreid.py` leaves the
upstream checkout untouched, strictly loads the official checkpoint, and calls
the repository's own `R1_mAP_eval` implementation directly.

Verified RGBNT201 result, without reranking:

- mAP: 82.0868%
- Rank-1: 85.1675%
- Rank-5: 90.3110%
- Rank-10: 92.5837%

These values reproduce the public 82.1% mAP / 85.2% Rank-1 after rounding. The
full protocol, hashes, runtime and strict-load result are stored in
`/root/mmreid-trifusion/artifacts/mdreid_rgbnt201_eval_20260831.json`.
That metric artifact predates the later fail-closed triplet/camera audit and
clean-commit checks. The underlying audit bytes and official commit were
correct; rerun the patched driver once R012 releases the GPU to bind the new
report schema before treating the current source revision as reproduced.

## Audit the stronger open-training-code comparator

PEFT-BoA reports 82.7% mAP / 86.1% Rank-1 and publishes training code but no
checkpoint or release asset. Its complete official log shows that this is the
test-mAP-selected epoch-80 result; the same run's fixed epoch120 is 82.2% /
85.8%. Its official checkout is pinned, without local modification, at:

```text
/root/mmreid-trifusion/baselines/PEFT-BoA
d2b198be634ac4f9f5744eebf6e0a6604e490deb
```

Create the isolated upstream-compatible runtime and run the source/CPU loader
audit with:

```bash
cd /root/mmreid-trifusion/TriFusion-ReID
/root/miniconda3/bin/conda env create \
  -f environment/peft_boa_environment.yml
/root/miniconda3/envs/peft_boa/bin/python -m pip check
CUDA_VISIBLE_DEVICES='' \
  /root/miniconda3/envs/peft_boa/bin/python tools/audit_peft_boa_source.py
```

The verified receipt binds the clean commit and official remote, six
source-file hashes, and the 350,837,078-byte CLIP archive with SHA-256
`5806e77c…df416f`. It reads the real `train_171` split and deterministically
returns one seed-42 B32/K4 RGB/NIR/TIR batch of shape `32×3×256×128`, 171
training identities and 836 queries. It also records that the published
configuration is frozen-CLIP, B64/K4, seed 1111, 120 epochs, 256×128 and no
reranking.

The dedicated `peft_boa` environment exactly carries the released torch
2.1.1+cu118 stack and a version lock matching its installed `pip freeze`. It
passes dependency resolution, CPU imports/tensor math and the real RGBNT201
loader audit. The project `tri_reid` environment remains separately fixed at
torch 2.5.1+cu121 for the new architecture. The environment receipt is
`evidence/peft_boa_environment_smoke_20260831.json`.

This is deliberately **not** a metric reproduction or a training-ready gate.
The upstream model constructor forces CUDA; the upstream loop evaluates the
official test every epoch, saves `best.pth` by test mAP, and writes model-only
periodic checkpoints. R017A therefore stays waiting until the accepted
fixed-epoch crash-safe wrapper and a real B64/K4 8 GiB forward/backward gate
pass. MDReID remains the strongest locally measured checkpoint baseline.
The source-line and log-hash evidence, and the binding fixed-versus-selected
reporting policy, are in `docs/BASELINE_PROTOCOL_AUDIT_2026-08-31.md` and
`evidence/peft_boa_protocol_audit_20260831.json`.

## Audit the Signal checkpoint comparator

The official Signal checkout is pinned and left unmodified at:

```text
/root/mmreid-trifusion/baselines/Signal
cd1b0a672d1fe642e7608731cb4899a19dda7d51
```

Its released torch 2.1.1+cu118 stack is compatible with the isolated
`peft_boa` environment for a CPU-only data-path probe. With CUDA masked,
`DATALOADER.NUM_WORKERS=0`, and only the path convention adapted in memory,
the real released B64/K8 loader returns eight identities and RGB/NI/TI tensors
of shape `64×3×256×128`. The audited corpus has 171 training identities and
3951 training records; validation concatenates the same 836-record test list
twice as query and gallery. No model was constructed during this probe.

Signal is not yet metric-reproduced. The README routes all three released
checkpoints through one Baidu share (code `sign`); the share resolved during
the audit, but the RGBNT201 file bytes were not acquired. The released
`test.py` has no checkpoint CLI argument and hard-codes
`/media/zpp2/Datamy/lyy/signal_50.pth`. The CLIP loader separately hard-codes
another author path, the constructor forces `.to("cuda")`, and
`requirements.txt` line 19 references an author-local `grad-cam` checkout.
The verified local OpenAI CLIP archive is available and exactly matches its
official digest, but replacing these paths and running a model/capacity gate
remains future implementation work.

The upstream test log reports 80.3% mAP / 85.2% Rank-1 / 91.4% Rank-5 /
93.7% Rank-10 with no reranking. It is a 3,390-byte released artifact with
SHA-256 `b200abf8…793c83e`, not a local result. The training loop evaluates the
official test after every epoch and keeps `Signalbest.pth` by test mAP; the
periodic epoch-50 checkpoint is saved before that evaluation. These boundaries,
source hashes, loader shapes and unresolved checkpoint provenance are frozen in
`evidence/signal_source_protocol_audit_20260831.json`.

## Reconstruct and audit the MFRNet checkpoint comparator

The official MFRNet checkout and directly downloaded RGBNT201 checkpoint are:

```text
/root/mmreid-trifusion/baselines/MFRNet
ec54a1302321cda4b5fad9ca1c0878dabf0b46b6
/root/mmreid-trifusion/checkpoints/MFRNet/RGBNT201_MFRNetbest.pth
```

The checkpoint came from the official README Google Drive link. Its HTTP
attachment name and final size are `RGBNT201_MFRNetbest.pth` and 407,297,967
bytes; local SHA-256 is
`f0c2df33f3901738051a917e728c73d9b494113e1e327361bc8f1acf4711126e`.
It was first inspected with torch 2.5.1 `weights_only=True`: all 297 entries are
tensors with 101,794,851 elements and no pickle-side metadata.

The released `requirements.txt` is not portable: its torch/vision/audio and
PyG wheels use author-local `file:///root/py38/...` URLs and it includes a broad
machine export unrelated to evaluation. Rebuild the isolated, minimal audited
runtime in two phases:

```bash
cd /root/mmreid-trifusion/TriFusion-ReID
/root/miniconda3/bin/conda env create \
  -f environment/mfrnet_environment.yml
PYTHONNOUSERSITE=1 \
  /root/miniconda3/envs/mfrnet/bin/python -m pip install \
  -r environment/mfrnet_requirements-lock.txt
PYTHONNOUSERSITE=1 \
  /root/miniconda3/envs/mfrnet/bin/python -m pip install \
  --no-build-isolation --no-deps \
  -r environment/mfrnet_tutel_source.txt
PYTHONNOUSERSITE=1 \
  /root/miniconda3/envs/mfrnet/bin/python -m pip check
```

The split installation is required because Tutel's source build imports torch
during metadata/build. The author's unavailable `tutel==0.3` dependency is
reconstructed from the official Tutel v0.3.2 commit
`d4c20c3e…b3c8b0`; v0.3.0 self-identifies as distribution 0.2 and does not
implement the custom-expert constructor used by MFRNet. The pinned v0.3.2
archive has SHA-256 `d1f9bbe0…c805ea` and installs as distribution 0.3.
`PYTHONNOUSERSITE=1` is mandatory so the environment cannot silently import
packages from `/root/.local`.

The resulting Python 3.8.20 / torch 1.12.0+cu113 / torchvision 0.13.0+cu113 /
timm 1.0.12 / Tutel 0.3 environment passes `pip check`, a real B64/K8 loader
batch and CPU model construction. The final checkpoint passes
`load_state_dict(strict=True)` with 297 model/checkpoint tensors,
101,794,851 elements, and zero missing, unexpected, shape-mismatched or
dtype-mismatched entries.

This is checkpoint compatibility, not metric reproduction. The released model
constructor forces CUDA, the exact cu113 stack has not passed on the RTX 4060,
and the upstream loop evaluates official test every epoch before saving
`MFRNetbest.pth` by test mAP. The official 80.7% mAP / 83.6% Rank-1 therefore
remains a test-selected published value until a GPU run reproduces it. The
source, environment, loader, checkpoint and claim boundary are frozen in
`evidence/mfrnet_rgbnt201_checkpoint_audit_20260831.json`. The public repository
also has no explicit `LICENSE` file, which is a reuse/distribution warning.

## Reproduce the DeMo implementation base

DeMo's released CLIP loader hard-codes an unavailable author-machine path.
`tools/run_demo_baseline.py` injects the verified local CLIP archive at runtime
and otherwise runs the upstream checkout at commit
`b4f323a430b32e3a1637c3e7acb25868cb52e9cd` unchanged.

The released RGBNT201 configuration uses B64/K8 and was designed for a GPU with
substantially more memory. A real-data capacity probe on this 8 GB RTX 4060
established B32/K4 as the largest safe profile with eight identities per batch:
eight official-loss AMP steps pass, the final two have finite gradients for all
322 trainable parameter tensors, and peak allocated memory is 6894.18 MiB. The
evidence is in `artifacts/demo_real_step_probe_b32_20260831.json`.

The released test batch of 128 reached 7885 MiB and produced an 857-second
epoch-2 evaluation stall after a normal 30-second epoch-1 pass. That partial run
is retained as a negative systems result and is not spliced into a later run.
The fixed local profile therefore uses `TEST.IMS_PER_BATCH=64`; it changes only
inference batching, not features, distances, query/gallery order or metrics.

Run the fixed 50-epoch profile with:

```bash
cd /root/mmreid-trifusion/TriFusion-ReID
bash scripts/run_demo_rgbnt201_seed42.sh
```

Summarize a completed run, including all checkpoint hashes, with:

```bash
python tools/summarize_demo_run.py \
  /root/mmreid-trifusion/runs/demo_rgbnt201_seed42_b32k4_tb64 \
  --expected-epochs 50 --require-complete --hash-checkpoints
```

Live summaries fail closed unless at least one training epoch is complete, the
completed epoch indices form a contiguous `1..latest` prefix within
`--expected-epochs`, and the observed evaluation keys exactly equal that prefix
crossed with `{ori, moe, joint}`. This prevents a truncated log or a
just-finished training epoch from being reported valid while epochs or
evaluation passes are missing.
The boundary was exercised on epoch 12: before the fix, 34 records (only the
epoch-12 `ori` pass present) incorrectly returned `valid=true`; after the fix, a
35-record partial snapshot returned false, and the fully closed 36-record
snapshot restored the coverage check to true. A 10,000-second limit was used
only to isolate this coverage behavior. Under the unchanged registered
300-second limit, the closed snapshot remains `valid=false` because the latency
gate still fails.

One-off public-CLI contract probes also cover an empty log, a `1,3` epoch gap,
an epoch beyond the configured bound, duplicate train/evaluation records, a
partial mode set and one complete epoch. Every malformed or partial case exits
1; the complete case exits 0. The immutable result receipt is versioned under
`evidence/`; a persistent regression-test seam remains gated on the user's TDD
seam agreement.

The TB64 run passed the pre-registered live gate after two fully evaluated
epochs. All six `(epoch, feature-mode)` records were complete, losses and
metrics were finite, no fatal log pattern was present, and the largest single
evaluation was 224.226 seconds, below the 300-second limit. The epoch-2 `ori`
pass temporarily occupied 7919/8188 MiB framebuffer memory and nearly all BAR1,
so the gate records a paging-pressure warning even though the process remained
at 100% GPU utilization and completed. The immutable live-gate snapshot is
`/root/mmreid-trifusion/artifacts/demo_rgbnt201_seed42_b32k4_tb64_gate2_20260831.json`
(SHA-256 `c7a7738c8e59d61b064092d5f6c59f7f1db8302ff599b04cb9fe5c4b0a092d40`).
This snapshot is a systems gate, not the final 50-epoch accuracy result.
After removing timestamps, the 52 logged training-loss and retrieval-metric
records from epochs 1--2 are byte-for-byte identical between the excluded
TB128 run and the restarted TB64 run. This confirms that the restart reproduced
the same optimization trajectory and that inference batch size did not alter
the six reported retrieval results; checkpoint or metric splicing was not used.

The two-epoch gate did not guarantee all later evaluations would remain below
300 seconds. Epoch-7 and epoch-9 `ori` passes took 329.741s and 327.717s. The
frozen epoch-10 summary therefore correctly reports
`evaluation_latency_within_limit=false` and `valid=false`; the limit is not
retuned post hoc. Because both passes completed, subsequent MoE/joint passes
continued normally, and fatal/nonfinite checks stayed clean, R012 continues as
a calibration run with a systems WARN rather than being misreported as a fully
valid latency run.

This B32/K4 result is a hardware-matched implementation-base calibration, not
an exact reproduction of the paper's B64/K8 optimization protocol. All
TriFusion ablations use the same B32/K4 identity sampling so comparisons remain
matched. The separately reproduced MDReID public checkpoint above remains the
primary high-metric baseline.

The released RGBNT201 dataset class points both query and gallery at `test`, and
the training loop evaluates it every epoch before selecting `DeMobest.pth` by
joint mAP. Accordingly, `DeMobest.pth` is explicitly test-selected. R012 will
also preserve `DeMo_50.pth`, which is saved before epoch-50 evaluation and is
the pre-registered fixed-epoch matched baseline. See
`docs/BASELINE_PROTOCOL_AUDIT_2026-08-31.md` for source lines, hashes and the
binding paper-reporting policy.

The first periodic checkpoint is `DeMo_10.pth`, SHA-256
`b2ab79f056d73d6b827c52fd27ec0607aeae1a10cd756db5c0cc62f3ab4631c0`.
Its filesystem timestamp is before the epoch-10 evaluation-start log entry; the
read-only receipt is versioned under `evidence/`.

R012 began before the launcher gained its commit/dirty and GPU-occupancy
fail-closed checks. The versioned
`evidence/demo_rgbnt201_seed42_b32k4_tb64_live_provenance_20260831.json`
receipt explicitly has `launch_attestation=false`; it binds the exact live
command, logged effective B32/K4/TB64 overrides, and the clean frozen DeMo
checkout observed during the run. It must not be described as proof that the
later guards executed at launch.

## Train-only development protocol

Do not treat the provided `train_171 - train_141` identity difference as a
retrieval dev set: 20 of its 30 identities have only one camera and therefore
have no valid positive after the official same-pid/same-camera exclusion.
Generate the frozen cross-camera protocol with:

```bash
python tools/build_rgbnt201_dev_protocol.py
```

The deterministic hash rule selects 30 of the 51 multi-camera training
identities, leaves 141 identities for fitting, and gives all 825 dev queries a
valid cross-camera positive. It is derived only from `train_171` metadata and
has zero test-identity overlap. Final promoted configurations are retrained on
all 171 training identities.

## Verify the versioned evidence

Key JSON outputs are mirrored byte-for-byte under `evidence/`, with the runtime
copies under `/root/mmreid-trifusion/artifacts` remaining canonical. Verify the
repository bundle with:

```bash
sha256sum -c evidence/SHA256SUMS
```

The bundle is intentionally limited to data, environment and baseline gates.
It contains no final TriFusion result and cannot by itself support a SOTA claim.
