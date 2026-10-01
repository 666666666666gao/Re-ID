# Exact current-runtime invocation

Existing warm environment; no package install or rebuild. Code review PASS is provisional and does not establish runtime or scientific performance. Execute from `/data/gaob/Re-ID/Trifusion` with `/data/gaob/Re-ID/conda-envs/tri_reid/bin/python -B`.

```bash
CUDA_VISIBLE_DEVICES=0 /data/gaob/Re-ID/conda-envs/tri_reid/bin/python -B tools/prepare_clean_clip_joint.py --output-dir /data/gaob/Re-ID/Trifusion/logs/clean_clip_preflight_20261002_v1 --gpus 0 1
```

This command verifies actual free resources, constructs six independent models, compares all common-state tensors, and runs only two real RGBNT201 eight-batch M0s on0/1. It creates new output paths; do not replay it or reuse failed directories. Preserve an actual failure and diagnose first. Formal training can run only after preflight.json COMPLETE and verified receipts:

```bash
/data/gaob/Re-ID/conda-envs/tri_reid/bin/python -B tools/queue_clean_clip_joint.py --campaign /data/gaob/Re-ID/Trifusion/logs/clean_clip_joint_20261002_v1 --preflight /data/gaob/Re-ID/Trifusion/logs/clean_clip_preflight_20261002_v1/preflight.json
```

The six endpoints each run full50 and one official-mAP-best strict full-state reload with all GT gallery rows. Four non201 arms run their own eight-batch M0 first; no M0 weights train a formal arm. Scheduler polls240sec on actual free0–3 GPUs, stops new dispatch upon a worker failure and drains active workers, no retry. Input is publicCLIP only, including fresh trainable camera and fresh heads. Public_clip_sha256 explicitly explains the reused legacy baseline_sha256 field. No new N1/N2/N3 structure runs here.
