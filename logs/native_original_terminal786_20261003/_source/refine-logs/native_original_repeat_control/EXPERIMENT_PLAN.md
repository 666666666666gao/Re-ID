# Native original/original control after sealed V5 failure

Status: implemented for source review; not run. Scientific nine-arm plan remains unmet.

Section 783 sealed V5: global209/semantic281 fixed-gradient comparisons pass, native90/295 fail; no updates, M0, formal training or weights. Fresh rescue `FAIL_STOP_V5_NO_LOCALIZED_REPAIR` localized no source repair. Its sole recommended next action is this bounded distinguishing control. Semantic-only section776 repeat cannot prove native repeatability.

Use two fresh unpartitioned `run_independent_native_evidence.build_core` native models, no partition/checkpoint/CPU-save helpers. Original publicCLIP/freshcamera/heads/roles/adapters, initial full-state `7f5ff300a45faed287a510dacc68d06c006931504a4510fffd8c47e43d93b852`, author RGBNT201 B64/K8 seed42. Same first32 of one author-loader batch; labels/cameras/paths/image digests must match the existing V5 native measurement or stop. Read existing source/config/initialization receipts; no new scientific variant.

2026 physical GPU0 compute, same GPU0/1 visibility and CPU+two-CUDA RNG accounting as V5 reference. Launch only after actual pair-free/no-preemption and disk check. FP32 parameter storage, train mode/original author losses and optimizer groups; AMP FP16/scaler256. Each model exactly one forward/backward/unscale. ZERO optimizer.step, scheduler.step, scaler.update, scorer calls or weight saves; no retry or automatic continuation. Two original models share values, not parameter objects or forward graphs.

Original gates unchanged: outputs/loss/allheads atol=rtol1e-5, all295 gradients1e-4, key/None/shape/finite checks; exact buffers, CPU+two-CUDA RNG, BN1, equal post-state, parameters unchanged, optimizer ownership/groups, capture-hook removal. Serialize every already-required comparison before assertions so a gradient failure cannot hide head/buffer/RNG/state results again. Input/state/runtime failure remains a failed control, not a replacement-batch opportunity.

Command (once, after fresh source review/publication):

```sh
CUDA_VISIBLE_DEVICES=0,1 /data/gaob/Re-ID/conda-envs/tri_reid/bin/python -B tools/check_native_original_repeat.py --reference-campaign logs/independent_native_evidence_20261003_v5 --output-dir logs/native_original_repeat_control_20261003_v1
```

Expected model runtime about 60–120 seconds. First observation at180 seconds; a confirmed live process may be observed again at180–300 seconds. Controller must retain child exit and log. No production queue/scorer/model edits, no2025 CUDA or model execution.

PASS proves only this newly controlled original-native repeat witness; V5 remains failed, not repaired/restartable, all9 M0/formal gates still pending and B128 capacity unproved. FAIL means original native also misses the fixed gate, so existing evidence cannot single out partition as cause: seal and stop GPU parity trials, source analysis only, no gate relaxation. Input/state/runtime failure: seal and stop, no alternative precision/batch/seed or repeat-until-pass. Historical full gradients/RNG were not saved; matching recorded metadata is not exact reconstruction of historical gradients/RNG.

No new weights expected. Formal best-only retention remains unchanged; existing public/author/input dependencies and receipts remain protected.
