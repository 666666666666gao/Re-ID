# Source review R4: region-use queue

**Verdict: FAIL (source only).** One medium/P2 manifest-pinning defect in the preserved R4 queue.

Requested route: `gpt-6-astra`, effort `max`. Actual serving model/effort: **unattested**. Native Codex, same-family provisional follow-up by the same reviewer; no new independent-review claim.

**R4-PIN-01: accepted initialization hashes are not persisted to the manifest.** `queue_region_use_probe.py:101` writes `manifest.json` once with an empty initialization map. The prepare branch updates only `state` at `:116`; `:124` writes only `campaign.json`. Inherited `queue_native_research.py:63` invokes `queue_foundation_recipe.require_sources`, which reads initializer hashes from **manifest.json** (`queue_foundation_recipe.py:110–115`). Thus this byte-level initializer gate stays empty for all six jobs.

The deterministic in-memory fixture finishes with six pins in campaign.json and zero in manifest.json. Appending JSON whitespace to a prepared fixture initializer before M0 passes the original manifest gate. No real file/model was altered. Actual binding/state comparison still exists, so this is a provenance guard omission, not proof of model-state inequality.

Minimum fix: immediately after setting the new hash, add `base.queue.write(campaign/"manifest.json", dict(state))`. No fallback, optimizer, objective or scheduling change is needed.

R4 source was preserved before the author applied that line. This report remains an immutable FAIL for queue SHA256 `8c00fd16f7539f70dbaf5bae85720a7a5fa283e5f6ef0a99b76963ec1c117964`; R5 separately reviews the correction.

| Check | Result | Evidence |
|---|---|---|
| old_semantic_initializer_equality | PASS_SOURCE_RUNTIME_UNVERIFIED | tools/queue_region_use_probe.py:43-54; tools/run_independent_native_evidence.py:76-90; tools/run_native_partitioned.py:23-30; tools/run_foundation_recipe.py:103-107 |
| six_per_endpoint_first_attempts | PASS_SOURCE_AND_CONTROL_FIXTURE | tools/queue_region_use_probe.py:29-40; tools/queue_region_use_probe.py:84-129; tools/queue_native_research.py:17; tools/run_foundation_recipe.py:189-193; tools/run_region_use_probe.py:115-122 |
| eight_update_gate_and_probe_retirement | PASS_SOURCE_AND_CONTROL_FIXTURE | tools/queue_region_use_probe.py:57-81; tools/accept_region_use_m0.py:8-36; tools/run_training_feature_scale.py:50-66; tools/run_foundation_recipe.py:221-284; tools/run_independent_native_evidence.py:138-174 |
| strict_official_evaluation | PASS_SOURCE | tools/queue_region_use_probe.py:119-123; tools/run_foundation_recipe.py:174-186; tools/run_foundation_recipe.py:287-312 |
| GPU_and_storage_boundaries | PASS_SOURCE_RUNTIME_UNVERIFIED | tools/queue_region_use_probe.py:18-19; tools/queue_region_use_probe.py:85-90; tools/queue_region_use_probe.py:105-109; tools/queue_native_research.py:49-78 |
| source_map_binding_and_future_seal | PASS_SOURCE_EXCEPT_R4_INITIALIZER_MANIFEST_FINDING | tools/queue_region_use_probe.py:22-26; tools/queue_region_use_probe.py:87-102; tools/queue_foundation_recipe.py:110-115 |

**old_semantic_initializer_equality.** Accepted bindings compare every old field except architecture/entry_sha256/scope, including complete model state, visual/camera/shared hashes, parameter counts, placement, cfg, seed, protocol and public CLIP provenance. Source producer computes state hash from the newly constructed actual model. The three old rows are semantic. A changed full-model hash is rejected by the actual acceptance function in a text fixture. Actual equal model initialization, RNG trajectory and output remain unexecuted; no new numerical-parity prerequisite is imposed.

**six_per_endpoint_first_attempts.** Exactly six planned jobs in order201 uniform,201 ambiguity,MSVR uniform,MSVR ambiguity,100 uniform,100 ambiguity. Success path has24 producer stages: prepare,m0,train,evaluate per endpoint. Each accepted M0 precedes its own fresh50; no all-six M0 barrier. Fresh subprocess/train builds public initialization, never the M0 probe. Existing campaign/initializer/training-output assertions prevent restart/overwrite. Failures stop the controller; remaining planned jobs can stay unattempted. There is no promise that six jobs finish if an earlier gate fails and no retry/fallback/preemption path.

**eight_update_gate_and_probe_retirement.** Native finite/update/BN/reload plus source support/activity gate is checked before training. BatchOrderLoader logs after yielding, so the break before processing fetched ninth batch leaves exactly8 processed rows; stdlib fixture confirms this. Old first8 batch order is pinned in original903 artifacts. Consumer uses check=True; a rejecting consumer stops before train and before unlink. Successful acceptance precedes probe hash/path checks, sole-pth check, deletion of only that new endpoint probe and retirement receipt. Source fixture uses a non-model byte string entirely in memory, never a real checkpoint.

**strict_official_evaluation.** Controller consumes official_metrics.json with the new schema, COMPLETE,50 epochs and matching comparison. Active evaluator requires all50 epochs, rebuilds equal binding, strict-loads best full state with condition/protocol checks, selects best mAP with latest tie and reproduces same-checkpoint metrics before writing that exact filename/schema. Historical base.verify is not called; its old SCHEMA is irrelevant. No new result exists.

**GPU_and_storage_boundaries.** Entry requires canonical remote repo and CUDA_VISIBLE_DEVICES=0,1. Active inherited waiter queries only --id=0,1, polls240s, never preempts; Popen explicitly uses0,1. Controller checks512MiB next-artifact plus2GiB reserve once initially and before every producer stage (25 calls on a24-stage successful fixture). Inherited logger additionally checks2GiB before waiting. These are launch-time budget assertions, not measured storage cost or a disk reservation. Actual machine/GPU availability and storage qualification remain pending.

**source_map_binding_and_future_seal.** base.source_map is rebound before inherited run_logged calls require_sources; the same function globals are used, and source map equals the saved manifest each stage. Artifact pins are checked at coordinate entry. Old seal has439source/50artifact/3semantic rows. New seal creation, exact five runtime py inclusion and fixed plan/report pins are still pending and are not falsely certified here. New scope absence is not a source bug in an unregistered draft. Initializer manifest omission is the single R4 defect.

Fixtures ran actual extracted controller, initialization acceptance, M0 retirement and inherited manifest checker functions with in-memory paths and stub producer/consumer. They did not import/construct a neural model, launch a process/queue, query a GPU, load a checkpoint/cache, perform an optimizer update, or edit production files. The gate itself retains the R2 text-fixture checks; no numerical component checks were repeated.

The six jobs are planned first attempts. The controller stops on failure and leaves remaining jobs unattempted; this does not impose a whole-six M0 barrier and does not retry a failed endpoint. The new seal is still pending, so exact registration/pins/storage/machine qualification and real initializer/M0 execution remain unverified. Source approval is not launch or retrieval approval.

## Exact round input hashes

| File | SHA256 |
|---|---|
| `tools/queue_region_use_probe.py` | `8c00fd16f7539f70dbaf5bae85720a7a5fa283e5f6ef0a99b76963ec1c117964` |
| `modeling/trifusion/region_use_objective.py` | `556876002cc7abef5a3525940781b8facdf6a1e8bfc792e5b900f16a5f72fe6d` |
| `modeling/trifusion/region_use_heads.py` | `1680d04f7b26f513380509af88bfce555cbc04497162e4af10c47dcbbcc9b63e` |
| `tools/run_region_use_probe.py` | `176e1e88a74ede5447f57cf0e3a89c1a0847e729647d37b11fdb6c76ea715591` |
| `tools/accept_region_use_m0.py` | `f378797bcb9c641156de215d38b9db8dc34668c5ac7dad15bf214decd442f3ac` |
| `refine-logs/region_use_supervision_v1/EXPERIMENT_PLAN.md` | `576138fc39d97d251fcb2522a11ad184c8eb66c0f388b009f2b8f1e634214074` |

Full directly reviewed file hashes and precise line coverage are in the JSON report. Inherited producer-source evidence is pinned to SOURCE_REVIEW_R3.json. Final hash check: 2026-10-12T11:31:02.148979+00:00.

Same-family provisional. Requested model/effort is distinct from unattested actual identity. Source verdict is not initialization/M0/NN/performance qualification; no novelty guarantee.
