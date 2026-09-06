# MSVR310 TriFusion original three-role source-only M0 audit

Audit date: 2026-09-06

Scope: completed original-role M0 only, including the prelaunch binding revision, initialization provenance, M0 executor report, M0 logs, M0 summary, independent fold/overfit training JSONs, receipts, source snapshots, and protocol/config bindings registered in `.aris/traces/experiment-audit/2026-09-06_run18/input_manifest.json`. The separately launched 20-epoch comparison has no terminal result in the manifest and is outside this audit. Signal B0 was inspected only as the fixed initialization prerequisite for M0.

Overall verdict: `WARN`, with `PASS_ENGINEERING_ONLY` for the completed M0 engineering claim. I found no real arithmetic, binding, source-boundary, initialization-reuse, update-count, exposure-count, or report-consistency error in the completed M0 evidence. The warning is a scope qualification: the evidence proves a source-only engineering smoke/overfit M0, not unknown-identity retrieval improvement, not official MSVR310/RGBNT201 results, and not independent local tensor/image/binary reproduction.

## Reviewer provenance

The reviewer was dispatched by the root session using `collaboration.spawn_agent` with `fork_turns: "none"` and requested model metadata `gpt-5.5` / `xhigh` (`.aris/traces/experiment-audit/2026-09-06_run18/001-m0.request.json:2-10`). The dispatch observation records the accepted task name and explicitly states that the backend was not independently attested and the review-family scope is GPT-family Type-A only (`.aris/traces/experiment-audit/2026-09-06_run18/dispatch_observation.json:2-10`). The shared routing instructions define the second Codex reviewer as Type-A and state that same-family review does not provide Type-B cross-family evidence (`C:/Users/gb/Auto-claude-code-research-in-sleep/skills/skills-codex/shared-references/reviewer-routing.md:8-15`).

I did not delegate this audit further. That fact is recorded in my replay output as `no_further_delegation_by_this_reviewer: true`; it is an audit-process fact, not evidence about backend model availability.

## Input integrity and immutable manifest

The audit started from `.aris/traces/experiment-audit/2026-09-06_run18/input_manifest.json`. The manifest lists 60 local input files and gives the actual remote LF source snapshots for the five original remote source files (`.aris/traces/experiment-audit/2026-09-06_run18/input_manifest.json:60-82`, `.aris/traces/experiment-audit/2026-09-06_run18/input_manifest.json:305-322`). Its scope line explicitly limits the inputs to completed original-role M0 and notes that the comparison terminal result is unavailable (`.aris/traces/experiment-audit/2026-09-06_run18/input_manifest.json:322`).

I rechecked every manifest input hash before writing this report and again before finishing. The replay run verified `file_count = 60`, `all_ok = true`, and manifest SHA-256 `8e96f807e582f8f9ed98afeb11a2223c7ee7979ead9e19189dffcbac994cb783` in `.aris/traces/experiment-audit/2026-09-06_run18/msvr310_trifusion_m0_replay_result.json`. A final hash-only recheck was also saved under run18 after the reports were produced.

## A-F audit classification

| Category | Verdict | Evidence |
|---|---:|---|
| A. GT/provenance and data boundaries | `PASS` | The protocol uses only `bounding_box_train`, a deterministic label-only split, and internal source/heldout evaluation metadata; official query/gallery data are unused (`protocols/msvr310_train_oof_v1.json:2-22`). Each fold source and heldout identity set is disjoint and has the expected source/heldout/gallery/query counts (`protocols/msvr310_train_oof_v1.json:15523-15536`, `protocols/msvr310_train_oof_v1.json:19127-19140`, `protocols/msvr310_train_oof_v1.json:22489-22502`). M0 itself records `official_test_image_access = 0`, `rgbnt201_dev_image_access = 0`, `source_only_training = true`, and `evaluation_type = source_only_engineering` (`evidence/trifusion_msvr310_trifusion_v1_m0_complete_20260906.json:3477-3484`). |
| B. Loss arithmetic and score normalization | `PASS` | The configured V8 loss weights are fixed in the experiment config (`configs/MSVR310/TriFusion-source-oof-v1.json:59-67`), and the actual weighted-loss function matches the V8 formula in `tools/run_signal_preserving_v5.py:99-142`. I independently recomputed all 124 weighted training losses from stored scalar components; max absolute difference from recorded totals was `4.122654591043329e-07`, consistent with float serialization. Epoch mean differences were exactly `0.0`. The label-smoothing floor and overfit ratio also recomputed exactly to the values below. |
| C. File existence, hashes, and report matching | `PASS` | The wrapper exited `0` (`evidence/trifusion_msvr310_trifusion_v1_m0_exit_20260906.txt:1`). The M0 summary reports `PASS_ENGINEERING_ONLY`, fixed config/runner/baseline/protocol hashes, and project commit `1c444cdf...` (`evidence/trifusion_msvr310_trifusion_v1_m0_complete_20260906.json:2-8`, `evidence/trifusion_msvr310_trifusion_v1_m0_complete_20260906.json:31`). Independent fold/overfit training JSONs match the embedded summary training blocks byte-for-byte after JSON normalization; receipt counts and states match the summary. The executor report claims match the summary/log/receipt evidence (`results/TRIFUSION_MSVR310_ORIGINAL_ROLES_M0_20260906_062417.md:3-57`). |
| D. Dead code, call paths, and five-output parity | `PASS` | M0 calls `build_model`, `output_mapping`, `train_roles`, the M0 capacity path, the fresh fold-0 overfit path, checkpoint save/reload, and strict five-output equality checks (`tools/train_msvr310_trifusion_oof.py:27-55`, `tools/train_msvr310_trifusion_oof.py:68-73`, `tools/train_msvr310_trifusion_oof.py:98-183`, `tools/train_msvr310_trifusion_oof.py:337-390`). Retrieval comparison code exists later in the same runner but is outside M0 mode (`tools/train_msvr310_trifusion_oof.py:196-238`). |
| E. Evidence scope | `WARN` | The registered plan keeps weights/features/distances/images remote and limits local review to JSON/text/ranking/hash evidence (`refine-logs/msvr310_trifusion_v1/EXPERIMENT_PLAN.md:113-114`). I did not import model/tensor/image libraries or inspect remote binary/image artifacts locally. The five-output and frozen-state claims therefore pass as receipt-and-code evidence, not as local tensor replay. |
| F. Evaluation type and scientific claim | `WARN` | The plan says M0 is a source-only engineering gate, not a new method/frontier claim, official result, or multiseed result (`refine-logs/msvr310_trifusion_v1/EXPERIMENT_PLAN.md:10-22`). M0 sets `checkpoint_selection = discarded_m0` and `m0_weights_reused = false` (`evidence/trifusion_msvr310_trifusion_v1_m0_complete_20260906.json:3477-3484`). No unknown-identity retrieval or official MSVR310/RGBNT201 conclusion is established by M0. |

## Independent scalar recomputation

I used only local JSON/text and Python stdlib. The calculation command and complete output were saved as:

```text
C:\Users\gb\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe C:\Users\gb\.trifusion_github_publish_22c3bee\.aris\traces\experiment-audit\2026-09-06_run18\msvr310_trifusion_m0_replay.py --output-json C:\Users\gb\.trifusion_github_publish_22c3bee\.aris\traces\experiment-audit\2026-09-06_run18\msvr310_trifusion_m0_replay_result.json
```

The replay output records command, timing, manifest verification, engineering arithmetic, initialization checks, label-smoothing floor, overfit gate, protocol checks, receipt checks, reviewer provenance, scalar recomputation, B0 binding, source binding, and sampling details (`.aris/traces/experiment-audit/2026-09-06_run18/msvr310_trifusion_m0_replay_output.txt:1-9`, `.aris/traces/experiment-audit/2026-09-06_run18/msvr310_trifusion_m0_replay_output.txt:26-111`, `.aris/traces/experiment-audit/2026-09-06_run18/msvr310_trifusion_m0_replay_output.txt:620-756`, `.aris/traces/experiment-audit/2026-09-06_run18/msvr310_trifusion_m0_replay_output.txt:759-844`).

Recomputed results:

| Quantity | Independent value |
|---|---:|
| Manifest file count | `60` |
| Manifest SHA-256 | `8e96f807e582f8f9ed98afeb11a2223c7ee7979ead9e19189dffcbac994cb783` |
| Recomputed training steps | `124` |
| Max weighted-loss absolute difference | `4.122654591043329e-07` |
| Max epoch-mean absolute difference | `0.0` |
| Capacity fold-0 epoch mean | `3.9842726290225983` |
| Capacity fold-1 epoch mean | `4.000509351491928` |
| Capacity fold-2 epoch mean | `4.056699812412262` |
| Overfit epoch mean | `0.7473260003328324` |

The V8 loss formula is:

```text
ID_FUSED * fused_id
+ TRIPLET_FUSED * fused_triplet
+ ID_BRANCH * (cnn_id + transformer_id + mamba_id)
+ TRIPLET_BRANCH * (cnn_triplet + transformer_triplet + mamba_triplet)
+ ID_RESIDUAL * (cnn_residual_id + transformer_residual_id + mamba_residual_id)
+ TRIPLET_RESIDUAL * (cnn_residual_triplet + transformer_residual_triplet + mamba_residual_triplet)
```

That matches the configured weights in `configs/MSVR310/TriFusion-source-oof-v1.json:59-67` and the implemented weighted-loss path in `tools/run_signal_preserving_v5.py:99-142`. The criterion path uses label-smoothed cross entropy and normalized batch-hard triplet components for fused, branch, and residual heads (`modeling/trifusion/signal_preserving_v8.py:690-742`).

## Label-smoothing entropy floor and fixed overfit gate

The overfit gate uses the analytic identity-loss entropy floor implemented in `tools/run_signal_preserving_v5.py:1580-1598` and the excess-loss ratio implemented in `tools/run_signal_preserving_v5.py:1601-1618`. For fold 0 overfit, the source class count is `103`, smoothing is `0.1`, and identity loss weight is `0.25 + 3*(1/12) + 3*(1/12) = 0.75`.

Independent recomputation:

| Quantity | Value |
|---|---:|
| Correct-class probability | `0.9009708737864078` |
| Other-class probability | `0.0009708737864077671` |
| Entropy | `0.7809515103250464` |
| Weighted entropy floor | `0.5857136327437849` |
| Initial overfit loss | `4.122129917144775` |
| Final overfit loss | `0.5881962776184082` |
| Initial excess over floor | `3.5364162844009908` |
| Final excess over floor | `0.002482644874623352` |
| Fixed overfit ratio | `0.0007020228035862781` |
| Gate threshold | `0.1` |
| Gate verdict | `PASS` |

These match the executor report's floor/initial/final/ratio claim (`results/TRIFUSION_MSVR310_ORIGINAL_ROLES_M0_20260906_062417.md:26-32`) and the summary loss-gate block (`evidence/trifusion_msvr310_trifusion_v1_m0_complete_20260906.json:12789-12797`).

## Update counts, exposures, and memory condition

The config fixes capacity steps to `8`, overfit steps to `100`, batch size to `64`, and AMP scale to `256` (`configs/MSVR310/TriFusion-source-oof-v1.json:49-57`). The plan expects M0 to run `3*8 + 100 = 124` optimizer updates, `7936` exposures, and zero heldout/dev/official forwards (`refine-logs/msvr310_trifusion_v1/EXPERIMENT_PLAN.md:73-88`). The M0 runner executes three capacity folds, then a fresh fold-0 overfit run (`tools/train_msvr310_trifusion_oof.py:337-405`).

Independent arithmetic:

| Quantity | Value |
|---|---:|
| Capacity optimizer steps | `24` |
| Overfit optimizer steps | `100` |
| Total optimizer steps | `124` |
| Capacity exposures per fold | `512` |
| Total capacity exposures | `1536` |
| Overfit exposures | `6400` |
| Total exposures | `7936` |
| Summary optimizer steps | `124` |
| Summary heldout record forwards | `0` |
| Peak allocated MiB | `5727.02978515625` |
| Peak reserved MiB | `6284.0` |
| Under 24 GiB condition | `true` |

The executor report's count line matches this (`results/TRIFUSION_MSVR310_ORIGINAL_ROLES_M0_20260906_062417.md:3-6`), and the top-level summary repeats `optimizer_steps = 124`, elapsed seconds, and zero heldout forwards (`evidence/trifusion_msvr310_trifusion_v1_m0_complete_20260906.json:12810-12812`).

## Real source-index and identity sampling

The runner records batch record indices for every training step and asserts each batch shape is `(64, 3, 128, 256)` with `8` identities and `8` samples per identity (`tools/train_msvr310_trifusion_oof.py:139-164`). The replay verified all sampled indices were in the fold source records and not in heldout records.

| Run block | Steps | Exposures | Unique source identities sampled | Unique source records sampled | Same-identity unordered pairs | Cross-scene positive unordered pairs |
|---|---:|---:|---:|---:|---:|---:|
| Capacity fold 0 | `8` | `512` | `63` | `333` | `1792` | `440` |
| Capacity fold 1 | `8` | `512` | `61` | `316` | `1792` | `482` |
| Capacity fold 2 | `8` | `512` | `62` | `350` | `1792` | `543` |
| Fresh overfit fold 0 | `100` | `6400` | `8` | `53` | `22400` | `9200` |

The replay also confirmed the fixed overfit batch is repeated after the first overfit step. The detailed sampling sections are in `.aris/traces/experiment-audit/2026-09-06_run18/msvr310_trifusion_m0_replay_output.txt:933-1010`, `.aris/traces/experiment-audit/2026-09-06_run18/msvr310_trifusion_m0_replay_output.txt:1106-1181`, `.aris/traces/experiment-audit/2026-09-06_run18/msvr310_trifusion_m0_replay_output.txt:1277-1353`, and `.aris/traces/experiment-audit/2026-09-06_run18/msvr310_trifusion_m0_replay_output.txt:1449-1471`.

## Source and heldout boundaries

The fixed protocol defines:

| Fold | Source identities | Source records | Heldout identities | Gallery records | Valid queries |
|---|---:|---:|---:|---:|---:|
| 0 | `103` | `672` | `52` | `360` | `210` |
| 1 | `103` | `683` | `52` | `349` | `207` |
| 2 | `104` | `709` | `51` | `323` | `183` |

Across the protocol, source and heldout identity assignments are disjoint within each fold, the heldout identity union count is `155`, the total heldout gallery record count is `1032`, and gallery assignments are unique. M0 training batches came only from source records. M0 summary and receipts record no heldout forwards, no RGBNT201 development forwards, and no official test forwards (`evidence/trifusion_msvr310_trifusion_v1_m0_complete_20260906.json:1173-1180`, `evidence/trifusion_msvr310_trifusion_v1_m0_complete_20260906.json:2320-2327`, `evidence/trifusion_msvr310_trifusion_v1_m0_complete_20260906.json:3467-3474`, `evidence/trifusion_msvr310_trifusion_v1_m0_complete_20260906.json:3477-3484`).

## Initialization provenance and weight reuse

The M0 model builder sets seed `42`, loads the fold-specific fixed Signal B0 checkpoint, asserts source/heldout/fold binding, loads the Signal state, asserts final Signal state, builds V8 experts, asserts the baseline is frozen, and records `role_weights_loaded: false` (`tools/train_msvr310_trifusion_oof.py:27-55`). The runner asserts source file hashes, baseline summary hash, protocol hash, and B0 checkpoint hashes before M0 work starts (`tools/train_msvr310_trifusion_oof.py:288-335`).

M0 summary initialization evidence:

| Block | Evidence |
|---|---|
| Capacity fold 0 | Signal checkpoint and final-state binding, role weights not loaded, seed 42 (`evidence/trifusion_msvr310_trifusion_v1_m0_complete_20260906.json:51-56`) |
| Capacity fold 1 | Signal checkpoint and final-state binding, role weights not loaded, seed 42 (`evidence/trifusion_msvr310_trifusion_v1_m0_complete_20260906.json:1198-1203`) |
| Capacity fold 2 | Signal checkpoint and final-state binding, role weights not loaded, seed 42 (`evidence/trifusion_msvr310_trifusion_v1_m0_complete_20260906.json:2345-2350`) |
| Fresh overfit fold 0 | Fresh Signal checkpoint binding, role weights not loaded, seed 42 (`evidence/trifusion_msvr310_trifusion_v1_m0_complete_20260906.json:3487-3493`) |

Independent replay checks show each capacity fold's training initial state matches its initialization state, role state changes during training, frozen baseline/Signal states remain unchanged, and the fresh overfit run starts from the same fold-0 initialization rather than from the fold-0 capacity M0 final state. The top-level summary also records `checkpoint_selection = discarded_m0`, `m0_weights_reused = false`, and `rgbnt201_role_weights_reused = false` (`evidence/trifusion_msvr310_trifusion_v1_m0_complete_20260906.json:3477-3484`).

Signal B0 was checked only as M0's fixed initializer. Its summary is marked `COMPLETE_BASELINE_NOT_METHOD_QUALIFICATION` with fixed config/protocol hashes (`evidence/trifusion_msvr310_signal_v1_baseline_complete_20260906.json:4-8`), and it reports source-only training, zero official/fixed-RGBNT201-dev accesses, 1950 total optimizer steps, and fixed epoch-50 checkpoint selection (`evidence/trifusion_msvr310_signal_v1_baseline_complete_20260906.json:181775-181793`). M0 checkpoint/source-fold bindings match those B0 checkpoint/final-state records.

## Frozen-state, live-gradient, receipt, and five-output checks

The training loop records initial/final role state, frozen state, Signal state, optimizer steps, trainable gradient counts, missing gradient names, AMP overflows, step history, and memory (`tools/train_msvr310_trifusion_oof.py:167-183`). The engineering checks require no NaN, no AMP overflow, all trainable gradients live, frozen baseline unchanged, and Signal unchanged (`tools/train_msvr310_trifusion_oof.py:187-193`).

M0 fold summary blocks show `203/203` trainable parameters with live gradients, no missing trainable gradients, no overflows, unchanged frozen state, and unchanged Signal state for each capacity fold and for fresh overfit (`evidence/trifusion_msvr310_trifusion_v1_m0_complete_20260906.json:424-446`, `evidence/trifusion_msvr310_trifusion_v1_m0_complete_20260906.json:1571-1593`, `evidence/trifusion_msvr310_trifusion_v1_m0_complete_20260906.json:2718-2740`, `evidence/trifusion_msvr310_trifusion_v1_m0_complete_20260906.json:3861-3883`). Engineering check booleans are recorded for fold 0 at `evidence/trifusion_msvr310_trifusion_v1_m0_complete_20260906.json:1164`; the same structure is present for the other folds and overfit.

For each capacity fold, the receipt records a saved checkpoint, strict reload state matching the final role state, direct Signal source parity, strict reload all-output bitwise equality, source-only role/direct-signal forwards, and zero heldout forwards (`evidence/trifusion_msvr310_trifusion_v1_m0_complete_20260906.json:1173-1180`, `evidence/trifusion_msvr310_trifusion_v1_m0_complete_20260906.json:2320-2327`, `evidence/trifusion_msvr310_trifusion_v1_m0_complete_20260906.json:3467-3474`). The output set is exactly `baseline_only`, `fused`, `cnn`, `transformer`, `mamba` (`configs/MSVR310/TriFusion-source-oof-v1.json:41-47`; `tools/train_msvr310_trifusion_oof.py:22-24`, `tools/train_msvr310_trifusion_oof.py:68-73`).

Because tensor execution and binary inspection were outside the local audit scope, I classify these as `PASS_WITH_RECEIPT_LIMIT`: the code and receipts support the engineering claim, but this audit did not locally re-execute tensors or inspect remote checkpoint/image binaries.

## Prelaunch binding revision and original remote source files

The first prelaunch attempt at commit `148f5a7...` failed before remote process/run-directory creation because source byte hashes did not match; the recorded failure had `training_model_tensor_image_calls = 0` and `optimizer_updates = 0` (`evidence/trifusion_msvr310_trifusion_v1_prelaunch_source_bytes_20260906.json:2-10`). The five affected files were local CRLF byte snapshots versus remote LF byte snapshots. The evidence records `local_lf_bytes_match_remote: true`, `ast_equal: true`, and remote bytes matching the git blob for each file (`evidence/trifusion_msvr310_trifusion_v1_prelaunch_source_bytes_20260906.json:113-180`).

The R2 binding records a prepared revised config hash, no model/optimizer/gate/protocol changes, no runtime fallback, and no remote model calls or optimizer updates at binding time (`evidence/trifusion_msvr310_trifusion_v1_source_binding_r2_20260906.json:2-9`, `evidence/trifusion_msvr310_trifusion_v1_source_binding_r2_20260906.json:82-85`). Its five changed rows repeat the same LF-normalized and AST-equivalent evidence (`evidence/trifusion_msvr310_trifusion_v1_source_binding_r2_20260906.json:10-80`). The local manifest includes the exact remote LF snapshots for `criterion.py`, `state.py`, `builder.py`, `mamba_stream.py`, and `semantic_residual.py` (`.aris/traces/experiment-audit/2026-09-06_run18/input_manifest.json:60-82`, `.aris/traces/experiment-audit/2026-09-06_run18/input_manifest.json:305-322`).

I therefore do not classify the LF-only local/remote distinction as a code change. The binding revision is a hash-binding correction before any M0 model work, with no evidence of optimizer, model, protocol, loss, or gate change.

## Executor report assessment

The timestamped M0 report claims:

- `M0_PASS_COMPARISON_RUNNING_INDEPENDENT_AUDIT_PENDING`
- wrapper exit `0`
- `3 folds * 8 + 100 = 124 updates`
- `7936` exposures
- zero heldout/RGBNT201-dev/official forwards
- fixed B0 Signal loaded per fold
- frozen Signal/tail and fresh role/head seed 42
- no RGBNT201 role weights, router, HFER, V23/V24 path, or new loss
- `203/203` trainable gradients live
- no AMP overflow
- frozen SHA unchanged
- strict reload equality for all five outputs
- overfit floor/initial/final/ratio values
- source-only forward counts
- scope limitation to text/JSON/source and remote binary artifacts

Those claims are supported by the M0 summary, log, wrapper exit, receipts, config, protocol, and replayed arithmetic (`results/TRIFUSION_MSVR310_ORIGINAL_ROLES_M0_20260906_062417.md:3-57`, `evidence/trifusion_msvr310_trifusion_v1_m0_run_20260906.log:16-74`, `evidence/trifusion_msvr310_trifusion_v1_m0_exit_20260906.txt:1`). I found no report error that would require changing thresholds, editing executor reports, or inventing a different experiment.

## Evidence limits that remain

This audit used only local text/JSON/source and Python stdlib computations. It did not import model, tensor, or image libraries; did not run remote commands; did not access credentials; did not use browser/network; did not install packages; and did not inspect remote binary checkpoints or image artifacts. The registered plan explicitly leaves weights/features/distances/images remote under the current scope (`refine-logs/msvr310_trifusion_v1/EXPERIMENT_PLAN.md:113-114`).

The completed M0 establishes that the original three-role source-only training path runs, updates the intended trainable role/head parameters, preserves frozen Signal/baseline state by receipt, satisfies the fixed overfit gate, and keeps M0 weights out of subsequent comparison selection. It does not establish retrieval quality on unknown identities or official MSVR310/RGBNT201 benchmark performance. The 20-epoch comparison was launched separately and has no terminal result in the manifest inputs for this audit.
