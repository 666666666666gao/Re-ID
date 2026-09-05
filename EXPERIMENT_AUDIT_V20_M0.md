# V20 M0 Independent Experiment Integrity Audit

Date: 2026-09-05

Auditor: GPT-5.5 xhigh independent integrity auditor

Scope: bounded V20 stage audit requested in `.aris/traces/experiment-audit/2026-09-05_run06/AUDIT_REQUEST.txt`. I read the request and the listed primary artifacts directly. I did not use executor summaries, did not launch training or evaluation, did not inspect checkpoints/tensors/images/features/distances, and did not extend the M0 stage snapshot.

## Verdict

Overall verdict: **WARN**

Integrity status: **warn**

Engineering integrity: **PASS**

Scientific qualification: **NOT EVALUATED / UNSUPPORTED BY M0**

Observed stage: **V20 T0 + M0 engineering gate; Q1 had started but had no completed retrieval folds in the M0 snapshot**.

The V20 M0 engineering claim is supported: the artifacts show the cross-modal identity loss was implemented, smoke-tested on remote CUDA, preflighted across all 3 folds and 2 endpoints, and executed through the M0 capacity and 100-step overfit gates with the reported pass conditions reproducible from the local JSON. The scientific claim is not supported at this stage: the M0 snapshot literal fields are `status: "RUNNING"` (`evidence/trifusion_v20_m0_seed42_3cea5bf.json:3`) and `folds: []` (`evidence/trifusion_v20_m0_seed42_3cea5bf.json:9294`), with `dev_access_count: 0`, `official_test_access_count: 0`, `d1_executed: false`, and `next_phase_qualified: false` (`evidence/trifusion_v20_m0_seed42_3cea5bf.json:36-39`). There are no completed Q1 endpoint retrieval results in the snapshot and no evidence for complete-path, dev65, official, SOTA, or deployment claims.

## Primary artifacts read

All listed primary artifacts were read directly from `C:/Users/gb/.trifusion_github_publish_22c3bee` or its `.aris` trace path.

| Artifact | Evidence used | SHA256 observed locally |
|---|---:|---|
| `.aris/traces/experiment-audit/2026-09-05_run06/AUDIT_REQUEST.txt` | audit request, A-F checklist, scope, output schema | read directly |
| `modeling/trifusion/cross_modal_identity_v20.py` | loss implementation lines 10-32 | `c02e14c4731aba33ebb41d5efdae1b2a077c4b80622691e68014fe5ac280337e` |
| `tests/test_trifusion_cross_modal_identity_v20.py` | T0 unit tests lines 11-38 | `8e9438c2bac6ca8c381eb1c0d2cdfd70be2ceea169b3805aafddecc3bf70e132` |
| `tools/train_signal_preserving_v20.py` | contract, model loading, losses, preflight, M0/Q1 execution lines 35-401 | `2599cf9cdf48016350afe51c11acaa9c1ced27afcde18b33063c8ccb4e71700d` |
| `configs/RGBNT201/TriFusion-signal-preserving-v20-cross-modal-identity-rtx3090.yml` | seed, source, architecture, optimization, gates, endpoints lines 3-78 | `87d5a53ceb88d2546b9edf62510c3a112d669c4b60293563aba0a7d78cc94026` |
| `refine-logs/v20/EXPERIMENT_PLAN.md` | hypothesis, boundaries, M0/Q1 gate definitions lines 5-148 | `28bfbe5dd324e2600bc4bea06d8bfe4c3b1730409d21409d97a981c2b8a86f8f` |
| `refine-logs/v20/EXPERIMENT_TRACKER.md` | stage status and M0 summary lines 3-23 | `74bc258b46e09ca1d3bf192e62f413424cf6a73d168214012f97093fa6ce752f` |
| `evidence/trifusion_v20_preregistration_20260905.json` | preregistered hashes and AST pass lines 1-10 | `97c8c396fe30e9e812bbcd0b3ea48858da2eacc2b0bf12d06ad90d4a82b9d053` |
| `evidence/trifusion_v20_t0_20260905.json` | T0 command/result lines 1-9 | `2cc5fca9eac933236598c435e51fbff1d8d44d6ca59c311562c2b0ef29fca00f` |
| `evidence/trifusion_v20_launch_20260905.json` | launch command/source bindings lines 1-24 | `5cf829094482168765d9826859400ae05e2338bf0195096db2b320d72d424523` |
| `evidence/trifusion_v20_m0_seed42_3cea5bf.json` | M0 stage JSON, recomputation source | `5fd4922a7a7036f6905c54397809faed18387666b1df18aa39e5429cd10876a0` |
| `evidence/trifusion_v20_m0_transfer_20260905.json` | remote/local transfer integrity lines 1-17 | `2581110f7a0592ce3b71519b69fe7def51ac91e61941ffe26d6b5b849bb4f6d1` |
| `evidence/trifusion_v20_m0_run_20260905.log` | remote run log including M0 and Q1 start lines 1-101 | `5ae3ecf49e9de70caf0154060782fa0becf2caa95e9e7f4245011c23eef8a267` |
| `modeling/trifusion/signal_preserving_v8.py` | frozen baseline/tail and embedding outputs | `97a7b5fe6dab882c2ed92ed1db95b9a8f48eae1d99a3a9b145b0cdfc4b8cefbc` |
| `modeling/trifusion/signal_preserving_v8_builder.py` | V8 model construction/provenance | `8afb028957ecff1a0a26497f7d0460bc240b0f8612a3c476b52a9a6667a3049d` |
| `modeling/trifusion/criterion.py` | V8 base ID/triplet loss | raw `9a028e5711981123310f5e0d441a35b7dbecaf33f9edcec4fd3773d070cf877f`; LF-normalized hash matches recorded `0b2a6370f434828d885945d1a46dd56f6668133ceed45ef359fe71eeca63740a` |
| `modeling/trifusion/aligned_data.py` | aligned triplet dataset and sampler | `3ea362d17660483b554cb599442b6377ace020fa114969b9bdd58906fbceedd5` |
| `tools/train_signal_preserving_v17.py` | source binding dependency | `333886d16f73987accddb70b0780661bb9400b8afd04af01b57e188d34e5228d` |
| `tools/train_signal_preserving_v18.py` | source binding dependency | `f6bdda4631d710d0b6db5fe2a8df124d8dcbd294cb54adcf3ec261d523e96cf8` |
| `tools/train_signal_preserving_v19.py` | source binding dependency | `5f000d3d3abe9636cae97e292db2c63a3a20f349c125dfe4b27bd3fca95bb9c5` |
| `tools/build_v12_complete_path_oof_targets.py` | V12 OOF split/fold dependency | `fc13354b1065c46677122ee9cf63816087facca4e0e98ec5e6e2fb0dece141e4` |
| `tools/run_signal_preserving_v5.py` | training batch, evaluation helpers | `e162184f68778b4991db0f97f26c5fda273b2ad2f7c8db2bbb2d53775eb717e5` |
| `tools/audit_v17_full_gallery.py` | retrieval metric dependency | `856881f984ab8793788d291018a60046a47e309c625a65394b1a3ff4e670d8a9` |
| `tools/diagnose_v6_oracle_complementarity.py` | AP/rank metric helper | `919b624156c57f92fa75f79a06fe7c872a02d730fa8cd021ce9d7b4e498b1db2` |
| `protocols/rgbnt201_dev_v1.json` | train/dev/test split protocol | raw `f7f6f8a906c06fec4918929fb5c17abfc197986b99e017aca1d3e15831c1541d`; LF-normalized hash matches recorded `d916e7daaa1d55b179c1ec77e93128b6e6a8d1526adc9eac060ea8e733881946` |
| `modeling/trifusion/signal_preserving_v13.py` | later Q1 bootstrap dependency | `7b7c4abb220ed234608553c77aeedd5f8ef763abb04cce5b21f1bfce0f4daa62` |

## A-F checklist

### A. Labels, positive masks, sample registry, folds, source weights

Status: **PASS for M0 engineering scope**.

The cross-modal identity loss uses dataset identity labels directly as a called-code fact. `modeling/trifusion/cross_modal_identity_v20.py:21-23` constructs `positive` and `target` from the `labels` tensor, and `tools/train_signal_preserving_v20.py:146-164` passes those labels into `cross_modal_identity_loss`. The batch helper moves `images`, `labels`, and `camera_ids` from the raw training batch to CUDA and supplies a full 3-modality mask (`tools/run_signal_preserving_v5.py:571-583`). There is no M0 retrieval proxy target and no use of V12/M0 probe outputs as labels.

The positive-pair count statement combines called-code assertions, literal snapshot fields, and an independent count. The preflight loop takes 8 batches (`tools/train_signal_preserving_v20.py:143-145`), asserts 8 same-identity samples per anchor and at least one cross-camera same-identity pair (`tools/train_signal_preserving_v20.py:147-150`), and appends literal keys `positives_per_anchor_per_directed_modality_pair`, `cross_camera_identity_pairs_per_directed_modality_pair`, and `directed_modality_pairs_per_expert` (`tools/train_signal_preserving_v20.py:151-153`). In the M0 snapshot, `preflight` begins at `evidence/trifusion_v20_m0_seed42_3cea5bf.json:40`, the first fold/endpoint is `fold: 0`, `paired: true`, `endpoint: "identity_concat"` at `evidence/trifusion_v20_m0_seed42_3cea5bf.json:42-47`, and the first `positive_pair_counts` entry is literal at `evidence/trifusion_v20_m0_seed42_3cea5bf.json:1343-1347`. The count of 48 positive-pair records is independently derived by counting 3 preflight folds x 2 endpoints x 8 entries; the six endpoint arrays start at `evidence/trifusion_v20_m0_seed42_3cea5bf.json:1343`, `2883`, `4429`, `5968`, `7513`, and `9051`. The observed cross-camera count values 14, 24, 30, and 32 are independently derived from those literal entries.

Fold and source handling are bounded to the train source in this stage. The runner strict-loads source fold states only after asserting `fit_identity_ids` and `heldout_identity_ids` match the split (`tools/train_signal_preserving_v20.py:58-77`), then records those identity arrays in the endpoint `binding` (`tools/train_signal_preserving_v20.py:79-83`; first snapshot binding begins at `evidence/trifusion_v20_m0_seed42_3cea5bf.json:47`, with `fit_identity_ids` at line 55 and `heldout_identity_ids` at line 151). The 3126-record and no-overlap statement is not a literal M0 JSON field; it is an executed source-code assertion in `tools/train_signal_preserving_v20.py:247-250` and a plan contract in `refine-logs/v20/EXPERIMENT_PLAN.md:66`. I found no evidence that heldout retrieval samples entered M0 fitting. The M0 stage itself did not run heldout retrieval.

### B. Loss, targets, normalization, gradients, entropy bound, floors, ratios, capacity

Status: **PASS**.

The implemented V20 loss matches the plan. For each expert and each of the six directed cross-modality pairs, the code normalizes the source and target modal embeddings, computes dot-product logits divided by temperature 0.07, applies log-softmax, and averages cross entropy against the same-identity soft target. Same-modality pairs are explicitly skipped. The target includes all same-identity samples in the batch, which gives 8 positives per anchor under the K=8 sampler contract. The loss is FP32-only in the V20 runner path.

The base objective remains the V8 ID/triplet objective: fused cross-entropy and batch-hard triplet, plus per-expert cross-entropy and batch-hard triplet. V20 adds `0.25 * cross_modal_identity` to that base loss. No new inference parameters are created by the V20 loss.

Independent arithmetic from `evidence/trifusion_v20_m0_seed42_3cea5bf.json` reproduced the recorded M0 values:

| Check | Recomputed value | Recorded value | Difference |
|---|---:|---:|---:|
| Identity entropy floor | 0.57838292104621 | 0.57838292104621 | 0 |
| Cross-modal alignment floor | 2.0794415416798357 | 2.0794415416798357 | 0 |
| Combined floor | 1.098243306466169 | 1.098243306466169 | 0 |
| Overfit excess ratio | 0.0030658060054957735 | 0.0030658060054957735 | 0 |
| Max `base + weight * alignment - total` residual | `5.960464477539063e-08` | n/a | FP rounding only |

Capacity equality was also reproducible from the JSON: both control and experiment arms reported 98,800,141 total parameters, 7,841,292 trainable parameters, 203 trainable tensors, and 6062 MiB peak reserved CUDA memory in the M0 capacity stage.

### C. File hashes, provenance, source bindings, logs, byte-level vs normalized line endings

Status: **WARN**.

The core V20 artifacts and M0 evidence files exist locally and match the launch/M0 recorded hashes. The M0 transfer receipt reports remote/local hash agreement for both the M0 JSON and the run log. The preregistration and launch receipts bind the V20 runner, config, plan, and module hashes used for the remote run.

The M0 JSON includes `source_file_sha256` bindings for fifteen source/protocol files. I recomputed all fifteen locally. Thirteen matched raw local bytes exactly. Two had raw-byte mismatches but matched exactly after converting CRLF to LF:

| Path | Recorded hash | Local raw hash | LF-normalized finding |
|---|---|---|---|
| `modeling/trifusion/criterion.py` | `0b2a6370f434828d885945d1a46dd56f6668133ceed45ef359fe71eeca63740a` | `9a028e5711981123310f5e0d441a35b7dbecaf33f9edcec4fd3773d070cf877f` | LF-normalized hash matches recorded hash |
| `protocols/rgbnt201_dev_v1.json` | `d916e7daaa1d55b179c1ec77e93128b6e6a8d1526adc9eac060ea8e733881946` | `f7f6f8a906c06fec4918929fb5c17abfc197986b99e017aca1d3e15831c1541d` | LF-normalized hash matches recorded hash |

This is an integrity warning rather than an engineering failure because the normalized content matches the recorded source content, but strict byte-level reproducibility is not present for those two local files.

Some required runtime inputs are receipt-bound and remote-only from this local audit position: the V12 fold checkpoints, CLIP weight artifact, raw dataset, checkpoint tensors, and CUDA execution environment. I did not download remote files or inspect checkpoint tensors because the request forbids it.

### D. CUDA tests, preflight, backward, gradients, state, optimizer, AMP, parameter sharing, reload

Status: **PASS for executed M0; Q1 reload/evaluation not executed in this snapshot**.

The T0 receipt records the CUDA test command `pytest -q tests/test_trifusion_cross_modal_identity_v20.py` with exit code 0 and `3 passed`. The tests cover the entropy floor, relabeling/batch/modality order invariance, and finite gradients reaching every expert/modality.

The V20 runner implements the actual M0 checks rather than a metadata-only pass. Preflight constructs both endpoints from the same V12 fold source (`tools/train_signal_preserving_v20.py:281-285`) and asserts equality for the literal keys `initial_state_sha256`, `batch_receipts`, `all_output_sha256`, `binding`, and `positive_pair_counts` (`tools/train_signal_preserving_v20.py:288-290`). Preflight returns `initial_state_sha256`, `batch_receipts`, `all_output_sha256`, `positive_pair_counts`, `alignment_nonzero_encoder_gradients`, `baseline_no_gradient`, and `state_unchanged` (`tools/train_signal_preserving_v20.py:178-180`). In the first endpoint snapshot these appear as `initial_state_sha256` at `evidence/trifusion_v20_m0_seed42_3cea5bf.json:210`, `batch_receipts` at line 211, `all_output_sha256` at line 1333, `positive_pair_counts` at line 1343, `alignment_nonzero_encoder_gradients` at line 1385, `baseline_no_gradient` at line 1582, and `state_unchanged` at line 1583.

The batch/augmentation receipt description is hash-level, not a decoded transform log. `_raw_batch_receipt` stores literal `paths`, `sampler_indices`, `tensor_sha256`, and `metadata_sha256`, where the metadata hash covers labels, physical cameras, model camera labels, and views (`tools/train_signal_preserving_v17.py:466-489`). The first M0 receipt shows `paths` at `evidence/trifusion_v20_m0_seed42_3cea5bf.json:213`, `sampler_indices` at line 279, `tensor_sha256` at line 345, and `metadata_sha256` at line 350.

The optimizer/AMP statement is now scoped to called code and recorded fields. `fixed_steps` creates an optimizer and scaler (`tools/train_signal_preserving_v20.py:187`), calls `step` with them for `count` iterations (`tools/train_signal_preserving_v20.py:192-194`), and returns `steps`, `alignment_weight`, `losses`, `components`, `trainable_tensors`, `nonzero_gradient_tensors`, `missing_nonzero_gradients`, `frozen_state_unchanged`, and `overflow_events` (`tools/train_signal_preserving_v20.py:199-202`). It does not return optimizer/scaler state. The M0 snapshot records those fields for capacity arm 0.0 at `evidence/trifusion_v20_m0_seed42_3cea5bf.json:9305-9366`, capacity arm 0.25 at `evidence/trifusion_v20_m0_seed42_3cea5bf.json:9368-9428`, and overfit at `evidence/trifusion_v20_m0_seed42_3cea5bf.json:9431-10047`. Capacity and overfit each literally report 203 trainable tensors, 203 nonzero-gradient tensors, empty `missing_nonzero_gradients`, zero `overflow_events`, and unchanged frozen state at `evidence/trifusion_v20_m0_seed42_3cea5bf.json:9361-9365`, `9423-9427`, and `10038-10042`.

The later Q1 code path includes checkpoint save/reload/evaluate logic (`tools/train_signal_preserving_v20.py:344-358`) and paired Q1 training receipt checks (`tools/train_signal_preserving_v20.py:368-373`), but these are not credited as executed evidence here because the M0 snapshot literal `folds` field is still empty (`evidence/trifusion_v20_m0_seed42_3cea5bf.json:9294`).

### E. Folds, endpoints, batches, steps, intended query-gallery scope, decisions and chronology

Status: **PASS for stage reconciliation; WARN for any complete-scope scientific interpretation**.

The intended V20 run has two endpoints, `identity_concat` and `cross_modal_identity`, fixed in source as `ENDPOINTS` (`tools/train_signal_preserving_v20.py:30-32`) and required by the config contract (`tools/train_signal_preserving_v20.py:40-43`). M0 preflight checked all 3 folds and both endpoints by iterating `splits` and `ENDPOINTS` (`tools/train_signal_preserving_v20.py:281-285`), then writing each fold result into the snapshot `preflight` array (`tools/train_signal_preserving_v20.py:290-291`; snapshot `preflight` begins at `evidence/trifusion_v20_m0_seed42_3cea5bf.json:40`).

The step-count statements are literal, but “fixed batch” applies only to the overfit stage. Capacity calls `fixed_steps(..., count=8, fixed=False)` for weights 0.0 and 0.25 (`tools/train_signal_preserving_v20.py:293-300`), so `fixed_steps` consumes the next loader batch on each step (`tools/train_signal_preserving_v20.py:190-194`). The snapshot records `steps: 8` and `alignment_weight: 0.0` for the control capacity arm at `evidence/trifusion_v20_m0_seed42_3cea5bf.json:9305-9308`, and `steps: 8`, `alignment_weight: 0.25` for the experiment capacity arm at `evidence/trifusion_v20_m0_seed42_3cea5bf.json:9368-9370`. Overfit calls `fixed_steps(..., count=100, fixed=True)` (`tools/train_signal_preserving_v20.py:303-304`), which selects one `fixed_batch` once (`tools/train_signal_preserving_v20.py:190`) and reuses it for each of the 100 optimizer steps (`tools/train_signal_preserving_v20.py:192-194`). The snapshot records `steps: 100` and `alignment_weight: 0.25` at `evidence/trifusion_v20_m0_seed42_3cea5bf.json:9431-9433`. This matches the plan distinction between “8 different batch” capacity steps and “fixed same real batch” 100-step overfit (`refine-logs/v20/EXPERIMENT_PLAN.md:93-100`).

The broader V20 scientific/Q1 scope was not complete in the audited snapshot. The literal snapshot keys are `status: "RUNNING"` (`evidence/trifusion_v20_m0_seed42_3cea5bf.json:3`), `evaluation_type: "real_gt_train_internal_complete_path_oof"` (`evidence/trifusion_v20_m0_seed42_3cea5bf.json:28`), `epochs_per_endpoint: 20` (`evidence/trifusion_v20_m0_seed42_3cea5bf.json:30`), `model_selection: "none_final_epoch_only"` (`evidence/trifusion_v20_m0_seed42_3cea5bf.json:31`), `dev_access_count: 0`, `official_test_access_count: 0`, `d1_executed: false`, and `next_phase_qualified: false` (`evidence/trifusion_v20_m0_seed42_3cea5bf.json:36-39`), and `folds: []` (`evidence/trifusion_v20_m0_seed42_3cea5bf.json:9294`). The earlier dev/official/D1 run-count wording was imprecise; the exact literal snapshot keys are the ones listed in the preceding sentence. Q1 source code would append to `folds` only after checkpoint reload, heldout evaluation, endpoint receipt writing, and paired Q1 checks (`tools/train_signal_preserving_v20.py:344-373`). I therefore did not assess full intended query-gallery membership, final endpoint deltas, complete-path retrieval improvement, or bootstrap significance as completed claims.

### F. Observed-stage classification and eligible claims

Status: **PASS for classification; scientific claims unsupported**.

The observed evidence supports this classification:

- **T0:** remote CUDA unit/smoke tests passed by receipt.
- **M0:** remote train-source engineering gate passed, with deterministic source bindings, paired endpoint preflight, capacity equality, gradient liveness, frozen-state checks, AMP overflow checks, and fixed-batch overfit check.
- **Q1:** code path exists and the log shows it started, but no completed Q1 fold metrics exist in the M0 snapshot.
- **Dev/official/D1/SOTA:** not run and not qualified.

Eligible claims are limited to V20 engineering feasibility and M0 readiness. V20 is not scientifically validated by these artifacts alone.

## Claim assessment

| Claim | Assessment | Basis |
|---|---|---|
| V20 implements a cross-modal same-identity alignment loss using dataset labels | Supported | Loss source derives positives directly from `labels`; batch helper supplies labels; M0 preflight records K=8 positive structure. |
| V20 adds no new inference parameters | Supported | Loss module has no parameters; runner binding reports `new_inference_parameters: 0`; V8 frozen source/tail remain source-bound. |
| V20 T0 passed on remote CUDA | Supported by receipt | T0 receipt records pytest exit 0 and 3 tests passed. I did not rerun CUDA locally. |
| V20 M0 passed | Supported | Independent replay of M0 JSON gates reproduced pass conditions. |
| Control and experiment capacity are equal | Supported | Both arms report identical total/trainable parameters, trainable tensors, and peak reserved memory. |
| Cross-modal identity overfit reached the configured 0.1 excess-ratio gate | Supported | Recomputed ratio is 0.0030658060054957735. |
| Heldout retrieval data was used for M0 fitting | Refuted for observed M0 | M0 is train-source only; split/binding checks separate fit and heldout; no retrieval folds are present. |
| Q1 retrieval improvement, complete-path benefit, or statistical win | Unsupported | M0 snapshot has `folds: []` and no completed endpoint metrics. |
| Dev65, official RGBNT201, D1, or SOTA performance | Unsupported | Recorded dev/official/D1 run counts are zero/false and plan explicitly marks these unmet. |

## Independent recomputation details

I recomputed the arithmetic using only local JSON/NumPy, as permitted by the request.

Runtime:

- Python: `3.13.12`
- NumPy: `2.5.2`

Recomputed status and constants from `evidence/trifusion_v20_m0_seed42_3cea5bf.json`:

- Schema: `v20-cross-modal-identity-main-v1`
- Repository commit: `3cea5bfc17e214b1829c020527699d939efa221d`
- Seed: `42`
- Temperature: `0.07`
- Cross-modal alignment weight: `0.25`
- Batch size/instances: `64/8`
- Classes for base identity entropy floor: `94`
- Label smoothing: `0.1`
- Identity weight sum used for the floor: `0.75`

Recomputed M0 gates:

- Preflight folds/endpoints: 3 folds x 2 endpoints.
- Preflight checked key equality: true for `initial_state_sha256`, `batch_receipts`, `all_output_sha256`, `binding`, and `positive_pair_counts` in every fold.
- Capacity control/experiment steps: 8/8.
- Capacity trainable tensors: 203/203.
- Capacity missing-gradient tensors: 0/0.
- Capacity AMP overflow count: 0/0.
- Capacity frozen-state unchanged: true/true.
- Capacity peak reserved memory: 6062 MiB for both arms, below 24576 MiB configured limit.
- Overfit steps: 100.
- Overfit trainable tensors/nonzero-gradient tensors: 203/203.
- Overfit missing-gradient tensors: 0.
- Overfit AMP overflow count: 0.
- Overfit frozen-state unchanged: true.
- Recomputed `m0_passed`: true, matching reported `m0.passed: true`.

Loss/floor arithmetic:

- Control capacity first/last total loss: 0.6110473871231079 -> 0.828916609287262.
- Experiment capacity first/last total loss: 1.886078119277954 -> 1.6901416778564453.
- Overfit total loss: 1.886078119277954 -> 1.100658655166626.
- Overfit base loss: 0.6110473871231079 -> 0.580313503742218.
- Overfit cross-modal loss: 5.100122928619385 -> 2.081380605697632.
- Recomputed combined floor: 1.098243306466169.
- Recomputed overfit excess ratio: 0.0030658060054957735.
- Maximum entropy-floor difference vs reported: 0.0.
- Maximum stored-total vs recomputed `base + weight * cross_modal_identity` residual: `5.960464477539063e-08`.

## Limitations

This audit is intentionally bounded. It does not validate scientific retrieval performance. It does not validate checkpoint tensor contents, raw images, precomputed features, retrieval distance matrices, or the remote CUDA environment beyond the local receipts. It does not run training, inference, evaluation, downloads, or network calls.

The two normalized-line-ending source matches should be treated as a provenance warning for byte-level reproducibility. They do not change the M0 arithmetic conclusion, but they mean local raw bytes are not identical to the recorded source hash for `criterion.py` and `rgbnt201_dev_v1.json`.

The M0 evidence is a stage snapshot while the overall V20 run status was `RUNNING`. Later Q1 code, checkpoint reloads, endpoint evaluation, bootstrapping, and complete-path query-gallery membership must be audited only from a completed Q1 artifact, not inferred from this M0 report.

## Final conclusion

V20 M0 passes as an engineering gate. The evidence supports that the new cross-modal identity loss is implemented, gradients flow, capacity is equal, source state remains frozen, AMP behavior is clean, and the fixed-batch overfit gate passes with independently reproduced arithmetic.

The audit remains **WARN** overall because V20 M0 is not a scientific evaluation and two local source files only match recorded hashes after LF normalization. No complete-scope retrieval, dev, official, D1, or SOTA claim is eligible from the M0 artifacts.
