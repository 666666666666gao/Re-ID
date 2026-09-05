# V20 Q1 Independent Experiment Integrity Audit

Date: 2026-09-05

Auditor: GPT-5.5 xhigh independent experiment integrity auditor

Scope: completed V20 main/Q1 audit over the primary files listed in the request. I read the listed files directly and followed the called local dependency paths needed to trace labels, splits, training, evaluation, terminal verification, and report generation. I did not use prior M0/V19 verdicts, executor summaries, or desired outcomes. I did not run training, inference, remote commands, checkpoint tensor loads, image reads, feature extraction, distance recomputation, downloads, or network calls.

## Verdict

Overall verdict: **WARN**

Integrity status: **warn**

Engineering integrity: **PASS**

Scientific qualification: **FAIL**

Evaluation type: **real ground-truth, reused train-internal complete-path OOF development evaluation**.

The completed V20 Q1 artifact is internally coherent as an execution and arithmetic record. The raw Q1 summary has literal `status: "Q1_FAIL"` (`evidence/trifusion_v20_q1_seed42_3cea5bf.json:3`), `evaluation_type: "real_gt_train_internal_complete_path_oof"` (`evidence/trifusion_v20_q1_seed42_3cea5bf.json:28`), `dev_access_count: 0`, `official_test_access_count: 0`, `d1_executed: false`, and `next_phase_qualified: false` (`evidence/trifusion_v20_q1_seed42_3cea5bf.json:36-39`). I independently replayed the metric, mask, aggregate, gain, bootstrap, and gate arithmetic from the Q1 JSON arrays and six endpoint receipts. The replay exactly supports the recorded scientific failure: fused mAP decreases by `-1.0108713598678634` percentage points, two of three fold fused gains are negative, the 21-identity bootstrap 95% lower bound is `-3.8126559810990917` mAP, and four of five fixed scientific gates fail.

The integrity status remains **warn**, not pass, because provenance is not byte-perfect from this local audit position. Two source/protocol files match recorded source bindings only after LF normalization, not as current local raw bytes. In addition, during this audit the four terminal sidecars requested for extra checklist C had changed current raw hashes compared with my initial inventory. Their current hash links are coherent after re-read, but the mid-audit local drift is a real provenance timing limitation.

## A. Ground truth provenance and path isolation

Status: **PASS**

Claim impact: supports heldout/source isolation and real-label evaluation for the recorded Q1 arrays; does not support independent dev/official/test claims.

The train/source split originates from the frozen RGBNT201 dev protocol. The protocol records train/dev/test disjoint checks (`protocols/rgbnt201_dev_v1.json:2-13`), train triplet count `3126` (`protocols/rgbnt201_dev_v1.json:15-18`), and selection without test labels (`protocols/rgbnt201_dev_v1.json:126-130`). `_load_records` reads the protocol train IDs and relabeled `train_171` records, then rejects any count mismatch against the frozen protocol (`tools/build_v12_complete_path_oof_targets.py:156-173`). `build_complete_path_fold_records` constructs `train_records`, `heldout_records`, `fit_identity_ids`, `heldout_identity_ids`, and `identity_overlap` by excluding each fold's heldout identities from fitting (`tools/build_v12_complete_path_oof_targets.py:29-55`).

V20 source initialization preserves the split boundary. The runner verifies fixed config properties and source hashes before use (`tools/train_signal_preserving_v20.py:35-55`). For each fold, it loads the configured V12 Signal and expert checkpoints, asserts their stored `fit_identity_ids` and `heldout_identity_ids` match the split, strict-loads model state, and records a `binding` containing fit/heldout identity arrays, frozen-source flags, `new_inference_parameters: 0`, and parameter counts (`tools/train_signal_preserving_v20.py:58-88`). The Q1 summary contains six endpoint bindings with those literal fields; examples start at `evidence/trifusion_v20_q1_seed42_3cea5bf.json:14309`, `14405`, `14454-14462`, and repeat for the paired endpoints/folds.

The fitting loader consumes only `split["train_records"]` in both endpoint training calls (`tools/train_signal_preserving_v20.py:331-337`). The heldout manifest is constructed separately from `split["heldout_records"]` before endpoint training/evaluation records are written (`tools/train_signal_preserving_v20.py:331-333`). The evaluator is called only on `split["heldout_records"]` after checkpoint save, rebuild, strict reload, and state verification (`tools/train_signal_preserving_v20.py:344-358`). This is a called-code fact; I did not execute model code locally.

I independently rebuilt the query/gallery masks from raw `gallery_manifest` identity and camera labels in the Q1 summary. The project evaluator defines eligible queries as records whose identity appears in another camera while keeping all heldout records in the gallery (`tools/audit_v17_full_gallery.py:14-43`). The AP scorer removes only same-identity/same-camera junk from each ranked gallery and uses identity equality as the positive label (`tools/diagnose_v6_oracle_complementarity.py:81-108`). From the literal gallery manifests (`evidence/trifusion_v20_q1_seed42_3cea5bf.json:9297`, `31690`, `54628`), I derived:

| Fold | Gallery records | Gallery identities | Eligible queries | Query identities | Excluded only from query |
|---:|---:|---:|---:|---:|---:|
| 0 | 1000 | 47 | 190 | 7 | 810 |
| 1 | 1051 | 47 | 179 | 7 | 872 |
| 2 | 1075 | 47 | 202 | 7 | 873 |
| Total | 3126 | 141 disjoint heldout IDs | 571 | 21 | 2555 |

These derived counts match the literal terminal summary fields `total_gallery_records: 3126` and `total_eligible_queries: 571` (`evidence/trifusion_v20_q1_seed42_3cea5bf.json:79230-79232`) and the result report's scope statements (`results/TRIFUSION_RGBNT201_V20_CROSS_MODAL_IDENTITY_2026-09-05.md:8-10`, `130-133`). I found no evidence that any learned component trained on heldout identities within Q1. Remote-only limits remain for the original V12 checkpoints, CLIP weight, raw images, final checkpoint tensors, and remote CUDA filesystem.

## B. Losses, scores, and mathematical recomputation

Status: **PASS**

Claim impact: supports Q1_FAIL arithmetic and refutes any claim of V20 fused improvement under the fixed gate.

The new V20 loss uses dataset labels as the target source. It builds a same-identity positive matrix from `labels`, normalizes the target row-wise, iterates every expert, normalizes each expert's modal embeddings, skips same-modality pairs, and averages six directed cross-modality cross-entropies per expert (`modeling/trifusion/cross_modal_identity_v20.py:10-32`). The V20 runner computes the V8 base objective from fused, branch, and residual ID/triplet components, then adds `weight * cross_modal_identity` in FP32 (`tools/train_signal_preserving_v20.py:99-115`). The fixed config records ID/triplet coefficients, label smoothing, cross-modal weight `0.25`, and temperature `0.07` (`configs/RGBNT201/TriFusion-signal-preserving-v20-cross-modal-identity-rtx3090.yml:38-48`). The V8 criterion applies smoothed cross-entropy and normalized batch-hard triplet to fused, branch, and residual outputs (`modeling/trifusion/signal_preserving_v8.py:690-742`).

Feature normalization is separate from metric normalization. The evaluator normalizes embeddings before distance computation (`tools/train_signal_preserving_v18.py:145-150`), then `full_gallery_scores` reports mAP as the mean of AP arrays and Rank-k as the fraction of first-match ranks within fixed k (`tools/audit_v17_full_gallery.py:34-42`). I found no metric normalization by the model's own maximum, minimum, or mean score.

I independently recomputed all fold/endpoint/output mAP and Rank-1/5/10 values from the recorded `average_precision` and `first_match_rank` arrays, all aggregate metrics, matched mAP gains, fold fused gains, query-level paired changes, and scientific gates. I also independently replayed the identity-cluster bootstrap over 21 identity clusters and 10000 resamples with NumPy `default_rng(42)`, using cluster sums and sizes rather than copying the runner's concatenation loop. The runner's bootstrap implementation samples whole identity clusters with replacement and returns the 2.5 percentile (`modeling/trifusion/signal_preserving_v13.py:253-279`); the Q1 runner calls it with fused AP differences and heldout identities (`tools/train_signal_preserving_v20.py:382-384`).

Runtime for the independent replay:

- Python: `3.13.12`
- NumPy: `2.5.2`

Maximum numerical differences:

- Fold/output metric and aggregate difference vs Q1 summary/comparison/current array-audit JSON: `1.3322676295501878e-15` percentage points.
- Training history component residual `abs(mean_base_identity_triplet + alignment_weight * mean_cross_modal_identity - mean_total)`: `2.128737297546479e-08`.
- Bootstrap 95% lower-bound difference vs Q1 summary: `1.3322676295501878e-15` mAP.

Aggregate metrics recomputed from the raw arrays:

| Output | Control mAP | Candidate mAP | mAP gain | Control R1 | Candidate R1 | Control R5 | Candidate R5 | Control R10 | Candidate R10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_only | 77.487603 | 77.487603 | 0.000000 | 79.334501 | 79.334501 | 89.492119 | 89.492119 | 93.520140 | 93.520140 |
| fused | 80.206258 | 79.195387 | -1.010871 | 83.012259 | 79.334501 | 89.667250 | 87.740806 | 93.169877 | 92.294221 |
| cnn | 79.126676 | 78.116938 | -1.009739 | 82.837128 | 79.684764 | 88.791594 | 87.040280 | 90.893170 | 90.192644 |
| transformer | 78.475388 | 73.695598 | -4.779791 | 79.859895 | 75.131349 | 89.316988 | 84.588441 | 92.469352 | 89.842382 |
| mamba | 77.780907 | 79.087275 | +1.306367 | 79.334501 | 82.486865 | 89.492119 | 88.791594 | 93.345009 | 92.819615 |

The literal Q1 summary contains the same aggregate values and gains at `evidence/trifusion_v20_q1_seed42_3cea5bf.json:79140-79212`. The fold fused gains are `[-1.087608245049637, -2.53998586884299, 0.4163143098900406]`, matching `evidence/trifusion_v20_q1_seed42_3cea5bf.json:79213-79217`. The fixed scientific gates are literal at `evidence/trifusion_v20_q1_seed42_3cea5bf.json:79218-79223`: aggregate fused gain >= +1 pp failed, all folds nonnegative failed, all expert aggregates nonnegative failed, bootstrap lower bound positive failed, and fused beats candidate baseline/experts passed. The bootstrap lower bound, cluster count, and resample count are literal at `evidence/trifusion_v20_q1_seed42_3cea5bf.json:79225-79228`.

## C. File existence and provenance

Status: **WARN**

Claim impact: execution and arithmetic artifacts are locally available and coherent, but provenance is not clean enough for a pass because of normalized-line-ending source matches and observed terminal sidecar hash drift.

All requested files existed when read. The Q1 summary hash is `23c683b92ad3551e9aa07a24470e82c47565ef54b6683e00213ce7ea0bfbf522`, matching the result report evidence line (`results/TRIFUSION_RGBNT201_V20_CROSS_MODAL_IDENTITY_2026-09-05.md:115`), file verification JSON (`evidence/trifusion_v20_terminal_file_verification_20260905.json:7`), and transfer ledger line (`evidence/trifusion_v20_terminal_transfer_20260905.json:9`). The complete log hash is `978a9f98f8c2d38cb59b101c834c8838acab139c580f88e2612bb2585a00d50e`, matching file verification and transfer (`evidence/trifusion_v20_terminal_file_verification_20260905.json:9`; `evidence/trifusion_v20_terminal_transfer_20260905.json:16`). The transfer ledger reports remote/local matches for the Q1 summary, complete log, and six endpoint receipts (`evidence/trifusion_v20_terminal_transfer_20260905.json:7-59`).

I independently confirmed that each of the six standalone endpoint receipt JSON files equals its corresponding embedded endpoint object in the Q1 summary byte-parsed JSON. The file verifier performs the same object equality check at `tools/verify_v20_terminal_files.py:41-49`, and records six `receipt_equals_summary: true` rows in `evidence/trifusion_v20_terminal_file_verification_20260905.json:236-277`. The Q1 runner writes each endpoint receipt before fold-level append (`tools/train_signal_preserving_v20.py:357-360`, `373-374`).

The Q1 summary source binding contains 15 source/protocol files. I recomputed current local raw and LF-normalized hashes. Thirteen source bindings match current raw bytes. Two match only after CRLF-to-LF normalization:

| Path | Recorded source binding | Current local raw SHA256 | Finding |
|---|---|---|---|
| `modeling/trifusion/criterion.py` | `0b2a6370f434828d885945d1a46dd56f6668133ceed45ef359fe71eeca63740a` | `9a028e5711981123310f5e0d441a35b7dbecaf33f9edcec4fd3773d070cf877f` | LF-normalized hash matches recorded binding |
| `protocols/rgbnt201_dev_v1.json` | `d916e7daaa1d55b179c1ec77e93128b6e6a8d1526adc9eac060ea8e733881946` | `f7f6f8a906c06fec4918929fb5c17abfc197986b99e017aca1d3e15831c1541d` | LF-normalized hash matches recorded binding |

Additional checklist C required re-reading four terminal sidecars before finalization. Current raw bytes are coherent: current `tools/audit_v20_terminal_arrays.py` is `e6c9acc5e3a50547898ad518450a59d11b184f965d362a17d049ee498582fa4d`, and the current array-audit JSON records that exact `audit_script_sha256` (`evidence/trifusion_v20_terminal_array_audit_20260905.json:283`; script writes that field at `tools/audit_v20_terminal_arrays.py:161-165`). Current `evidence/trifusion_v20_terminal_array_audit_20260905.json` is `4579ee11406a9666d7e254c7b1092cd91e0079a27e1cbdb65621d4bbaae92b9b`, and the current comparison JSON records that exact `array_audit_sha256` (`evidence/trifusion_v20_complete_comparison_20260905.json:13068`). The current result Markdown cites the same array-audit SHA (`results/TRIFUSION_RGBNT201_V20_CROSS_MODAL_IDENTITY_2026-09-05.md:116`). Current comparison JSON records the Q1 summary SHA and generator SHA at `evidence/trifusion_v20_complete_comparison_20260905.json:13067-13069`.

The provenance warning is that those four local sidecars were not stable during this audit window. Initial inventory saw these raw hashes: array script `4d53986e6661a10c4b4e4776e066888ac7dd9a4262defb51a9dda159a67db846`, array audit JSON `82a2199064d12dd2dbecb46232653575f5f5400606d3fb3b950af5f28b311f2a`, comparison JSON `24b96241e2134ebf7fb0b7bf8be9bbf201627f9dbee5ed436c698adf6a693943`, and result Markdown `b398ea29945951161e2d368521d5fdd06553ef74466dda532a9ae7ba658821f8`. The required re-read observed current hashes `e6c9acc5e3a50547898ad518450a59d11b184f965d362a17d049ee498582fa4d`, `4579ee11406a9666d7e254c7b1092cd91e0079a27e1cbdb65621d4bbaae92b9b`, `6b512f2385f2ba93543c909864302476070dce66b6dc53db25cd719510fb1e9b`, and `ce71979e495342607dd95e237e65d26318a432c75bf59cf0f1a3c5fd4626a3f6`, respectively. This drift affects terminal audit/report sidecars, not the Q1 summary or six endpoint receipts used for independent metric replay.

## D. Execution and engineering

Status: **PASS**

Claim impact: supports that all six Q1 endpoints completed the recorded training/evaluation pipeline and that candidate/control were paired; does not independently validate checkpoint tensor contents beyond receipts and remote SHA verification.

The runner uses AdamW and CUDA GradScaler (`tools/train_signal_preserving_v20.py:91-96`). Each training step zeros gradients, computes loss terms, checks finite total loss, backpropagates through scaled loss, unscales, verifies finite gradients for every trainable parameter, steps the optimizer, updates the scaler, and reports whether scale decreased (`tools/train_signal_preserving_v20.py:118-133`). `fit_endpoint` trains 20 epochs, records sample order and first eight batch receipts, prints epoch rows, asserts zero overflow, no missing live gradients, and unchanged frozen state, then returns literal training fields including `optimizer_steps`, `history`, `alignment_weight`, `sample_order_sha256`, `first_eight_batch_receipts`, `trainable_tensors`, `nonzero_gradient_tensors`, `missing_nonzero_gradients`, and `frozen_state_unchanged` (`tools/train_signal_preserving_v20.py:205-239`).

The final checkpoint/evaluation path was reached for all six endpoints. The runner saves the non-baseline V20 state and binding, rebuilds the model, strict-loads the checkpoint, asserts reloaded state hash equals the final training state, evaluates heldout records, verifies checkpoint hash, writes the endpoint receipt, and appends the fold after paired Q1 checks (`tools/train_signal_preserving_v20.py:344-374`). The complete log has six `Q1_final` records (`evidence/trifusion_v20_complete_run_20260905.log:130`, `169`, `208`, `247`, `286`, `325`) and terminal `Q1_FAIL` aggregate output at line 326.

I checked all six endpoint receipts and embedded summary endpoints:

| Fold | Endpoint | Epochs/history rows | Optimizer steps | Alignment weight | First-eight receipts | Trainable tensors / nonzero gradients | Missing gradients | Overflow | Frozen state | Strict reload/read-only |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | identity_concat | 20 / 20 | 580 | 0.0 | 8 | 203 / 203 | 0 | 0 | true | true / true |
| 0 | cross_modal_identity | 20 / 20 | 580 | 0.25 | 8 | 203 / 203 | 0 | 0 | true | true / true |
| 1 | identity_concat | 20 / 20 | 560 | 0.0 | 8 | 203 / 203 | 0 | 0 | true | true / true |
| 1 | cross_modal_identity | 20 / 20 | 560 | 0.25 | 8 | 203 / 203 | 0 | 0 | true | true / true |
| 2 | identity_concat | 20 / 20 | 540 | 0.0 | 8 | 203 / 203 | 0 | 0 | true | true / true |
| 2 | cross_modal_identity | 20 / 20 | 540 | 0.25 | 8 | 203 / 203 | 0 | 0 | true | true / true |

The total derived training scope is 120 epoch rows and 3360 optimizer steps, matching the runner's terminal assertion (`tools/train_signal_preserving_v20.py:391-392`), the Q1 summary, the tracker (`refine-logs/v20/EXPERIMENT_TRACKER.md:17-26`), and the result report (`results/TRIFUSION_RGBNT201_V20_CROSS_MODAL_IDENTITY_2026-09-05.md:126-135`). Pairing checks passed in my independent replay: endpoint bindings, `initial_state_sha256`, `sample_order_sha256`, `first_eight_batch_receipts`, and `baseline_only` outputs match within each fold. These are literal training/output fields in the Q1 summary and receipts, with representative line anchors at `evidence/trifusion_v20_q1_seed42_3cea5bf.json:14466-14674`, `22990-22991`, and equivalent receipt anchors such as `evidence/trifusion_v20_fold_0_identity_concat_receipt.json:165-375`, `8691-8692`.

## E. Scope and selection

Status: **PASS**

Claim impact: supports a complete fixed-scope Q1 result and supports the negative Q1 gate decision; does not support rerun/scanned/adaptive claims or independent validation claims.

The source fixes the two endpoints as `identity_concat` and `cross_modal_identity` (`tools/train_signal_preserving_v20.py:30-32`), and `load_contract` asserts the config matches those endpoints, seed 42, B64/K8, 20 epochs, loss weight 0.25, temperature 0.07, no dev access, no official test, and no reranking (`tools/train_signal_preserving_v20.py:35-48`; config literals at `configs/RGBNT201/TriFusion-signal-preserving-v20-cross-modal-identity-rtx3090.yml:3-12`, `31-48`, `65-78`). The plan states two endpoints, equal model/params/init/budget/base loss, and no new inference parameters (`refine-logs/v20/EXPERIMENT_PLAN.md:20-32`), fixed 20-epoch final-checkpoint-only training (`refine-logs/v20/EXPERIMENT_PLAN.md:56-79`), and Q1 scientific gates over three folds, all five outputs, 571 eligible queries, and 21 identities (`refine-logs/v20/EXPERIMENT_PLAN.md:106-129`).

The completed evidence includes all three folds and both endpoints. The Q1 summary literal `folds` array starts at `evidence/trifusion_v20_q1_seed42_3cea5bf.json:9294`, with fold gallery manifests at `9297`, `31690`, and `54628`. It contains 30 fold/endpoint/output metric rows by construction: 3 folds x 2 endpoints x 5 outputs. The result report includes the full aggregate table (`results/TRIFUSION_RGBNT201_V20_CROSS_MODAL_IDENTITY_2026-09-05.md:15-21`), all fold/endpoint/output rows (`results/TRIFUSION_RGBNT201_V20_CROSS_MODAL_IDENTITY_2026-09-05.md:23-56`), all 21 identity mAP-gain rows (`results/TRIFUSION_RGBNT201_V20_CROSS_MODAL_IDENTITY_2026-09-05.md:58-85`), query-level paired change counts (`results/TRIFUSION_RGBNT201_V20_CROSS_MODAL_IDENTITY_2026-09-05.md:87-93`), and all scientific gate outcomes (`results/TRIFUSION_RGBNT201_V20_CROSS_MODAL_IDENTITY_2026-09-05.md:95-103`).

Selection is final-epoch only and non-adaptive in the recorded artifacts. The Q1 summary literal `model_selection` is `none_final_epoch_only` (`evidence/trifusion_v20_q1_seed42_3cea5bf.json:31`). The source aggregates only after all folds are appended and computes status from all fixed scientific checks (`tools/train_signal_preserving_v20.py:375-401`). I found no evidence of endpoint cherry-picking, best-epoch selection, temperature/weight scan, rerun selection, omitted negative identities, or omitted negative outputs in the provided artifacts. The fixed scientific result is negative, and the tracker explicitly states V20 is sealed without scans or another seed retrain (`refine-logs/v20/EXPERIMENT_TRACKER.md:36-40`).

## F. Evaluation classification and claims

Status: **PASS for classification; fixed scientific gate result FAIL**

Claim impact: engineering feasibility and completed Q1 reporting are supported; scientific advancement/generalization/SOTA claims are unsupported or refuted by the fixed gates.

Classification:

- T0/M0/Q1 fitting source: train-internal source training on the frozen 141-fit registry.
- Q1 retrieval: real ground-truth identity/camera evaluation from the train-internal complete-path OOF heldout folds.
- Dev/official/D1: not run in these artifacts. Literal fields are `dev_access_count: 0`, `official_test_access_count: 0`, `d1_executed: false`, and `next_phase_qualified: false` (`evidence/trifusion_v20_q1_seed42_3cea5bf.json:36-39`).
- Proxy status: not a synthetic/model-output ground-truth proxy. It is also not an independent dev or official RGBNT201 test because `oof_is_reused_development_qualification` is true (`evidence/trifusion_v20_q1_seed42_3cea5bf.json:29`) and the result report states it is reused train-internal OOF, not independent dev/official (`results/TRIFUSION_RGBNT201_V20_CROSS_MODAL_IDENTITY_2026-09-05.md:8`).

Supported claims:

- The V20 Q1 run completed six endpoint trainings/evaluations and produced a terminal `Q1_FAIL` result.
- The new loss did not meet the fixed fused scientific gate in this protocol.
- The candidate fused output is higher than candidate baseline/CNN/Transformer/Mamba within the same checkpoint, but this was the only passing scientific check.
- Mamba aggregate mAP improved by `+1.306367` pp, while fused, CNN, and Transformer declined.

Unsupported or refuted claims:

- Refuted: V20 cross-modal identity improves fused aggregate mAP under the fixed Q1 gate.
- Refuted: V20 qualifies for D1/dev/official continuation from this run.
- Unsupported: SOTA, official RGBNT201, independent dev65, causal mechanism, broad generalization, or “cross-modal supervision is universally ineffective.”
- Unsupported: Any claim based on checkpoint tensors/features/images/distances beyond the recorded receipts and AP/rank arrays, because those were excluded from this audit by scope.

## Limitations

I did not inspect checkpoint tensor contents, final checkpoint files, raw images, feature tensors, retrieval distances, or the remote CUDA filesystem. I did not rerun T0/M0/Q1, did not call remote commands, and did not load models. AP internals were not recomputed from distances because distance replay and feature extraction were explicitly forbidden; I recomputed metrics and gates from the recorded AP/rank arrays and rebuilt query/exclusion masks from raw heldout manifest labels/cameras.

The Q1 summary, complete log, and six endpoint receipts are backed by transfer and file-verification receipts. The V12 source checkpoints, CLIP weight, final Q1 checkpoints, raw data, and remote source tree remain receipt-bound from this local audit position.

The observed mid-audit hash drift in the terminal sidecars is retained as a provenance warning. The current sidecars are internally linked and match the independent arithmetic, but the fact that their local hashes changed during the audit prevents a clean provenance pass.

## Final conclusion

V20 Q1 is an integrity **WARN** and scientific **FAIL**.

The completed Q1 run is a coherent fixed-scope real-label train-internal OOF experiment. Engineering execution and local arithmetic replay support the recorded terminal result. The result is negative: the fixed scientific gates fail, and V20 does not qualify for D1/dev/official progression or any SOTA/generalization claim. The warning is provenance-related, not a rescue of the scientific result: two source bindings require LF normalization, several terminal sidecars changed local hashes during the audit window, and remote checkpoints/data remain receipt-bound rather than independently possessed.
