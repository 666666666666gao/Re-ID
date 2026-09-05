# Experiment Audit - V24 Source Prototype Diagnosis

Date: 2026-09-06

Auditor: Codex same-family read-only audit. The request called for "GPT-5.5 xhigh", but this run was also explicitly constrained to no delegation. I did not spawn a reviewer or claim a cross-family GPT-5.5 xhigh acquittal.

Overall verdict: WARN

Integrity status: warn

Evaluation type: real_gt_seen_source_descriptive_geometry_not_unknown_identity_validation

This audit is bounded to the registered V24 read-only source prototype diagnosis. It is separate from Q1 scientific qualification, D1/dev/official evaluation, and any future training decision. I loaded no model tensors, checkpoint tensors, feature tensors or images; the independent arithmetic was limited to local JSON arrays and metadata.

## Verdicts

### A. Ground Truth And Scope: PASS

The diagnosis uses real RGBNT201 train-source identity and camera labels, then forms fold-local source populations by excluding each fold's heldout identities. The record source is traceable through `tools/run_signal_baseline_dev.py:27-54`, `tools/build_v12_complete_path_oof_targets.py:156-173`, and `tools/build_v12_complete_path_oof_targets.py:29-56`. The source diagnostic reconstructs fold records from the Q1 fold receipts, asserts no identity overlap, asserts the 2126/2075/2051 record counts and 94 source labels, and stores each source record's file, source label, camera and fit-registry identity at `tools/diagnose_v24_source_prototypes.py:175-203`.

The saved diagnostic declares the source-only evaluation type and zero heldout/dev/official access at `evidence/trifusion_v24_source_diagnostic_20260906.json:2-18`, and its final counters report 9 models, 18756 source record forwards and 153 source batch forwards at `evidence/trifusion_v24_source_diagnostic_20260906.json:850431-850433`. My JSON-array recomputation reconstructed every source manifest from Q1 gallery manifests and found 94 source identities and 108 identity-camera pairs in each fold, with 381/392/369 cross-camera source queries.

No heldout image or crossfold distance enters the source diagnosis path. Heldout retrieval is a different Q1 path in `tools/train_signal_preserving_v24.py:423-449`, while the source diagnosis iterates `split["train_records"]` at `tools/diagnose_v24_source_prototypes.py:190-237` and stores folds separately.

### B. Mathematics: PASS

I independently recomputed the saved source-record distributions, all 1692 per-identity mean rows, probability/CE identities, combined CE, positive and negative masks, positive and negative indices, cosine-margin algebra, cross-camera positive subset indices, prototype/sample gap algebra and the nine model summary rows from the supplied JSON arrays.

The executed math is traceable in `tools/diagnose_v24_source_prototypes.py:27-84` for prototype CE and source-label statistics, `tools/diagnose_v24_source_prototypes.py:87-125` for sample geometry, and `modeling/trifusion/source_prototype_v24.py:12-49` for fresh prototype construction and prototype loss. The verification script checks the same families of saved-array claims at `tools/audit_v24_source_diagnostic_arrays.py:24-140`; I reran the arithmetic independently without writing its output file.

Arithmetic results:

- Local arithmetic runtime: 0.346868299995549 seconds.
- Max distribution or per-identity mean discrepancy: 1.4210854715202004e-14.
- Max probability or FP32 algebra discrepancy: 2.987195324433145e-08.
- Claim-check failures from the local recomputation: none.
- Saved-array rows checked: 18756 source-record arrays and 1692 identity-mean rows.
- Not independently regenerated: model features, pairwise distance matrices, torch checkpoint tensors and image transformations.

The nine summary rows match the report table at `results/TRIFUSION_RGBNT201_V24_SOURCE_PROTOTYPE_DIAGNOSIS_2026-09-06.md:28-38` and the saved verifier rows at `evidence/trifusion_v24_source_diagnostic_array_verification_20260906.json:8-126`.

### C. Provenance And Execution: WARN

The local input-byte manifest is intact: all 27 requested input files matched the bytes and SHA-256 entries in `.aris/traces/experiment-audit/2026-09-06_run16/input_file_sha256.json`. The run binding is consistent across launch metadata, wrapper, original launch receipt, log, exit file and intake receipt: launch argv and hashes are at `evidence/trifusion_v24_source_diagnostic_launch_20260906.json:1-43`, the wrapper starts the recorded argv and writes launch/exit files at `evidence/trifusion_v24_source_diagnostic_launch_wrapper_20260906.py:1-12`, the log has all nine model events plus the final terminal event at `evidence/trifusion_v24_source_diagnostic_run_20260906.log:16-97`, the exit file is `0` at `evidence/trifusion_v24_source_diagnostic_exit_20260906.txt:1`, and the intake receipt maps the remote artifacts to the local evidence copies at `evidence/trifusion_v24_source_diagnostic_intake_20260906.json:1-30`.

The source diagnosis code verifies plan, Q1 summary, config, source-file hashes, signal commit/diff, checkpoint SHA, memory SHA, strict state reload, state preservation and final counters at `tools/diagnose_v24_source_prototypes.py:162-188`, `tools/diagnose_v24_source_prototypes.py:210-224`, `tools/diagnose_v24_source_prototypes.py:229-243`, and `tools/diagnose_v24_source_prototypes.py:262-270`. The Q1 terminal file verifier reports exit0 and matching remote source/checkpoint/memory files at `evidence/trifusion_v24_q1_terminal_file_verification_20260906.json:1-9`, `evidence/trifusion_v24_q1_terminal_file_verification_20260906.json:242-282`, and `evidence/trifusion_v24_q1_terminal_file_verification_20260906.json:497-512`.

The WARN is for scope, not for a found numerical discrepancy. I did not load remote binary checkpoint, memory, CLIP, feature or image artifacts locally. One listed local source file also differs byte-for-byte from the Q1 remote source receipt because of Windows CRLF line endings: local `protocols/rgbnt201_dev_v1.json` is 5685 bytes with SHA-256 `f7f6f8a906c06fec4918929fb5c17abfc197986b99e017aca1d3e15831c1541d` in the input manifest at `.aris/traces/experiment-audit/2026-09-06_run16/input_file_sha256.json:106`, while the Q1 remote receipt is 5376 bytes with SHA-256 `d916e7daaa1d55b179c1ec77e93128b6e6a8d1526adc9eac060ea8e733881946` at `evidence/trifusion_v24_q1_terminal_file_verification_20260906.json:82-86`. LF-normalizing the local file produces the remote hash, so this is a byte-provenance limitation rather than a content mismatch.

Execution counts by code reading:

- Source-diagnosis batch forward calls: 153, covering 18756 source records across nine model extractions. The code extracts one clean source pass per model at `tools/diagnose_v24_source_prototypes.py:128-141` and asserts final totals at `tools/diagnose_v24_source_prototypes.py:262-268`.
- Optimizer updates: 0 in the source diagnosis.
- Backward calls: 0 in the source diagnosis.
- Checkpoint writes: 0 in the source diagnosis.
- Source-diagnosis writes: one remote `diagnostic.json` is written after each fold and once at completion at `tools/diagnose_v24_source_prototypes.py:261` and `tools/diagnose_v24_source_prototypes.py:270`. The wrapper separately writes launch, log and exit receipts at `evidence/trifusion_v24_source_diagnostic_launch_wrapper_20260906.py:7-12`.

### D. Dead Code And Metric Type: PASS

The statistics claimed in the source report are produced by executed paths and stored in actual evidence. Prototype CE, probabilities, correctness, candidate-pair counts, nearest negative labels, per-identity means, fresh/cached prototype drift and sample geometry are stored under each model at `tools/diagnose_v24_source_prototypes.py:244-253`; the script writes the cumulative diagnostic at `tools/diagnose_v24_source_prototypes.py:261` and the final diagnostic at `tools/diagnose_v24_source_prototypes.py:270`.

Metric type classification: real dataset labels are used, but the evaluation is seen-source descriptive geometry and in-sample prototype classification. It is not an unknown-identity validation. The saved diagnostic states this at `evidence/trifusion_v24_source_diagnostic_20260906.json:4`, and the narrative discloses the boundary at `results/TRIFUSION_RGBNT201_V24_SOURCE_PROTOTYPE_DIAGNOSIS_2026-09-06.md:68-78`.

The fresh source prototypes include the scored source records, which is disclosed in the saved diagnostic at `evidence/trifusion_v24_source_diagnostic_20260906.json:18` and in the narrative at `results/TRIFUSION_RGBNT201_V24_SOURCE_PROTOTYPE_DIAGNOSIS_2026-09-06.md:70-71`. Sample positive geometry excludes query self by construction at `tools/diagnose_v24_source_prototypes.py:90-96`, and my recomputation found zero positive self-matches. The report correctly says no new source retrieval mAP evaluation was executed at `results/TRIFUSION_RGBNT201_V24_SOURCE_PROTOTYPE_DIAGNOSIS_2026-09-06.md:40-43`.

### E. Claim Scope: WARN

Supported with qualifiers: the clean seen-source arrays support that all nine source models have 100% global/environment prototype correctness and strictly positive nonself hardest-positive-minus-nearest-negative margins on the saved clean source features. This is supported by `results/TRIFUSION_RGBNT201_V24_SOURCE_PROTOTYPE_DIAGNOSIS_2026-09-06.md:7-9`, `results/TRIFUSION_RGBNT201_V24_SOURCE_PROTOTYPE_DIAGNOSIS_2026-09-06.md:28-38`, and the recomputed arrays.

The report mostly keeps the right boundary. It explicitly says this is not unknown-identity generalization evidence, does not replay strong augmentation, cannot turn cached/fresh disagreement wholly into stale-cache evidence, and cannot prove the heldout Q1 failure cause at `results/TRIFUSION_RGBNT201_V24_SOURCE_PROTOTYPE_DIAGNOSIS_2026-09-06.md:59-65` and `results/TRIFUSION_RGBNT201_V24_SOURCE_PROTOTYPE_DIAGNOSIS_2026-09-06.md:68-78`. The WARN remains because the phrase "source prototypes are sufficiently fitted" is only justified for clean, seen-source, in-sample prototype classification. It should not be reused as a causal claim about heldout generalization, XBM, hard-negative learning, EMA, temperature or refresh-period tuning.

Q1 remains failed and not upgraded by this source diagnosis. The Q1 summary status is `Q1_FAIL` at `evidence/trifusion_v24_q1_seed42_6a4ac2c.json:3`, with `next_phase_qualified: false` at `evidence/trifusion_v24_q1_seed42_6a4ac2c.json:46`; the source report repeats that boundary at `results/TRIFUSION_RGBNT201_V24_SOURCE_PROTOTYPE_DIAGNOSIS_2026-09-06.md:3-5` and `results/TRIFUSION_RGBNT201_V24_SOURCE_PROTOTYPE_DIAGNOSIS_2026-09-06.md:75-78`.

### F. Conclusion: WARN

There is no evidence of fake ground truth, self-normalized scoring, phantom rows, dead metric reporting, heldout leakage inside the source diagnosis, crossfold distance use, or undisclosed self-match in the sample geometry. The source-diagnosis saved arrays and narrative claims are internally consistent when limited to clean seen-source descriptive geometry.

The overall audit remains WARN because it is a same-family local audit, not the requested GPT-5.5 xhigh cross-family review, and because it did not independently regenerate features or distances from remote checkpoints and images. It also cannot validate unknown-identity retrieval, official/dev performance, or the Q1 scientific qualification.

## Claim Impact

- C1 all nine source models diagnosed read-only with 18756 source record forwards and 153 batch forwards: supported.
- C2 fold-local source memberships are 94 identities, 108 identity-camera pairs and 2126/2075/2051 records: supported.
- C3 source diagnosis did not train, backpropagate, write checkpoints, or touch heldout/dev/official images: supported for the executed source-diagnosis code and saved counters.
- C4 initialization/checkpoint/memory bindings and strict reload/state-preservation checks happened in the remote source diagnostic: qualified, because this audit verified code, logs and receipts but did not locally load binary tensors.
- C5 prototype CE, accuracy, distribution, per-identity mean and nine-row summary numbers: supported for saved JSON arithmetic.
- C6 clean source global/environment prototype correctness is 100% for all nine models: supported, with the seen-source and in-sample qualifier.
- C7 nonself sample positive margins are strictly positive, and the sample geometry does not use self-match positives: supported.
- C8 fresh source prototypes include the scored source records: supported and disclosed; this is an in-sample prototype classification limitation.
- C9 cached/fresh cosine differences describe disagreement between saved training memory and clean-view fresh means: qualified; do not collapse it into pure cache staleness.
- C10 sample negatives being closer than prototype negatives is descriptive association: qualified; it is not causal proof of heldout Q1 behavior and does not rule out hard-negative methods.
- C11 Q1 remains Q1_FAIL and this source diagnosis does not upgrade dev/official or next-phase qualification: supported.

## Limitations

- No GPT-5.5 xhigh reviewer was spawned, due the no-delegation constraint in the audit request.
- No model, checkpoint, memory tensor, feature tensor or image was loaded locally.
- No feature vectors or pairwise distances were independently regenerated.
- Remote binary inputs are verified through source code paths, saved receipts and terminal file verification, not by this local audit's tensor replay.
- Local raw `protocols/rgbnt201_dev_v1.json` bytes are CRLF and differ from the LF remote Q1 byte hash; LF normalization matches the remote hash.
- The conclusion applies only to the registered source diagnosis, not to Q1 scientific qualification, D1, dev, official test or future research claims.
