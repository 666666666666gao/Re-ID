# Experiment Audit V22 Initialization

Date: 2026-09-05

Auditor: Codex GPT-5 direct bounded audit, local read-only except this report pair

Project: TriFusion RGBNT201 V22 fixed-initialization full-gallery diagnostic

Overall verdict: WARN

Integrity status: warn

Engineering integrity: PASS with provenance limitations

Scientific qualification: FAIL_TO_PROMOTE. This diagnostic is descriptive only and does not change V22 Q1_FAIL, does not authorize D1/dev/official evaluation, and does not select an initialization checkpoint.

Evaluation type: real dataset identity/camera ground-truth labels on reused train-internal complete-path OOF held-out folds, full-gallery diagnostic. It is not an independent validation set.

## Executive Finding

I found no evidence of fake ground truth, self-normalized metrics, phantom aggregate numbers, hidden optimizer steps, checkpoint selection, or dev/official access in the bounded fixed-initialization diagnostic. Local JSON arithmetic recomputed the query masks, fold metrics, aggregate metrics, identity rows, paired-change counts, and terminal-minus-initial deltas with maximum full-precision numeric difference 2.842170943040401e-14. The top markdown table matches the evidence JSON within six-decimal rounding; the largest markdown rounding difference was 4.958391315312838e-7.

The warning is about claim boundary and possession, not arithmetic fraud. Checkpoint/image tensors were not locally loaded or possessed for this audit, and the remote tensor/checkpoint evidence is a ledger of hashes and strict-load assertions, not an independent local tensor rehash. Also, `evidence/trifusion_v22_initialization_full_comparison_20260905.json` stores paired query-change counts, not per-query paired-delta rows, although the raw initialization and terminal JSON arrays retain enough AP/Rank/query-index data to recompute those counts exactly.

## A. Ground Truth, Folds, and Full-Gallery Scope: PASS

The diagnostic traces identity and camera labels from the RGBNT201 train-only protocol and file-derived records. `protocols/rgbnt201_dev_v1.json:126-130` fixes the eligible identity selection rule and says test labels are not used. `tools/build_v12_complete_path_oof_targets.py:156-173` reads the frozen train IDs from the protocol and builds the 3126 train-record registry from `train_171`. `tools/run_signal_baseline_dev.py:27-37` parses camera IDs from filenames and records paired RGB/NI/TI paths.

Fold construction is identity-disjoint. `tools/build_v12_complete_path_oof_targets.py:29-56` splits records into fit and held-out records, returns fit and heldout identity IDs, and exposes any overlap. The diagnostic then rebuilds the three splits from the V12 fold receipts and asserts there is no identity overlap at `tools/diagnose_v22_initialization_full_gallery.py:44-48`.

The full-gallery query policy is real ReID ground truth using identity and camera labels. `tools/audit_v17_full_gallery.py:14-41` marks a query eligible only when the same identity exists in a different camera, keeps single-camera identities in the gallery as distractors, excludes same identity plus same camera from ranking, and computes AP/Rank from true identity matches. My independent recomputation from the saved `gallery_manifest` entries matched all saved query and exclusion arrays:

| Fold | Gallery | Eligible queries | Query-only exclusions | Eligible identities | Gallery identities |
|---:|---:|---:|---:|---:|---:|
| 0 | 1000 | 190 | 810 | 7 | 47 |
| 1 | 1051 | 179 | 872 | 7 | 47 |
| 2 | 1075 | 202 | 873 | 7 | 47 |
| Total | 3126 | 571 | 2555 | 21 | 141 fold-identity placements |

These counts match the derived comparison scope at `evidence/trifusion_v22_initialization_full_comparison_20260905.json:9-27` and the result report claim at `results/TRIFUSION_RGBNT201_V22_INITIALIZATION_FULL_GALLERY_2026-09-05.md:121`.

The V12 comparison boundary is correctly qualified. V12 used `select_cross_camera_records` at `tools/build_v12_complete_path_oof_targets.py:521-529`; that helper keeps only identities with at least two cameras at `tools/probe_v8_frozen_router.py:74-83`. V12 then evaluated the held-out subset of `eligible_records` at `tools/build_v12_complete_path_oof_targets.py:610-617`. The current V22 diagnostic uses all heldout records as full gallery and excludes only query-denominator records without cross-camera positives. The report's statement that V12 residual/bank about 88 mAP is not directly comparable to the current full-gallery fused value is supported by `refine-logs/v22/INITIALIZATION_FULL_GALLERY_DIAGNOSTIC_PLAN.md:7-12` and `results/TRIFUSION_RGBNT201_V22_INITIALIZATION_FULL_GALLERY_2026-09-05.md:6`.

## B. Metric Normalization and Numeric Claims: PASS

The AP/Rank code does not divide metrics by a model's own maximum, minimum, or mean prediction statistic. `tools/train_signal_preserving_v18.py:133-150` sets eval mode, runs inference mode, collects baseline/fused/expert features, applies L2 feature normalization before Euclidean distance, and calls `full_gallery_scores`. That feature normalization is not metric self-normalization. `tools/diagnose_v6_oracle_complementarity.py:91-104` ranks by distance, removes junk same-ID same-camera gallery entries, computes AP from precision at true matches, and records Rank-1 from the first valid match. Aggregate metrics are means over saved AP/rank arrays, as shown in `tools/diagnose_v22_initialization_full_gallery.py:118-126`.

I independently recomputed all fold metrics, all three-stage aggregate metrics, both terminal-minus-initial mAP maps, all 21 identity rows, and the paired AP/Rank-1 change counts from the raw saved arrays. The maximum full-precision difference was 2.842170943040401e-14. The derived comparison file reports a maximum arithmetic difference of zero at `evidence/trifusion_v22_initialization_full_comparison_20260905.json:7-8`; the small nonzero value above is ordinary floating-point summation noise from an independent local pass.

The headline numbers are supported by the raw and derived evidence. Initialization fused is 80.59032840164478 mAP and 83.71278458844134 Rank-1 at `evidence/trifusion_v22_initialization_full_gallery_20260905.json:37857-37861`. Batch-hard terminal fused is 80.64067653265477 mAP at `evidence/trifusion_v22_initialization_full_gallery_20260905.json:37890-37895`, so the mAP delta is +0.05034813100998292 at `evidence/trifusion_v22_initialization_full_gallery_20260905.json:37949-37954`. MCNL terminal fused is 78.98445428311234 mAP at `evidence/trifusion_v22_initialization_full_gallery_20260905.json:37922-37927`, so the mAP delta is -1.6058741185324408 at `evidence/trifusion_v22_initialization_full_gallery_20260905.json:37956-37961`. The markdown report repeats these values rounded to six decimals at `results/TRIFUSION_RGBNT201_V22_INITIALIZATION_FULL_GALLERY_2026-09-05.md:9-15`.

The paired-change summaries also recompute exactly. For example, batch-hard fused has 190 AP-improved, 182 AP-declined, 199 AP-equal, 10 Rank-1 repaired, and 10 Rank-1 broken queries at `evidence/trifusion_v22_initialization_full_comparison_20260905.json:4131-4137`; camera-negative fused has 194 AP-improved, 217 AP-declined, 160 AP-equal, 9 Rank-1 repaired, and 17 Rank-1 broken queries at `evidence/trifusion_v22_initialization_full_comparison_20260905.json:4168-4174`.

## C. Source, Hashes, and Artifact Possession: WARN

All 27 primary files named in the audit request exist locally and their raw SHA-256 hashes matched the audit request manifest. The diagnostic script itself asserts the script, plan, terminal summary, config, source bindings, CLIP weight, V12 summary, and supervision metadata hashes before evaluation at `tools/diagnose_v22_initialization_full_gallery.py:24-43`, and asserts source hashes again after evaluation at `tools/diagnose_v22_initialization_full_gallery.py:127-130`. The observation file reports 30 source-file checks with `matches: true` from `evidence/trifusion_v22_initialization_diagnostic_observation_210912_20260905.json:10-220`.

The source and execution bindings are coherent. The diagnostic was preregistered as frozen-not-executed with model execution false, optimizer steps zero, and checkpoint writes zero at `evidence/trifusion_v22_initialization_diagnostic_preregistration_20260905.json:2-18`. The launch record binds execution commit `824fcfde441c4277ef45a3f1d8e929120cc72295`, planned optimizer steps zero, planned checkpoint writes zero, 3126 planned triplet forwards, and 3 planned models at `evidence/trifusion_v22_initialization_diagnostic_launch_20260905.json:2-44`. The result JSON records diagnostic commit `824fcfde441c4277ef45a3f1d8e929120cc72295` and terminal execution commit `5ae096b65eb4c9987b0b8edaa7bfcd8a4cee1c36` at `evidence/trifusion_v22_initialization_full_gallery_20260905.json:5-9`.

There is one line-ending possession distinction. The local primary `protocols/rgbnt201_dev_v1.json` raw hash is `f7f6f8a906c06fec4918929fb5c17abfc197986b99e017aca1d3e15831c1541d` with CRLF line endings, while the remote execution and source ledger used `d916e7daaa1d55b179c1ec77e93128b6e6a8d1526adc9eac060ea8e733881946` at `evidence/trifusion_v22_terminal_file_verification_20260905.json:75-79`. LF-normalizing the local file gives the remote hash exactly, so this is a byte-level line-ending distinction, not a semantic protocol mismatch. The JSON content is parseable and the protocol fields used by the audit match.

The main provenance limitation is remote tensor possession. The config points to six V12 source checkpoint paths and the CLIP weight at `configs/RGBNT201/TriFusion-signal-preserving-v22-camera-negative-rtx3090.yml:49-64`. The observation ledger reports remote checkpoint and CLIP hashes as matching at `evidence/trifusion_v22_initialization_diagnostic_observation_210912_20260905.json:145-191`, but this local bounded audit did not load or rehash those tensors independently. Likewise, actual RGB/NI/TI image bytes were not loaded locally; the audit checked saved manifests, source code, and JSON arithmetic.

## D. Called Execution Path and Read-Only Behavior: PASS

The diagnostic uses the V22 model builder and V18 evaluator directly. `tools/diagnose_v22_initialization_full_gallery.py:15-19` imports `build_complete_path_fold_records`, `build_model`, and `evaluate`. The V22 builder loads the V12 Signal and expert checkpoints, checks fit/heldout IDs, strict-loads model state, and records no router/HFER/new inference parameters at `tools/train_signal_preserving_v22.py:59-89`. The V8 model path returns baseline-only, fused, or expert retrieval outputs at `modeling/trifusion/signal_preserving_v8.py:614-635`, and its auxiliary diagnostics state baseline exact prefix, frozen baseline, frozen pretrained tail, router disabled, and HFER disabled at `modeling/trifusion/signal_preserving_v8.py:670-686`.

The original remote diagnostic run was read-only with respect to training and checkpoints. The script records optimizer steps zero, checkpoint writes zero, dev access zero, official access zero, no checkpoint selection, no Q1 qualification change, and no D1 authorization at `tools/diagnose_v22_initialization_full_gallery.py:49-66`. During each fold it registers a forward pre-hook, asserts inference mode and gradients disabled, evaluates exactly the fold gallery, checks forward-call and triplet counts, checks model-state SHA unchanged, and verifies all parameter gradients remain `None` at `tools/diagnose_v22_initialization_full_gallery.py:84-97`. It totals 3126 triplet forwards, 26 model forward calls, 3126 gallery records, and 571 eligible queries at `tools/diagnose_v22_initialization_full_gallery.py:131-142` and `evidence/trifusion_v22_initialization_full_gallery_20260905.json:37964-37969`.

The only file write in the diagnostic script is the JSON output write inside `save()` at `tools/diagnose_v22_initialization_full_gallery.py:69-71`; there is no `torch.save`, optimizer construction, backward call, or checkpoint selection in this script. The observation receipt confirms exit code 0, completed folds 3, and elapsed seconds 74.08849382400513 at `evidence/trifusion_v22_initialization_diagnostic_observation_210912_20260905.json:5-8`, and records new verifier model loads and optimizer steps as zero at `evidence/trifusion_v22_initialization_diagnostic_observation_210912_20260905.json:276-277`.

## E. Retention, Scope, and Scientific Qualification: WARN

The raw evidence retains all three folds, all five outputs, query indices, exclusions, AP arrays, Rank arrays, gallery manifests, bindings, initial-state hashes, and fold-level read-only checks. Fold 0 begins at `evidence/trifusion_v22_initialization_full_gallery_20260905.json:57-58`, its binding/state/check fields are at `evidence/trifusion_v22_initialization_full_gallery_20260905.json:5060-5227`, and each output stores query indices, exclusions, AP, ranks, and metrics starting at `evidence/trifusion_v22_initialization_full_gallery_20260905.json:5233-6621`. The same structure repeats for folds 1 and 2.

The derived comparison retains all 45 fold/stage/output metric rows at `evidence/trifusion_v22_initialization_full_comparison_20260905.json:143-548` and all 21 identity rows beginning at `evidence/trifusion_v22_initialization_full_comparison_20260905.json:550-575`. It does not retain one row per query for terminal-minus-initial paired changes; `all_query_paired_changes` stores count summaries at `evidence/trifusion_v22_initialization_full_comparison_20260905.json:4122-4195`. This is a documentation/retention qualifier for any phrase implying the comparison JSON itself contains every per-query paired delta. The underlying raw initialization and terminal arrays still retain enough data to recompute all paired counts exactly.

The terminal V22 Q1 scientific gate remains failed. The terminal summary records Q1_FAIL and no dev/official/D1 at `evidence/trifusion_v22_q1_seed42_5ae096b.json:2-43`. Its matched gains are negative for fused, CNN, and Transformer, with only Mamba positive, at `evidence/trifusion_v22_q1_seed42_5ae096b.json:82130-82136`; all fold fused gains are negative at `evidence/trifusion_v22_q1_seed42_5ae096b.json:82137-82141`; all five scientific checks are false at `evidence/trifusion_v22_q1_seed42_5ae096b.json:82142-82148`; and the 95% bootstrap lower bound is -3.8769957222550886 mAP with 21 clusters and 10000 resamples at `evidence/trifusion_v22_q1_seed42_5ae096b.json:82149-82152`. The V22 experiment plan requires all five gates to pass before any next-stage qualification at `refine-logs/v22/EXPERIMENT_PLAN.md:109-127`.

## F. Claim Impact

| Claim | Verdict | Impact |
|---|---|---|
| The fixed-initialization diagnostic completed 3 models, 26 model calls, and 3126 triplet forwards with optimizer steps 0 and checkpoint writes 0. | Supported | Evidence lines: `results/TRIFUSION_RGBNT201_V22_INITIALIZATION_FULL_GALLERY_2026-09-05.md:3`, `evidence/trifusion_v22_initialization_full_gallery_20260905.json:37964-37969`. |
| Initialization fused is 80.590328 mAP/R1 83.712785 on the same full-gallery diagnostic. | Supported | Recomputed from raw AP/Rank arrays; evidence lines: `evidence/trifusion_v22_initialization_full_gallery_20260905.json:37857-37861`. |
| Batch-hard terminal fused is +0.050348 mAP versus initialization, so this bounded evidence does not support overall fused mAP decline under ordinary continuation. | Supported with boundary | Evidence lines: `evidence/trifusion_v22_initialization_full_gallery_20260905.json:37890-37895` and `evidence/trifusion_v22_initialization_full_gallery_20260905.json:37949-37954`. This does not prove future or dev/official behavior. |
| MCNL terminal fused is -1.605874 mAP versus initialization. | Supported | Evidence lines: `evidence/trifusion_v22_initialization_full_gallery_20260905.json:37922-37927` and `evidence/trifusion_v22_initialization_full_gallery_20260905.json:37956-37961`. |
| The old V12 about 88 mAP result must not be directly subtracted from current V22 fused full-gallery results. | Supported | V12 used eligible-only heldout records and residual/bank outputs, while V22 uses full heldout gallery and fused/baseline/expert outputs. Evidence lines: `tools/build_v12_complete_path_oof_targets.py:610-617`, `tools/train_signal_preserving_v22.py:337-360`, `results/TRIFUSION_RGBNT201_V22_INITIALIZATION_FULL_GALLERY_2026-09-05.md:6`. |
| The diagnostic changes V22 Q1 qualification, authorizes D1/dev/official access, or selects an initialization checkpoint. | Rejected | The diagnostic and tracker explicitly deny this at `evidence/trifusion_v22_initialization_full_gallery_20260905.json:48-54`, `refine-logs/v22/INITIALIZATION_FULL_GALLERY_DIAGNOSTIC_TRACKER.md:11-12`, and `results/TRIFUSION_RGBNT201_V22_INITIALIZATION_FULL_GALLERY_2026-09-05.md:7`. |
| The comparison JSON preserves every per-query paired delta row. | Needs qualifier | It preserves paired-count summaries, while raw arrays preserve enough to recompute query-level changes. Evidence lines: `evidence/trifusion_v22_initialization_full_comparison_20260905.json:4122-4195`. |
| This diagnostic proves a unique mechanism or camera-causal failure reason. | Unsupported | The result report correctly says it does not prove a unique failure cause at `results/TRIFUSION_RGBNT201_V22_INITIALIZATION_FULL_GALLERY_2026-09-05.md:5` and `results/TRIFUSION_RGBNT201_V22_INITIALIZATION_FULL_GALLERY_2026-09-05.md:128`. |

## Recomputed Values

| Stage | Output | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---|---:|---:|---:|---:|
| initialization | baseline_only | 77.48760311601094 | 79.33450087565674 | 89.49211908931699 | 93.52014010507881 |
| initialization | fused | 80.59032840164478 | 83.71278458844134 | 88.96672504378283 | 92.6444833625219 |
| initialization | cnn | 79.31987375374185 | 82.31173380035027 | 88.61646234676007 | 90.54290718038528 |
| initialization | transformer | 79.13635154968365 | 83.36252189141857 | 91.06830122591944 | 95.44658493870402 |
| initialization | mamba | 78.78752599990524 | 81.26094570928196 | 88.2661996497373 | 91.06830122591944 |
| batch_hard_residual | fused | 80.64067653265477 | 83.71278458844134 | 90.01751313485113 | 93.52014010507881 |
| camera_negative_residual | fused | 78.98445428311234 | 82.31173380035027 | 90.71803852889667 | 94.04553415061297 |

Terminal-minus-initial mAP deltas:

| Endpoint | baseline_only | fused | cnn | transformer | mamba |
|---|---:|---:|---:|---:|---:|
| batch_hard_residual | 0.0 | 0.05034813100998292 | 0.7295644789192295 | 0.5398503019029732 | -1.0445320268367908 |
| camera_negative_residual | 0.0 | -1.6058741185324408 | -0.16774782055459525 | -3.725590045522779 | -0.72690743438379 |

Audit arithmetic runtime for the final full local JSON pass was 0.16530100000090897 seconds using `E:\python.exe` 3.12.6 and standard JSON/integer/float arithmetic. The broken first `python` shim returned `No pyvenv.cfg file`, so it was not used for the audit arithmetic.

## Primary File Hashes

All audit-request primary files were present and matched the request manifest. Most local primary files are LF-only. `protocols/rgbnt201_dev_v1.json` is CRLF locally with raw hash `f7f6f8a906c06fec4918929fb5c17abfc197986b99e017aca1d3e15831c1541d`; its LF-normalized hash is `d916e7daaa1d55b179c1ec77e93128b6e6a8d1526adc9eac060ea8e733881946`, matching the remote source ledger.

Key raw evidence hashes:

| File | SHA-256 |
|---|---|
| `evidence/trifusion_v22_initialization_full_gallery_20260905.json` | `21a73baacca91834eb5f47ec0c129731cfdb42ff92a5b90c2d712bef40f334ca` |
| `evidence/trifusion_v22_initialization_full_gallery_20260905.log` | `756aa7a7eb67c8bde7c1ce41d62274677a62522cd01864bb4549f45d362c2a55` |
| `evidence/trifusion_v22_initialization_full_comparison_20260905.json` | `a7a42bcda5ec01ef9cf55952a7d0a8947d0dad0fcc2f6d3135861c8130d589fa` |
| `evidence/trifusion_v22_q1_seed42_5ae096b.json` | `b8cd7db81efc3827a91d165d47e001073785420baf8ddfc8507a2eead9c3d6a3` |
| `evidence/trifusion_v12_complete_path_oof_seed42.json` | `9105b86a9c079c44843e9b118a21599746a87d279370341b8cf8d26bd18a8b69` |

## Limitations

- This audit did not load models, checkpoints, tensors, images, GPU state, or remote files. It verified local primary files, source code, saved manifests, JSON arrays, log summaries, and remote-ledger hash records.
- Remote V12 checkpoint files, the CLIP weight, and actual image bytes remain remote-ledger-only for this audit. Their reported hashes are consistent with the supplied evidence, but they were not independently rehashed from local possession.
- The metric recomputation starts from saved AP/Rank arrays and gallery manifests. It does not recompute embeddings or distances from raw images.
- This is a reused train-internal complete-path OOF diagnostic on RGBNT201 fit identities. It is not new independent validation, not official-test evidence, and not a SOTA or deployment claim.
- The derived comparison JSON stores paired-change count summaries rather than explicit per-query delta rows. The raw initialization and terminal JSON files retain enough arrays to recompute those summaries exactly.
