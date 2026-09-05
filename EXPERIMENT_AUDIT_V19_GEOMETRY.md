# EXPERIMENT_AUDIT_V19_GEOMETRY

Date: 2026-09-05  
Auditor: gpt-5.5-xhigh-independent-read-only  
Request: `C:/Users/gb/.trifusion_github_publish_22c3bee/.aris/traces/experiment-audit/2026-09-05_run05/AUDIT_REQUEST.txt`

## Conclusion

`overall_verdict`: **warn**  
`integrity_status`: **warn**  
`engineering_integrity`: **pass**  
`scientific_qualification`: **fail**  
`evaluation_type`: **read_only_train_source_and_reused_oof_postmortem_not_new_validation**  
`scientific_status`: **Q1_FAIL unchanged**

I read the run05 request and the listed primary artifacts directly. I did not use executor summaries as evidence, did not launch training or evaluation, did not run model inference, did not load checkpoint tensors, did not read images/features/distances, and did not download or run remote work. The independent arithmetic scope was local JSON/NumPy replay from the raw diagnostic JSON arrays.

The V19 geometry diagnostic has read-only engineering integrity. The raw diagnostic reports all six final models strict-reloaded, all heldout five-output metric arrays exactly equal to the original Q1 arrays, source and heldout model states unchanged, optimizer steps 0, checkpoint writes 0, dev access 0, and official access 0. The transfer receipt binds the 47,990,970-byte diagnostic JSON and log SHA, and the summary binds the diagnostic SHA and summarizer SHA.

The diagnostic arithmetic is reproducible. I independently recomputed:

- all source and heldout AP/Rank aggregate metrics for both endpoints and five outputs;
- all 60 fold-level metric rows from stored AP/rank arrays;
- all 14 aggregate source classification head correctness and mean CE rows, plus the 42 fold-head rows;
- all 108 endpoint/scope/expert/modality-pair 3x3 geometry rows and all 24 grouped geometry rows;
- all five paired heldout deployment-distance change rows over 571 matched queries;
- source/heldout membership counts, fold source mappings, original-identity membership counts, and Q1 heldout score equality.

The recomputed metric, geometry, and paired-distance values match the terminal summary with max absolute difference **0.0**. Source classification aggregates match the summary exactly; fold-record CE replay differs by at most **9.68037e-08**, consistent with recorded per-fold floating-point rounding.

The scientific interpretation remains bounded. The diagnostic supports a descriptive postmortem: source identities are fully fitted while heldout identity performance remains limited, and cross-modality residual geometry is weak or negative across many source and heldout modality pairs. It does not prove a causal failure mechanism, does not validate a prospective alignment loss, and does not change V19 Q1_FAIL. It does not support D1, dev, official, public-test, or SOTA claims.

The integrity status is **warn** rather than pass because the diagnostic run itself depended on remote-only checkpoint and image bytes. The local audit verifies the JSON/log/report/receipt arithmetic and source-path evidence, but the large model/data bytes remain receipt-bound.

## Primary artifact hashes

| Artifact | SHA-256 |
|---|---|
| `docs/V19_GENERALIZATION_DIAGNOSIS_PROTOCOL_2026-09-05.md` | `fbaffb040da21e9f07917da1e37ddd579e19237b595282eedeaf55c8659bbd73` |
| `tools/diagnose_v19_generalization_geometry.py` | `53d94c54d40ed139e6f0230d9da28d07b2f604839976cc57aabcab7b084f7ceb` |
| `tools/summarize_v19_generalization_geometry.py` | `f7a3162a04edd39b3ee32e994a85f6534e39c7717751e61c6a0941d0e204a015` |
| `evidence/trifusion_v19_generalization_geometry_20260905.json` | `0e40093688ed568b7e0584672e4a74098c5fba4e57df06fba4bab1b6405adbe6` |
| `evidence/trifusion_v19_generalization_geometry_summary_20260905.json` | `0564a600c4aca9d5597355fa26e468413c8be7d53e1c391bfaeae4d407706b21` |
| `evidence/trifusion_v19_generalization_geometry_20260905.log` | `455d80583f686101f371b51b3f545434ab4fc8cee26392e58ab541c53a968fb2` |
| `evidence/trifusion_v19_generalization_diagnosis_launch_20260905.json` | `dc6e79ddf06857a6b0f2a77e39640e0db7c12d31aba1d497c6f7b8dffd82542c` |
| `evidence/trifusion_v19_generalization_transfer_receipt_20260905.json` | `69c701b761ac6f4f9667a93832553b42fe45ffcb044d4f2f6310dc87b5fc15dc` |
| `results/TRIFUSION_RGBNT201_V19_GENERALIZATION_DIAGNOSIS_2026-09-05.md` | `2c2d4e21bee9ddc272fb18947520985f2e1abf7ca7a9f1a0ad27d7f33f419d8a` |
| `evidence/trifusion_v19_q1_seed42_4b749cd.json` | `e0c9c2e0683c934fd65ae594186d89452c9786e203e1f4b1a9b7612505316d59` |
| `tools/train_signal_preserving_v19.py` | `5f000d3d3abe9636cae97e292db2c63a3a20f349c125dfe4b27bd3fca95bb9c5` |
| `tools/train_signal_preserving_v18.py` | `f6bdda4631d710d0b6db5fe2a8df124d8dcbd294cb54adcf3ec261d523e96cf8` |
| `tools/audit_v17_full_gallery.py` | `856881f984ab8793788d291018a60046a47e309c625a65394b1a3ff4e670d8a9` |
| `modeling/trifusion/signal_preserving_v8.py` | `97a7b5fe6dab882c2ed92ed1db95b9a8f48eae1d99a3a9b145b0cdfc4b8cefbc` |
| `tools/build_v12_complete_path_oof_targets.py` | `fc13354b1065c46677122ee9cf63816087facca4e0e98ec5e6e2fb0dece141e4` |
| `refine-logs/v19/EXPERIMENT_TRACKER.md` | `a47ef301134e9264752e40c4d80f29a6b72615ef7d43188a5eed055ea2eb345f` |

## Independent recomputation results

### Source and heldout retrieval metrics

All source rows are fitted-source diagnostics. They are not independent generalization evidence.

| Endpoint | Scope | Output | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---|---|---:|---:|---:|---:|
| frozen_private_tail | source | baseline_only | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| frozen_private_tail | source | fused | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| frozen_private_tail | source | cnn | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| frozen_private_tail | source | transformer | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| frozen_private_tail | source | mamba | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| frozen_private_tail | heldout | baseline_only | 77.487603 | 79.334501 | 89.492119 | 93.520140 |
| frozen_private_tail | heldout | fused | 80.240792 | 83.187391 | 90.017513 | 93.870403 |
| frozen_private_tail | heldout | cnn | 79.915105 | 84.763573 | 88.966725 | 91.593695 |
| frozen_private_tail | heldout | transformer | 78.150546 | 82.136602 | 90.542907 | 92.994746 |
| frozen_private_tail | heldout | mamba | 77.801980 | 78.984238 | 89.316988 | 94.045534 |
| trained_private_tail | source | baseline_only | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| trained_private_tail | source | fused | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| trained_private_tail | source | cnn | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| trained_private_tail | source | transformer | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| trained_private_tail | source | mamba | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| trained_private_tail | heldout | baseline_only | 77.487603 | 79.334501 | 89.492119 | 93.520140 |
| trained_private_tail | heldout | fused | 80.496828 | 84.238179 | 89.842382 | 93.695271 |
| trained_private_tail | heldout | cnn | 80.054797 | 84.238179 | 90.367776 | 92.469352 |
| trained_private_tail | heldout | transformer | 79.331729 | 83.187391 | 89.141856 | 92.469352 |
| trained_private_tail | heldout | mamba | 77.379156 | 79.509632 | 89.667250 | 93.870403 |

Replay counts and checks:

| Item | Value |
|---|---:|
| Fold metric rows recomputed | 60 |
| Aggregate metric rows recomputed | 20 |
| Max absolute metric difference vs reported arrays/summary | 0.0 |
| Q1 heldout score arrays equal for all folds/endpoints/outputs | true |

### Source classification

I recomputed labels from each source gallery manifest and compared them with the stored per-sample predictions. All 14 aggregate endpoint/head rows are 100% accurate and match the summary.

| Endpoint | Head | Correct / total | Accuracy | Mean smoothed CE |
|---|---|---:|---:|---:|
| frozen_private_tail | fused | 6252 / 6252 | 100.000000 | 0.851722 |
| frozen_private_tail | cnn | 6252 / 6252 | 100.000000 | 0.836652 |
| frozen_private_tail | transformer | 6252 / 6252 | 100.000000 | 0.838168 |
| frozen_private_tail | mamba | 6252 / 6252 | 100.000000 | 0.835432 |
| frozen_private_tail | residual_cnn | 6252 / 6252 | 100.000000 | 0.850777 |
| frozen_private_tail | residual_transformer | 6252 / 6252 | 100.000000 | 0.846994 |
| frozen_private_tail | residual_mamba | 6252 / 6252 | 100.000000 | 0.847294 |
| trained_private_tail | fused | 6252 / 6252 | 100.000000 | 0.851222 |
| trained_private_tail | cnn | 6252 / 6252 | 100.000000 | 0.836807 |
| trained_private_tail | transformer | 6252 / 6252 | 100.000000 | 0.836646 |
| trained_private_tail | mamba | 6252 / 6252 | 100.000000 | 0.836359 |
| trained_private_tail | residual_cnn | 6252 / 6252 | 100.000000 | 0.850241 |
| trained_private_tail | residual_transformer | 6252 / 6252 | 100.000000 | 0.844807 |
| trained_private_tail | residual_mamba | 6252 / 6252 | 100.000000 | 0.848283 |

Replay counts and checks:

| Item | Value |
|---|---:|
| Aggregate classification rows recomputed | 14 |
| Fold-head classification rows recomputed | 42 |
| Max aggregate difference vs summary | 0.0 |
| Max fold-record CE difference | 9.68037e-08 |

### Scope and source/heldout membership

| Fold | Source mapping | Source gallery | Source queries | Source cross-camera IDs | Heldout gallery | Heldout queries | Heldout cross-camera IDs | Fit/heldout overlap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 94 | 2126 | 381 | 14 | 1000 | 190 | 7 | 0 |
| 1 | 94 | 2075 | 392 | 14 | 1051 | 179 | 7 | 0 |
| 2 | 94 | 2051 | 369 | 14 | 1075 | 202 | 7 | 0 |

Additional membership checks:

| Item | Value |
|---|---:|
| Source memberships across three fold models | 6252 |
| Source unique physical records after original-ID remap | 3126 |
| Heldout memberships across folds | 3126 |
| Heldout unique physical records | 3126 |
| Original identities appearing as source in exactly two folds | 141 |
| Original identities appearing as heldout in exactly one fold | 141 |
| Max fit/heldout identity overlap per fold | 0 |
| Strict reload/state unchanged in all endpoints | true |

### Grouped modality geometry

These values are cosine statistics on existing unit residual vectors. They are descriptive geometry diagnostics, not new retrieval outputs or a model-selection scan. I independently recomputed all 108 individual 3x3 modality-pair rows and all 24 grouped rows. The table below shows the grouped rows; the full audited result report lists all 3x3 directions.

| Endpoint | Scope | Expert | Group | Same-instance cos | Positive mean cos | Nearest positive cos | Nearest negative cos | Nearest margin | Negative at least as close |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| frozen_private_tail | source | cnn | same_modality | 1.000000 | 0.821023 | 0.865857 | 0.703815 | 0.162041 | 1.868068 |
| frozen_private_tail | source | cnn | different_modality | 0.289571 | 0.268804 | 0.351527 | 0.540962 | -0.189435 | 73.890835 |
| frozen_private_tail | source | transformer | same_modality | 1.000000 | 0.843354 | 0.885809 | 0.694532 | 0.191277 | 0.875657 |
| frozen_private_tail | source | transformer | different_modality | 0.259927 | 0.243841 | 0.318324 | 0.508334 | -0.190010 | 72.942207 |
| frozen_private_tail | source | mamba | same_modality | 1.000000 | 0.818856 | 0.863553 | 0.707579 | 0.155974 | 3.998832 |
| frozen_private_tail | source | mamba | different_modality | 0.273442 | 0.252096 | 0.332925 | 0.550812 | -0.217887 | 78.677758 |
| frozen_private_tail | heldout | cnn | same_modality | 1.000000 | 0.517660 | 0.621415 | 0.673746 | -0.052331 | 58.844133 |
| frozen_private_tail | heldout | cnn | different_modality | 0.242257 | 0.165064 | 0.272613 | 0.511239 | -0.238626 | 90.338587 |
| frozen_private_tail | heldout | transformer | same_modality | 1.000000 | 0.488148 | 0.598927 | 0.666625 | -0.067698 | 61.120841 |
| frozen_private_tail | heldout | transformer | different_modality | 0.222806 | 0.136542 | 0.235646 | 0.474360 | -0.238715 | 92.148278 |
| frozen_private_tail | heldout | mamba | same_modality | 1.000000 | 0.525483 | 0.628026 | 0.679142 | -0.051115 | 58.960887 |
| frozen_private_tail | heldout | mamba | different_modality | 0.222544 | 0.147219 | 0.248188 | 0.513006 | -0.264818 | 94.016346 |
| trained_private_tail | source | cnn | same_modality | 1.000000 | 0.835726 | 0.878872 | 0.709499 | 0.169373 | 1.722125 |
| trained_private_tail | source | cnn | different_modality | 0.286285 | 0.266877 | 0.348316 | 0.544885 | -0.196569 | 74.197315 |
| trained_private_tail | source | transformer | same_modality | 1.000000 | 0.851885 | 0.892388 | 0.693197 | 0.199191 | 0.904845 |
| trained_private_tail | source | transformer | different_modality | 0.256580 | 0.241895 | 0.312031 | 0.500812 | -0.188781 | 71.555750 |
| trained_private_tail | source | mamba | same_modality | 1.000000 | 0.825635 | 0.869808 | 0.701125 | 0.168683 | 2.626970 |
| trained_private_tail | source | mamba | different_modality | 0.263698 | 0.242713 | 0.322266 | 0.545172 | -0.222905 | 78.108581 |
| trained_private_tail | heldout | cnn | same_modality | 1.000000 | 0.528937 | 0.634277 | 0.680235 | -0.045958 | 58.960887 |
| trained_private_tail | heldout | cnn | different_modality | 0.239747 | 0.160428 | 0.266306 | 0.516598 | -0.250292 | 91.856392 |
| trained_private_tail | heldout | transformer | same_modality | 1.000000 | 0.489085 | 0.599688 | 0.667453 | -0.067764 | 62.521891 |
| trained_private_tail | heldout | transformer | different_modality | 0.231340 | 0.147524 | 0.242247 | 0.473694 | -0.231447 | 90.951547 |
| trained_private_tail | heldout | mamba | same_modality | 1.000000 | 0.522000 | 0.627512 | 0.678949 | -0.051436 | 59.136019 |
| trained_private_tail | heldout | mamba | different_modality | 0.223294 | 0.149974 | 0.251493 | 0.507915 | -0.256422 | 93.286632 |

Replay counts and checks:

| Item | Value |
|---|---:|
| 3x3 pair rows recomputed | 108 |
| Grouped rows recomputed | 24 |
| All numeric geometry finite | true |
| Max 3x3 pair difference vs summary | 0.0 |
| Max grouped difference vs summary | 0.0 |

### Paired heldout deployment-distance changes

Values are trained private tail minus frozen private tail over matched heldout queries. Lower positive distance is better; higher negative distance is better.

| Output | Positive-distance delta | Negative-distance delta | Nearest-margin delta |
|---|---:|---:|---:|
| baseline_only | 0.000000000 | 0.000000000 | 0.000000000 |
| fused | -0.002814117 | -0.000990033 | 0.001824084 |
| cnn | -0.008615169 | -0.005311347 | 0.003303822 |
| transformer | 0.000143808 | 0.001325416 | 0.001181608 |
| mamba | 0.000291435 | 0.002920868 | 0.002629433 |

Replay checks:

| Item | Value |
|---|---:|
| Outputs recomputed | 5 |
| Matched queries per output | 571 |
| Max difference vs summary | 0.0 |

## A-F checklist

### A. Dataset identity/camera and source/heldout GT provenance — PASS

The diagnostic uses dataset identity and camera labels from the same frozen train-internal complete-path records as Q1. Source records are fold-local relabeled fit identities; heldout records keep original identities. The protocol explicitly requires preserving the original registry mapping so source labels are not confused with heldout global encodings.

Evidence:

- `docs/V19_GENERALIZATION_DIAGNOSIS_PROTOCOL_2026-09-05.md:13-20` fixes strict Q1 dependency verification, heldout re-evaluation, and source classification/retrieval diagnostics.
- `docs/V19_GENERALIZATION_DIAGNOSIS_PROTOCOL_2026-09-05.md:25-27` requires independent source/heldout scopes and preserving source registry mappings.
- `tools/build_v12_complete_path_oof_targets.py:29-55` splits fit and heldout identities and relabels only fit records.
- `tools/build_v12_complete_path_oof_targets.py:156-173` loads records from the frozen 141-fit train registry.
- `tools/diagnose_v19_generalization_geometry.py:81-87` derives identities, cameras, and eligible cross-camera queries from records.
- `tools/diagnose_v19_generalization_geometry.py:143-172` rebuilds each fold split, checks no identity overlap, and verifies source/heldout identity counts.
- `evidence/trifusion_v19_generalization_geometry_20260905.json:119-123`, `66775-66779`, `461326-461330`, `525258-525262`, `921105-921109`, and `992039-992043` record heldout/source identity and query counts for all folds.

Audit result: source and heldout provenance pass. Source results remain training-fit diagnostics, not independent validation.

### B. Genuine feature/metric normalization and denominators — PASS

The audited source normalizes embeddings with `F.normalize`, computes pairwise distances/cosines, and reports AP/Rank, CE, and cosine means from dataset identity/camera masks and feature arrays. I found no metric denominator derived from the model’s own output maximum, minimum, or mean in the audited formulas. The source 100% values come from exact source retrieval/classification correctness over fit identities, not self-normalized scores.

Evidence:

- `tools/diagnose_v19_generalization_geometry.py:31-57` computes modality cosine geometry from residual embeddings and identity/camera masks.
- `tools/diagnose_v19_generalization_geometry.py:87-101` normalizes feature embeddings, computes `torch.cdist`, calls `full_gallery_scores`, and records modality geometry.
- `tools/diagnose_v19_generalization_geometry.py:105-113` computes source classification predictions and label-smoothed CE against source labels.
- `tools/summarize_v19_generalization_geometry.py:27-33` recomputes mAP and Rank-k from average precision and first-match-rank arrays.
- `tools/summarize_v19_generalization_geometry.py:39-58` recomputes modality geometry means, margins, and negative-at-least-as-close percentages.
- `tools/audit_v17_full_gallery.py:14-43` defines eligible query filtering, junk same-identity/same-camera handling, AP arrays, and Rank-k metrics.
- `tools/train_signal_preserving_v18.py:133-152` shows the original Q1 evaluation path used the same normalized embedding and full-gallery scoring logic.

Audit result: metric and geometry denominator checks pass.

### C. Primary result existence, hashes, numerical claims, and receipt consistency — WARN

All requested local artifacts exist and were read. The raw diagnostic JSON SHA matches the summary and transfer receipt. The log shows six fold/endpoint completions and final completion. The summary records diagnostic SHA, summary-script SHA, 18,756 source+heldout triplet forwards, and Q1_FAIL unchanged. Numerical claims in the result report match the independent replay.

Evidence:

- `evidence/trifusion_v19_generalization_geometry_20260905.json:2-13` records final diagnostic status, scope, Q1 SHA, protocol SHA, script SHA, optimizer steps 0, checkpoint writes 0, dev access 0, and official access 0.
- `evidence/trifusion_v19_generalization_transfer_receipt_20260905.json:3-15` records final status, diagnostic SHA/bytes, log SHA, original Q1 SHA unchanged, source/script/protocol SHA, optimizer steps 0, checkpoint writes 0, dev 0, official 0, and elapsed seconds.
- `evidence/trifusion_v19_generalization_geometry_20260905.log:16-67` records all six endpoint completions and final completion.
- `evidence/trifusion_v19_generalization_geometry_summary_20260905.json:1545-1548` records 18,756 triplet forwards, Q1_FAIL unchanged, diagnostic SHA, and summary-script SHA.
- `results/TRIFUSION_RGBNT201_V19_GENERALIZATION_DIAGNOSIS_2026-09-05.md:3-10` records the central numerical claims and read-only scope.
- `results/TRIFUSION_RGBNT201_V19_GENERALIZATION_DIAGNOSIS_2026-09-05.md:243-249` records the diagnostic JSON SHA and warns that large weights/images remain server-side.

The check is WARN because checkpoint and image bytes were not independently read locally. The local audit is arithmetic- and receipt-complete, not a byte-complete reproduction of the remote diagnostic run.

### D. Actual feature/evaluation/model reload calls and state/write protections — PASS

The diagnostic script is a real feature/evaluation replay script. It verifies Q1 and protocol hashes, rebuilds each source/heldout split, strict-reloads all six endpoint checkpoints, validates final state SHA, re-extracts heldout and source features, asserts heldout metrics match Q1 arrays, extracts source classification logits, and asserts model state is unchanged after extraction. The run records optimizer steps 0 and checkpoint writes 0.

Evidence:

- `tools/diagnose_v19_generalization_geometry.py:118-140` verifies Q1/protocol/source hashes, loads the contract, records read-only scope, and initializes optimizer/checkpoint/dev/official counters as 0.
- `tools/diagnose_v19_generalization_geometry.py:143-177` iterates three folds and both endpoints, checks checkpoint SHA, strict-loads state, checks final model state SHA, extracts heldout/source diagnostics, verifies heldout arrays equal Q1, and stores strict reload status.
- `tools/diagnose_v19_generalization_geometry.py:60-80` performs inference-mode extraction and asserts model state is unchanged.
- `tools/diagnose_v19_generalization_geometry.py:184-188` writes only diagnostic JSON and final status.
- `evidence/trifusion_v19_generalization_geometry_20260905.log:16-67` records six endpoint completions and final completion.
- `evidence/trifusion_v19_generalization_geometry_20260905.json:114-117`, `230667-230670`, `461321-461324`, `691160-691163`, `921100-921103`, and `1150603-1150606` record strict reload and exact heldout array replay for all endpoints.

Audit result: execution-path and state/write protection checks pass within receipt-bound remote byte limits.

### E. All-endpoint, all-scope, all-query, all-modality-pair, source/heldout scope — PASS

The diagnostic covers all three folds, both endpoints, source and heldout scopes, all five existing outputs, all eligible queries, all source classification heads, and all 3x3 modality pairs for each expert. It does not create a new modality-subset retrieval benchmark or scan fusion weights.

Evidence:

- `docs/V19_GENERALIZATION_DIAGNOSIS_PROTOCOL_2026-09-05.md:21-24` fixes all 3x3 modality-pair geometry and prohibits modality subset retrieval, fusion-weight adjustment, modality/layer/head scans, and cherry-picking.
- `docs/V19_GENERALIZATION_DIAGNOSIS_PROTOCOL_2026-09-05.md:25-32` fixes 18,756 triplet forwards, independent source/heldout scopes, optimizer0, checkpoint writes0, and no dev/official.
- `tools/diagnose_v19_generalization_geometry.py:25-28` defines three experts, five outputs, two endpoints, and three modalities.
- `tools/summarize_v19_generalization_geometry.py:20-61` aggregates both endpoints, both scopes, all five outputs, all 3x3 pairs, and grouped geometry.
- `tools/summarize_v19_generalization_geometry.py:73-85` computes paired heldout distance changes and records 18,756 forwards and Q1_FAIL unchanged.
- `results/TRIFUSION_RGBNT201_V19_GENERALIZATION_DIAGNOSIS_2026-09-05.md:12-17` records source6252/1142 queries, heldout3126/571 queries, source identity reuse across fold models, and no mixed galleries.
- `results/TRIFUSION_RGBNT201_V19_GENERALIZATION_DIAGNOSIS_2026-09-05.md:64-70` states all grouped modality geometry was preserved and not used as new retrieval outputs.
- `results/TRIFUSION_RGBNT201_V19_GENERALIZATION_DIAGNOSIS_2026-09-05.md:99-210` lists the full 3x3 directional geometry table.

Audit result: scope passes.

### F. Evaluation/diagnostic type and claim qualification — PASS

The diagnostic is a read-only train-source and reused-heldout OOF postmortem. It is not a new validation result, not dev, not official, not public test, and not SOTA evidence. The supported scientific content is descriptive: source fit saturation coexists with heldout instability, and cross-modality residual geometry is often negative. Causal explanations and alignment-loss benefits remain prospective hypotheses.

Evidence:

- `docs/V19_GENERALIZATION_DIAGNOSIS_PROTOCOL_2026-09-05.md:3-4` states V19 Q1_FAIL remains unchanged and the diagnostic must not be treated as new validation.
- `docs/V19_GENERALIZATION_DIAGNOSIS_PROTOCOL_2026-09-05.md:18-20` classifies source results as training-fit diagnostics, not generalization evidence.
- `docs/V19_GENERALIZATION_DIAGNOSIS_PROTOCOL_2026-09-05.md:34-36` states correlation from one terminal diagnostic cannot uniquely prove camera, modality, or capacity causality and cannot alter V19 failure status.
- `results/TRIFUSION_RGBNT201_V19_GENERALIZATION_DIAGNOSIS_2026-09-05.md:225-237` states which interpretations are supported and which causal/next-hypothesis claims remain untested.
- `results/TRIFUSION_RGBNT201_V19_GENERALIZATION_DIAGNOSIS_2026-09-05.md:239-241` states this is not independent new validation and does not support dev/official/SOTA.
- `refine-logs/v19/EXPERIMENT_TRACKER.md:49-51` records that the diagnostic completed, had no training/dev/official, and still required its own scope audit.
- `evidence/trifusion_v19_q1_seed42_4b749cd.json:3`, `26`, `33-35`, and `78074-78082` retain Q1_FAIL, train-internal OOF type, no dev/official/D1, and the original Q1 scope/scientific checks.

Audit result: diagnostic classification passes. V19 scientific qualification remains failed.

## Claim assessment

| Claim | Audit result | Basis |
|---|---|---|
| The geometry diagnostic completed all six final endpoint replays read-only | supported with receipt-bound caveat | raw diagnostic status, log, transfer receipt, optimizer0/checkpoint writes0/dev0/official0 |
| Heldout five-output AP/Rank arrays exactly reproduce Q1 | supported | all endpoint heldout scores equal original Q1 arrays; metric replay max diff 0.0 |
| Source retrieval is saturated at 100% mAP/R1/R5/R10 for all five outputs and both endpoints | supported as training-fit diagnostic | independent metric replay from source arrays |
| All seven source classification heads are 100% accurate | supported as training-fit diagnostic | independent replay: 6252/6252 correct for every endpoint/head |
| There is a source-to-heldout descriptive generalization gap | supported descriptively | source fused mAP 100.0 vs heldout fused 80.240792/80.496828 |
| Cross-modality residual geometry does not automatically form a shared identity direction | supported descriptively within this diagnostic | different-modality source margins remain negative and negative-at-least-as-close rates are 71.555750% to 78.677758%; heldout rates are about 90.338587% to 94.016346% |
| Cross-modality direction inconsistency caused V19 failure | unsupported | protocol and result report explicitly warn the diagnostic correlation is not causal proof |
| A future identity-aware cross-modality alignment loss will improve dev/official performance | unsupported/prospective | must be separately preregistered and tested |
| V19 Q1 scientific status changed | rejected | summary and Q1 artifact remain Q1_FAIL |
| Any D1/dev/official/SOTA result is supported | unsupported | no D1/dev/official/public-test run was performed or audited |

## Limitations and byte-read exclusions

- I did not launch training, evaluation, remote execution, model inference, feature extraction, checkpoint tensor loading, image reads, feature replay, distance replay, or downloads.
- I did not execute the executor summarizer as sole evidence. I used an independent JSON/NumPy replay from raw diagnostic arrays and compared to the summary.
- Large checkpoint bytes and image/data bytes used by the original remote diagnostic were not independently read locally; they remain receipt-bound.
- The source 6252 memberships and 1142 source queries are fold-model memberships, not unique new physical data counts. After remapping fold-local source labels to original identities, there are 3126 unique physical source records, and each original identity appears in source for exactly two fold models.
- The heldout side is reused Q1 train-internal OOF, not dev, official, public-test, or independent deployment validation.
- The diagnostic supports descriptive postmortem claims only. It does not establish a causal mechanism or validate a new training intervention.

Final audit conclusion: **engineering_integrity pass**, **scientific_qualification fail**, **integrity_status warn**, **overall_verdict warn**.
