**Overall verdict: WARN.** Recorded execution and saved-result integrity pass. The registered scientific advancement gate is **FAIL** on all three datasets.

Reviewer: fresh **gpt-6-astra / max**, `review_independence: same-family`, `acceptance_status: provisional`. This is neither cross-family acceptance nor a performance-success finding.

All references below are relative to remote `/data/gaob/Re-ID/Trifusion`.

**A — Ground-truth provenance: PASS**

- Labels originate in dataset filenames and split records, not model outputs. See `tools/build_official_three_dataset_protocols.py:32–80`; author parsers `comparators/Signal-cd1b0a6/data/datasets/RGBNT201.py:61–85`, `RGBNT100.py:63–84`, and `msvr310.py:67–87`.
- I independently parsed every registered filename, verified every referenced image path exists, enumerated the complete primary-modality split files, checked contiguous record indices, and verified train/test identity disjointness.
- Saved identity/camera/scene arrays exactly match all protocol records. Query/gallery counts are **836/836**, **1715/8575**, and **591/1055**. MSVR310 retains **103 gallery-only identities**; none were removed.
- Every query has a valid positive after filtering. RGBNT201/RGBNT100 remove same-ID/same-camera entries; MSVR310 removes same-ID/same-scene entries. Evidence: `tools/run_correspondence_context_identity.py:207–224`, author `utils/metrics.py:68–106,137–168`.
- Limits: this audit checked installed records and paths, not a fresh download or rehash of every image payload.

**B — Improper score normalization: PASS**

AP uses dataset-positive positions; CMC averages query success. Neither divides performance by prediction maxima, means, or other model-derived references. See `tools/train_rgbnt100_signal_oof.py:253–268`, `tools/train_msvr310_signal_oof.py:223–238`, and `tools/run_correspondence_context_identity.py:222–224`.

Feature L2 normalization and attention softmax are model/distance operations, not improper performance normalization: `tools/run_official_three_dataset_roles.py:230–238`, `modeling/trifusion/slot_competition_roles.py:9–15`.

**C — Result existence and metric/epoch/SHA correspondence: PASS**

All six accepted rows correspond to existing best checkpoints, distance artifacts, training receipts and evaluation receipts. Full SHA256 values match across these artifacts, the parent campaign, accepted matrix and summary.

| Dataset | Condition | Best epoch | mAP | Rank-1 | Accepted-matrix evidence |
|---|---|---:|---:|---:|---|
| RGBNT201 | reid_visual | 2 | 72.524680 | 73.923445 | `logs/visual_start_roles_20261001_v1/accepted_matrix.json:451–489` |
| RGBNT201 | public_visual | 34 | 70.161768 | 72.368419 | Same file `:118–156` |
| RGBNT100 | reid_visual | 1 | 85.105062 | 95.043731 | Same file `:340–378` |
| RGBNT100 | public_visual | 16 | 77.991338 | 93.411076 | Same file `:7–45` |
| MSVR310 | reid_visual | 10 | 52.415867 | 67.851102 | Same file `:562–600` |
| MSVR310 | public_visual | 24 | 48.016562 | 66.159052 | Same file `:229–267` |

Each checkpoint epoch matches the maximum fused mAP over all 50 recorded epochs, with the implemented last-epoch tie rule. Evidence: `tools/run_correspondence_context_identity.py:137–142`, `tools/collect_role_global_tokens.py:82–107`.

Independent CPU scoring of **18 matrices / 72 metrics** agrees with recorded metrics within **2.737821006348895×10⁻⁶ percentage points**. All saved per-query AP/rank values, paired identity summaries, flips, bootstrap intervals and Markdown result rows also check out.

**D — Actual metric execution and report provenance: PASS, within historical-record limits**

- The reachable evaluator performs strict selected-checkpoint loading, full query/gallery extraction, calls both independent and author scorers, and saves distances plus metrics: `tools/run_role_global_tokens.py:52–79`; `tools/run_correspondence_context_identity.py:175–241`.
- All six child campaigns contain actual `m0 → train → evaluate` subprocess PIDs, timestamps and exit code zero. The executor obtains these using `process.wait()`, not inferred completion: `tools/queue_visual_start_roles.py:53–79`. In every child `campaign.json`, stage exit records occur at lines **57,108,159**.
- I cross-checked all **18 stage logs**, all **300 training epochs**, **20,416 formal training steps**, and **48 M0 steps** against receipts. Loss arithmetic and batch sequences pass.
- All six M0 receipts report **128/128** trainable tensors receiving nonzero gradients across the eight-step probe, unchanged frozen Signal state, and zero reload output difference. This is the recorded M0 result; I did not rerun it.
- Controller completion is independently wrapped: `logs/visual_start_roles_wrapper_20261001_v1.json:2–24`. Report execution has a pinned source SHA, child PID, actual wait and exit zero: `.codex_tmp/wait_visual_start_complete_analysis_20261001.py:38–50`; `logs/visual_start_roles_20261001_v1/analysis_waiter_status.json:8–24`.
- The report’s main was **not rerun** during this audit.

**E — Scientific scope, initialization, cost and reproducibility: WARN**

The six-run experiment supports a qualified negative result under its registered setup.

| Dataset | Public-minus-ReID mAP, pp | Rank-1, pp | Repairs / new errors | Gate |
|---|---:|---:|---:|---|
| RGBNT201 | −2.3629119942 | −1.5550239234 | 64 / 77 | FAIL |
| RGBNT100 | −7.1137245430 | −1.6326530612 | 63 / 91 | FAIL |
| MSVR310 | −4.3993049678 | −1.6920473773 | 42 / 52 | FAIL |

These independently recomputed gates match the preregistered thresholds: `refine-logs/visual_start_roles_v1/EXPERIMENT_PLAN.md:21`; implementation `tools/report_visual_start_roles_complete.py:106–110`; results `results/visual_start_roles_complete_20261001/SUMMARY.json:91094–91098,91307–91311,91620–91624,91943`.

Initialization checks pass:

- All **456 replaced visual tensors** directly match the registered public CLIP tensors, including the declared positional interpolation.
- All **45 nonvisual checkpoint entries** remain bitwise equal, including trained camera embeddings.
- Public CLIP and all six original/reset checkpoint file hashes match; all six frozen Signal-state digests also match.
- Evidence: `tools/prepare_visual_start_inputs.py:23–60`; `pertrained-model/visual_start_reset_20261001/INPUTS.json:1145–1146,2301–2302,3457–3465`.
- M0 and formal-training initializer bindings exactly match the preflight witness. Its complete nonvisual-initialization equality assertion is supported by the source and archived witness: `tools/check_visual_start_initialization.py:54–75`; `refine-logs/visual_start_roles_v1/INITIALIZATION_WITNESS.json:8–10,264–266,520–522`. I did not reconstruct the production models to repeat that witness.

Warnings that must remain:

1. **One seed and official-set selection.** Each model selects its best official fused mAP after 50 epochs. These are development-selected results, not untouched-test estimates, stability evidence or seed variance. The identity bootstrap resamples fixed-model identities, not training seeds. Evidence: plan `:19–21`; `tools/analyze_correspondence_distances.py:58–75`; report `:27`.
2. **The intervention is conditional.** `public_visual` retains dataset-trained camera and other nonvisual states. It is not a wholly public-pretrained system or a new architecture. These results do not establish that public visual representations are generally inferior or identify the unique cause of previous small role gains.
3. **Readout diagnostics are jointly trained.** Shared-global features are affected by trainable in-backbone adapters; `joint_local` is the total normalized role correction. Neither is an independently trained control. Evidence: `modeling/trifusion/correspondence_roles.py:58–90`; `modeling/trifusion/role_global_tokens.py:79–94`; report `:23–26`.
4. **Cost is partial.** Recorded campaign duration is **7,943.015467 seconds**; summed endpoint duration is **20,796.080268 seconds**. These match timestamps but exclude upstream ReID training and do not measure GPU compute equivalence. Evidence: summary `:91944–91945`; reporter `:82–87`.
5. **A small reporting requirement remains unmet:** plan `:23` requests disk usage, but SUMMARY contains no disk accounting. My read-only snapshot measured **534,492,553 logical bytes** across the six endpoints’ M0 and full-run output directories. This excludes baseline/public inputs and is neither historical peak storage nor complete project storage.
6. Checkpoints omit frozen Signal state and require the separately preserved input checkpoints: `tools/run_correspondence_roles.py:111–113`. Source hashing alone is not complete environment/native-kernel reproducibility.

**F — Evaluation classification: PASS — `real_gt`**

All 18 fused/shared-global/total-correction retrieval evaluations use real dataset-provided identity and environment labels. Paired AP, rank-flip and identity-bootstrap analyses are **post-selection diagnostics over real-GT evaluations**. Initialization and M0 checks are engineering witnesses, not retrieval-performance evaluations.

**Concrete disposition**

- Accept the six completed endpoints and saved metrics as verified within this audit’s limits.
- Preserve the original **FAIL** gate. Do not reinterpret the panel as an initialization improvement, novelty, stability or SOTA success.
- Add a dated disk-usage observation to the terminal accounting without rewriting the frozen registration or rerunning the report.
- Preserve the frozen dependencies and existing warning language. No training restart, additional seed or architecture change is justified by this integrity audit.

**Checks actually executed and limits**

- Main independent verification: **126,890 assertions**, **286 distinct file-hash checks**, including **221 runtime-source SHAs** and **41 report-bound artifacts**.
- Additional **348 aggregate/Markdown checks**; seven input-weight SHA checks; public-reset tensor comparisons and six frozen-state digest checks.
- CPU deserialization of six best checkpoints, six M0 probes and six saved-distance artifacts; no production model forward, inference, optimizer, M0, training, GPU job or report-main execution.
- No files were edited. Native kernels, historical gradients and neural outputs were not independently replayed.
- One audit diagnostic initially used the witness’s dtype/shape-inclusive hashing convention for Signal state. After reading the actual Signal hash implementation (`tools/run_signal_preserving_v5.py:523–528`), all six digests passed. This was an auditor-side comparison error, not an artifact discrepancy.

Audited summary SHA256: `214ea13646e218bb3a1f771ba5bebd9255c6065f05b99cda37d37680f2563567`  
Audited report SHA256: `79b2359f607623d735d1e834f8ce4fb9e63d3b820730a19db21632ed21f3076d`
