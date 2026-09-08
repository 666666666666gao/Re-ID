## Reviewer prompt

You are an experiment integrity auditor. Read source artifacts directly and independently establish what the evidence supports. Do not assume the executor's summaries or prior audits are correct. Audit the complete MSVR310 historical-candidate-gradient Q1 experiment, including evaluation integrity, actual optimization definition, full scope, reproducibility bindings, and limits of the available evidence.

Read-only with respect to scientific code, configurations, checkpoints, existing evidence and logs. You may write your own audit scripts, deterministic check outputs and final response only under C:/Users/gb/.codex_tmp/history_gradient_q1_independent_audit_20260908. Do not train, install packages remotely, edit the experiment, restart jobs or read official-test images. Use existing environments. Do not expose credentials; if remote read-only artifact checks are needed, the existing connection helper path is listed below and its contents must never be printed or included in traces.

Project root: C:/Users/gb/.trifusion_github_publish_22c3bee

Files and directories to inspect directly:

- C:/Users/gb/.agents/skills/experiment-audit/SKILL.md
- C:/Users/gb/.agents/skills/shared-references/local-codex-policy.md
- C:/Users/gb/.agents/skills/shared-references/reviewer-independence.md
- C:/Users/gb/.agents/skills/shared-references/experiment-integrity.md
- C:/Users/gb/.agents/skills/shared-references/review-tracing.md
- C:/Users/gb/.trifusion_github_publish_22c3bee/AGENTS.md
- C:/Users/gb/.trifusion_github_publish_22c3bee/configs/MSVR310/TriFusion-history-gradient-paired-v1.json
- C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/msvr310_history_gradient_v1/EXPERIMENT_PLAN.md
- C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/msvr310_history_gradient_v1/EXPERIMENT_TRACKER.md
- C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/msvr310_history_gradient_v1/POST_Q1_RESEARCH_CANDIDATES.md
- C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_msvr_history_gradient.py
- C:/Users/gb/.trifusion_github_publish_22c3bee/tools/check_msvr_history_gradient.py
- C:/Users/gb/.trifusion_github_publish_22c3bee/tools/train_msvr_history_gradient.py
- C:/Users/gb/.trifusion_github_publish_22c3bee/tools/verify_msvr_history_gradient.py
- C:/Users/gb/.trifusion_github_publish_22c3bee/tools/build_msvr310_train_oof_protocol.py
- C:/Users/gb/.trifusion_github_publish_22c3bee/tools/audit_vehicle_query_protocol_labels.py
- C:/Users/gb/.trifusion_github_publish_22c3bee/tools/analyze_msvr_history_gradient_training.py
- C:/Users/gb/.trifusion_github_publish_22c3bee/tools/audit_msvr_paired_ranking_text.py
- C:/Users/gb/.trifusion_github_publish_22c3bee/evidence/msvr310_history_gradient_training_preexecution_20260908/audited_input_hashes.json
- C:/Users/gb/.trifusion_github_publish_22c3bee/results/MSVR310_HISTORY_GRADIENT_V1_M0_2026-09-08.md
- C:/Users/gb/.trifusion_github_publish_22c3bee/docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md
- All actual imported model, dataset, sampler, loss, evaluator and frozen-coordinate replay dependencies of the above scripts. Use the input-hash inventory as an additional path locator, not as proof of correctness.
- C:/Users/gb/.codex_tmp/history_gradient_q1_complete_20260908/intake_manifest.json
- All 29 received files under C:/Users/gb/.codex_tmp/history_gradient_q1_complete_20260908, including pipeline.json, q1/summary.json, q1_cpu.json, q1.log, q1_cpu.log and all six endpoints' memory_steps.jsonl, training.json, receipt.json and rankings.json. Discover actual names from the manifest, and inspect every listed file.
- C:/Users/gb/.codex_tmp/history_gradient_q1_terminal_processing_20260908/state.json
- C:/Users/gb/.codex_tmp/history_gradient_q1_terminal_processing_20260908/training_text_analysis.json
- C:/Users/gb/.codex_tmp/history_gradient_q1_terminal_processing_20260908/rankings/ranking_replay.json
- C:/Users/gb/.codex_tmp/history_gradient_q1_terminal_processing_20260908/rankings/all3000_query_output_changes.csv
- C:/Users/gb/.codex_tmp/history_gradient_q1_terminal_processing_20260908/rankings/all300_identity_output_changes.csv
- C:/Users/gb/.codex_tmp/collect_history_gradient_q1_20260908.py
- C:/Users/gb/.codex_tmp/history_gradient_terminal_watcher_plan_20260908.json
- C:/Users/gb/.codex_tmp/watch_history_gradient_terminal_20260908.py
- C:/Users/gb/.codex_tmp/trifusion_ssh_session_recovery_20260908.py (private connection helper; never print)
- Remote run: /root/autodl-tmp/trifusion-v2/artifacts/msvr310_history_gradient_v1_seed42_a1b4777
- Remote repository: /root/autodl-tmp/trifusion-v2/TriFusion-ReID

Audit checklist:

1. Ground-truth provenance and evaluation classification: trace labels, full-path identity separation, source initialization, legal MSVR scene/time masks, distractor-gallery retention and official-image access. Determine whether reported results are internal Q1, source diagnostics, official tests or synthetic checks. Trace metrics to real dataset labels; distinguish synthetic math tests from performance evidence.
2. Score construction: read the complete distance/ranking/AP/CMC implementations, tie/order handling and actual call sites. Check for model-dependent metric normalization, synthetic references, omitted queries/identities/folds or unreported filtering. Independently reconstruct all available rankings/AP/CMC/identity statistics and registered gate groups from raw records, not only summary fields or executor PASS files.
3. Scope and provenance: check all ends, updates, registered batches, pairing, initialization and augmentation pixel identity, checkpoint and configuration/execution bindings, final reloads, endpoint receipts, pipeline stages and CPU completion. Compare local file hashes to intake and remote artifacts where feasible. Do not confuse later documentation HEAD with execution code or CRLF differences with unexplained scientific changes.
4. Optimizer definition: trace current anchors, in-batch candidate derivatives, fresh historical leaves, upstream gradients, history group replay, RNG/buffer restoration, zero-upstream skips, class0, duplicate/age rules, loss weights and single scaled/unscaled optimizer update. Establish what gradient actually enters each endpoint. Independently check available per-step distance mining, scalar reassembly, selected groups, upstream norms and applied-gradient records over the full run.
5. Verification limits: distinguish recomputation of recorded scalars from regeneration of all model losses; record-level gradient witnesses from independent full model backward; first single-group direct-graph tests from full multi-group equivalence; temporal queue age under fixed models from parameter staleness. If underlying arrays or model recomputation are unavailable locally, state exactly what was and was not checked. Do not convert an executor analysis into an independent check merely by reading it.
6. Costs and claims: inspect all role-forward/VJP counts, peak memory and elapsed times; equal update count is not automatically equal compute. Evaluate repairs/new errors across all queries and identities, role/fold changes and bootstrap assumptions. Single seed42 and reused internal identities do not support multi-seed or untouched independent-test claims. Judge pending research candidates as pending; do not assign them experimental success.
7. Audit all A-F sections from experiment-audit: GT provenance, score normalization, result existence/number agreement, dead code, scope, and evaluation type. Read relevant evaluator/training code line by line and use independent deterministic checks for full datasets of available text records. Record exact file:line evidence and hashes of inputs actually audited.

Return overall PASS/WARN/FAIL, each checklist finding, exact supporting references, deterministic check commands/output paths, missing evidence, claim impact and specific required corrections. Include review_independence=same-family and acceptance_status=provisional. A failed scientific gate is distinct from an integrity failure. Do not predict future experiments or write a favorable verdict to meet project targets.
