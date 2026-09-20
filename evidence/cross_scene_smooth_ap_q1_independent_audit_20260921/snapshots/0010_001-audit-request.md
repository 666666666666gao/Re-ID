Perform an independent, read-only experiment-integrity audit of the complete MSVR310 cross-scene Smooth-AP Q1. Follow C:/Users/gb/.codex/skills/experiment-audit/SKILL.md and its shared execution, integrity and reviewer-independence references. Assume possible defects and derive conclusions directly from files. Do not accept the executor's scientific interpretation.

Read all applicable files below and follow their actual imports, protocol, ground-truth, initialization and artifact references:

- Repository: C:/Users/gb/.trifusion_github_publish_22c3bee
- Config: configs/MSVR310/TriFusion-cross-scene-smooth-ap-paired-v1.json
- Plan and tracker: refine-logs/msvr310_cross_scene_smooth_ap_v1/EXPERIMENT_PLAN.md and EXPERIMENT_TRACKER.md
- tools/run_msvr_cross_scene_smooth_ap.py
- tools/train_msvr_cross_scene_smooth_ap.py
- tools/verify_msvr_cross_scene_smooth_ap.py
- tools/msvr_cross_scene_smooth_ap.py
- tools/check_msvr_cross_scene_smooth_ap.py
- tools/check_msvr_cross_scene_smooth_ap_math.py
- tools/audit_msvr_paired_ranking_text.py
- Source-support evidence: evidence/smooth_ap_cross_scene_support_20260909/smooth_ap_cross_scene_support_20260909.json
- Prior M0 audit: refine-logs/msvr310_cross_scene_smooth_ap_v1/EXPERIMENT_AUDIT_M0.md and .json; use its limitations as obligations to inspect, not as a Q1 verdict.
- Actual complete text intake: D:\Program Files\UserCache\gb\codex\tmp\trifusion_cross_scene_q1_complete_20260921
- Executor complete report and log-analysis output: D:\Program Files\UserCache\gb\codex\tmp\trifusion_cross_scene_q1_source_log_analysis_20260921.json; no executor narrative report yet
- Remote original run: /root/trifusion-storage/artifacts/msvr310_cross_scene_smooth_ap_v1_seed42_d35864d
- Remote repository: /root/autodl-tmp/trifusion-v2/TriFusion-ReID

Audit A-F from the skill: dataset GT provenance and full-path source/held-out isolation; raw metric normalization; actual file existence, values and provenance; objective/evaluation live call paths; full scope of claims; real GT versus proxy classification. Independently recompute all completed ranking results, aggregate definitions, original gates and identity bootstrap where the saved evidence permits. Check complete gallery distractors and MSVR scene filtering, not a person-ReID camera substitute. Cover every fold, endpoint, output, identity and legal query without sampling a favorable subset.

Inspect all source training steps and target-mask arithmetic. Verify the current-anchor eligibility mean, ignored same-ID/same-scene positions, negative identities retained regardless of scene, self handling, missing-positive anchors retained elsewhere, actual zero-eligible batches, unchanged other supervision, warmup schedule, seed, initialization, fixed terminal epoch, and absence of data-dependent checkpoint selection. Trace current and historical candidate derivatives, random state and buffer handling, single optimizer update, historical-anchor definition and any direct/VJP comparisons. Distinguish cumulative gradient coverage from every-step coverage. Verify frozen states and role checkpoint bindings where evidence is available.

Separate independently recomputable distances/rankings/objective scalars from saved gradient/runtime assertions. Do not promote one direct-gradient example to all-step gradient reconstruction. If exact original model forwards or all parameter-gradient vectors are unavailable, state this plainly rather than replacing them with synthetic results. Candidate and control active total losses may have different definitions; review comparative claims accordingly. Review actual computation and storage accounting, source exposure versus independent sample counts, and fixed-seed bootstrap scope. Do not infer causal loss weights from endpoint gradient ratios or partial-fold scores.

No training, optimizer updates, official-test reads, environment installs, new evaluation protocols, gate changes or failed-version rescues. Arrays/checkpoints/images remain remote; only code, text, reports and hashes may be copied locally. Use fresh output paths under D:/Program Files/UserCache/gb/codex/tmp. Do not overwrite existing artifacts or edit training code. Remote read-only payloads may use the established role_set_remote_command_20260908.py wrapper with the existing uv/numpy/paramiko environment. Never print, copy or publish the private authentication helper or credentials. Avoid rerunning the separate M0 audit or completed source census.

Output independent scripts, evidence results and hashes, an MD/JSON A-F audit with exact file:line references, the integrity verdict separate from scientific qualification, limitations and actionable blockers, plus a verbatim final_response.md. Record failed checks as failures, not omitted attempts. This audit must not change registered thresholds to turn a negative result into a pass.

Save the independent report, scripts, results and verbatim final_response.md under NEW D:/Program Files/UserCache/gb/codex/tmp/trifusion_cross_scene_q1_independent_audit_20260921. Do not modify the repository or existing evidence. Send progress and exact output paths.
