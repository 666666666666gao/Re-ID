# Fixed same-evidence comparison903: terminal904 integrity audit

**Overall verdict: WARN.** No blocking source, protocol or numerical integrity defect was found. The bounded diagnostic is supported provisionally; this is a source, text and arithmetic audit, not an independent neural or tensor-cache reproduction.

Date: 2026-10-12T10:01:09.987169+00:00. Requested reviewer: `gpt-6-astra`, effort `max`. Actual backend model/effort: **UNATTESTED**. Fresh task: `/root/audit_same_evidence_terminal904`; review independence **same-family**; acceptance **provisional**. No cross-family acceptance or SOTA/new-method guarantee is claimed.

Path notation: source paths below are relative to `C:/Users/gb/.trifusion_github_publish_22c3bee` (R). `D201`, `D310`, and `D100` denote the respective `DIAGNOSIS.json` under `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/terminal904/logs/same_evidence_comparison_v1_20261012_903/`. Author narrative paths are relative to `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903` (B). Author evaluator and protocol text are from the hash-matched snapshot `C:/Users/gb/.codex_tmp/independent_evidence_draft/text_semantic_prior901/full_result_review901/source_snapshot` (S).

**A–F checks**

| Check | Status | Evidence and finding |
|---|---|---|
| A. Ground truth | PASS with provenance boundary | IDs/cameras/scenes come from frozen dataset records, not predictions (`R/tools/build_official_three_dataset_protocols.py:32`, `:44`; `R/tools/diagnose_native_research_best.py:45`). Independently verified protocol counts, every query mask and 43,868 modality-path filename labels. All 3,142 queries have legal positives; 103 MSVR gallery-only identities remain distractors. Dataset images and historical inventory authenticity were not reread. |
| B. Normalization | PASS | Ordinary AP/CMC at `R/tools/train_rgbnt100_signal_oof.py:253` and `R/tools/train_msvr310_signal_oof.py:223`; cosine/energy score definitions at `R/tools/diagnose_same_evidence_comparison.py:80` and `:158`. No reported metric is normalized by a prediction maximum or mean. |
| C. Existence and numbers | PASS within text scope | Three COMPLETE endpoints and exit0 are present (`B/terminal904/logs/same_evidence_comparison_v1_20261012_903/campaign.json:3`, `:12`, `:20`, `:28`; `B/terminal904/logs/text_semantic_prior_resume901/EVIDENCE_COMPARISON_EXIT_903.json:2`). All 21 metric panels and 21 contrast panels are internally consistent. Current remote text hashes match all six supplied terminal JSONs. |
| D. Active computation | PASS | `main` dispatches the serial campaign and `diagnose`; extraction, all seven scores, comparisons and output are connected (`R/tools/diagnose_same_evidence_comparison.py:41`, `:152`, `:165`, `:180`, `:190`, `:226`). `score` invokes the independent per-query scorer and exact pinned author evaluator (`R/tools/diagnose_native_research_best.py:51`). No current claim depends on an unused historical helper or optimizer routine. |
| E. Scope | WARN | Three historically selected semantic checkpoints, one training seed42, seven fixed scores, already consumed official development sets (`R/refine-logs/evidence_comparison_diagnosis_v1/EXPERIMENT_PLAN.md:5`, `:21`, `:27`; `B/RESULTS_904.md:43`). Identity summaries are not training-seed replication. Regional comparisons change representation and deployment cost. |
| F. Evaluation class | PASS: `real_gt` retrieval | AP/CMC uses dataset labels. Region dispersion (`R/tools/diagnose_same_evidence_comparison.py:172`) is an unlabeled feature statistic. Algebra fixtures are implementation checks, not retrieval experiments (`R/refine-logs/evidence_comparison_diagnosis_v1/SCORE_ALGEBRA_CHECK_903.json:14`). |

**Actual evidence and formula checks**

The requested `modeling/trifusion/role_evidence.py` does not exist. `RoleEvidence` is the dataclass at `R/modeling/trifusion/correspondence_roles.py:17`. The actual raw-semantic path is `DetachedSemanticTriFusion` → `RawFeatureSemanticTriFusion.forward_features` → `self.roles(...)` → `read_evidence(evidence)` (`role_input_detach.py:5`; `independent_native_roles.py:63`). The roles return the final normalized CNN/Transformer/Mamba slot tensors, excluding the extra interaction token (`role_global_tokens.py:43`, `:52`, `:61`). The hook records those returned tensors, not attention keys or geometrically annotated parts (`diagnose_same_evidence_comparison.py:45`). Their shape is `[batch,3 roles,3 modalities,16 slots,128]` in source; no tensor content was loaded in this audit.

Structured readout is enabled by `ContextIdentityTriFusion.__init__` (`correspondence_context_identity.py:70`) and pools the 16 slots into four groups before the learned 1536-dimensional readout (`correspondence_evidence_readout.py:65`). GlobalTaskRoleHeads inference returns the same fused feature (`global_task_role_heads.py:17`). Thus bypassing its training-head branch does not change inference semantics.

For raw global g, raw correction c, and the existing gain γ, additive cosine uses `g + γc`, including both cross terms and its additive-vector norm. Same-residual block cosine is exactly `cos([gq; γcq], [gg; γcg])`, with numerator `gq·gg + γ² cq·cg` and denominator `D = sqrt(||gq||² + γ²||cq||²) sqrt(||gg||² + γ²||cg||²)`. Consequently it changes cross terms **and** normalization; it cannot isolate a causal cross-term effect (`diagnose_same_evidence_comparison.py:142`, `:158`). One hundred independent stdlib scalar fixtures agree with direct concatenation and the residual-cosine substitution to maximum error 1.67e-16.

Region matching keeps the nine role/modality groups separate. Within each group it averages source-to-target and target-to-source maximum cosines over all 16 slots, then averages groups equally (`diagnose_same_evidence_comparison.py:72`). It is symmetric many-to-one MaxSim, not a bijection, physical visibility estimator or cross-role match. Indexed comparison averages matching slot indices. Fixed-energy regional block is `(gq·gg + γ² ||cq|| ||cg|| M(q,g)) / D` (`:169`): only the residual-cosine factor is replaced, without a fitted coefficient. It retains different evidence from the exact same-residual block. Scalar score differences and within-group directional dispersion do not establish part separability.

**Protocol and metric reproduction**

| Dataset | Queries / gallery | Query IDs / gallery IDs | Legal positives per query | Selected epoch |
|---|---:|---:|---:|---:|
| RGBNT201 | 836 / 836 | 30 / 30 | 7–21 | 8 |
| MSVR310 | 591 / 1055 | 52 / 155 | 1–31 | 38 |
| RGBNT100 | 1715 / 8575 | 50 / 50 | 50–175 | 5 |

Camera evaluation removes only same-identity/same-camera rows; MSVR removes same-identity/same-scene rows. All different identities remain, including MSVR’s 103 gallery-only identities (`S/comparators/Signal-cd1b0a6/utils/metrics.py:68`, `:137`; independent scorers above). Source fills complete query × gallery matrices; chunk16×128 limits temporary memory, not candidates (`diagnose_same_evidence_comparison.py:87`). The frozen protocols cover 13,608 query-plus-gallery records and 16,028,526 pairs per method. No score-based filtering, top-k shortlist, query dropping or new mask appears.

Independently recomputed mAP and Rank-1/5/10 from all 21,994 saved AP/first-rank records, then checked all 21,994 paired query records against those arrays and ordered protocol identities. Recomputed identity means, every repair/new-error/kept-correct/kept-wrong count and total-mAP contribution, all seven AP-delta percentiles, and the mean of the worst ceil(5%·N) queries (42/30/86). All 1,407 aggregate assertions pass; maximum arithmetic difference is 2.84e-14. These are text arithmetic checks; ranking from feature/distance tensors was not rerun.

The diagnostic additive scores differ from original selected receipt metrics by at most 2.22448e-6 / 2.52135e-6 / 2.35638e-6 percentage points for RGBNT201/MSVR310/RGBNT100, all below 1e-5. RGBNT100 mAP differs by -1.607887e-6. Original distance differences are reported as 7.152557e-7 / 7.152557e-7 / 1.192093e-6 (`D201:12`, `D310:12`, `D100:12`). This is metric-tolerance reproduction, not bitwise historical-distance identity. The reviewer authenticated the original metric/initialization texts and their sealed hashes; the original binary matrices were not loaded.

**All predetermined score panels, mAP / Rank-1 (%)**

| Score | RGBNT201 | MSVR310 | RGBNT100 |
|---|---:|---:|---:|
| global | 74.296664 / 78.947368 | 50.542129 / 68.020305 | 84.369219 / 96.093294 |
| additive | 74.374862 / 78.827751 | 50.542225 / 67.851100 | 84.090303 / 95.860058 |
| residual | 11.578064 / 9.090909 | 6.820775 / 16.243655 | 36.625906 / 57.084548 |
| same_residual_block | 74.338753 / 78.947368 | 50.538969 / 68.020305 | 84.077846 / 95.976676 |
| aligned_regions_only | 14.237502 / 13.995215 | 8.769176 / 21.319797 | 41.684449 / 62.448980 |
| symmetric_regions_only | 14.160263 / 14.354067 | 8.773044 / 21.319797 | 44.634323 / 68.163265 |
| fixed_energy_region_block | 74.326469 / 78.947368 | 50.540551 / 68.020305 | 84.457925 / 95.918367 |

Source result anchors: `D201:13`, `:1698`, `:5066`, `:10118`; `D310:13`, `:1208`, `:3596`, `:7178`; `D100:13`, `:3456`, `:10340`, `:20666`. The final author narrative’s 87 tabulated numeric fields agree at their stated rounding precision (`B/RESULTS_904.md:9`, `:19`).

**Primary contrast: same-residual block minus additive**

| Dataset | ΔmAP | ΔRank-1 | First-rank repairs / new errors | Worst5% mean ΔAP |
|---|---:|---:|---:|---:|
| RGBNT201 | -0.036109 | +0.119617 | 4 / 3 | -3.252781 |
| MSVR310 | -0.003256 | +0.169205 | 1 / 0 | -0.493788 |
| RGBNT100 | -0.012457 | +0.116618 | 16 / 14 | -8.613195 |

The primary mAP contrast is negative on all three datasets despite small net Rank-1 repair counts. The corresponding result sections are `D201:53090` (groups `:59942`, tail `:59969`), `D310:38560` (groups `:43562`, tail `:43589`), and `D100:108188` (groups `:122172`, tail `:122199`). Thus the experiment does not establish a general benefit from separating the two vectors. A first-rank “repair” is not necessarily an AP gain.

For fixed-energy regions versus same-model global, RGBNT201 is +0.029805 mAP with unchanged Rank-1; MSVR310 is -0.001577 with unchanged Rank-1; RGBNT100 is +0.088707 mAP but -0.174927 Rank-1, with 3 repairs/6 new errors and worst5% mean ΔAP -4.437161 (`D100:94175`, `:108159`, `:108186`). RGBNT100’s gain is mainly the kept-correct group contribution (+0.086860 total-mAP points). Versus additive, RGBNT100 is +0.367622 mAP/+0.058309 Rank-1, but Rank-10 falls0.058309 points; 18 repairs/17 new errors coexist with worst5% mean ΔAP -11.326943. These latter contrasts were independently derived from existing text, without generating new scores. Regional score/representation gains are descriptive tradeoffs, not proof of a stronger trained method or acceptable deployment.

**Source sealing, prior review and limits**

One read-only SSH call used `/usr/bin/python3` stdlib to rehash all 439 currently sealed remote source files: 439 match, zero mismatches. It also authenticated six remote terminal text files against the local copies, three protocol texts, and the three original selected metric receipts plus initialization witnesses. Local publication coverage alone is 305 exact files, 107 absent files and 27 raw-byte differences; every difference is exactly CRLF versus LF. These were not silently normalized or called exact local matches. Remote hashes and the directly inspected hash-matched author snapshot supply the appropriate source evidence.

R1 remains REVISE, and R2 remains source-only. Removing the current runner’s added comment and `os.chdir(output.resolve())` exactly reconstructs the old R1 source hash. The known MSVR `re.txt` write at author evaluator lines38/71 is contained in the fresh endpoint directory for the actual absolute-output command (`diagnose_same_evidence_comparison.py:149`; `B/terminal904/logs/text_semantic_prior_resume901/EVIDENCE_COMPARISON_TERMINAL_904.json:10`). No evaluator or score formula changed. Storage arithmetic is correct: 1,254,113,280 raw FP32 bytes +64MiB allowance +2GiB reserve =3,468,705,792B. This is source/seal arithmetic, not a new resource-admission query.

Six cache files exist with sizes matching their result declarations. Only stat metadata was read: cache SHA-256s and contents were **not** independently reauthenticated. No images, model weights, public CLIP payload or original distance tensors were loaded or rehashed. Model-before/after equality, gradient absence, actual forward completion, dispersion, timing and distance discrepancies are original-run evidence supported by code and receipts, not newly executed checks. No present GPU/process query was made. The historical terminal `live_commands: []` and observer-continuity statements are not fresh process verification.

The final narrative correctly marks full Signal80.3029 as a historical user-resume reference (`R/refine-logs/USER_RESUME_20261010.md:7`; `B/RESULTS_904.md:45`), not an authenticated comparator in this diagnostic. The sentence about the next mechanism (`B/RESULTS_904.md:37`) is a research direction, not a demonstrated causal necessity. No automatic new training, teacher, gain search or visibility rule follows.

**Disposition and artifacts**

No source or original-result correction is required for the bounded diagnostic on the evidence reviewed. Keep all seven outcomes, provenance limits, seed/checkpoint selection disclosure, development-set exposure and the unmet broad goal. Do not convert tiny mAP/rank tradeoffs or region direction similarity into compression causality, physical absence/visibility decisions, novelty, trained-method improvement or SOTA claims.

Final narrative SHA-256: `2269ec9e87dc47cf2c61a2228c782b013d24b0536afe1d81a1cfa63fc866cdaf`.
Input seal SHA-256: `3e6eb3be936f0f8bbd695fe6c797413613b746444768de42a41cbf0ac2b1f2ab`.
Audited runner SHA-256: `71fe059e26744c670661cab6fb3bbaaf13da56a5567adf78b8079e2ba0db9ad5`.

Full input hashes, precise result line maps and independent arithmetic are in `DETERMINISTIC_CHECKS_904.json`; current remote text hashes and cache-stat-only evidence are in `REMOTE_TEXT_READ_904.json`; the A–F verdict and claim disposition are in `EXPERIMENT_AUDIT.json`. All files are confined to the requested private `review/terminal904` directory. No production/source/results edits, neural execution, GPU query, optimizer update, remote mutation, package installation, external reviewer or sub-agent was performed.
