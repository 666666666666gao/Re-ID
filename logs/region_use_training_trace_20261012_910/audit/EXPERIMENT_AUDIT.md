# Experiment audit: saved training trace 910

Date: 2026-10-12T23:48:27.499799+08:00

**Overall: WARN.** The saved scalar content is verified. Definition and trend wording were corrected explicitly with original bytes retained. Original producer execution remains unverified because its full native invocation/stdout is unavailable. There are no remaining blocking arithmetic defects for qualified descriptive use.

Requested route: gpt-6-astra / max. Actual serving model and effort: **UNATTESTED**. Fresh delegated reviewer `/root/audit_training_trace910`; same-family / provisional. No cross-family or runtime model attestation is claimed.

## Boundary and actual invocation

Only source/text reading and native CPU standard-library JSON, AST, hashes and scalar arithmetic were performed. Producers and renderer were never imported or executed. No server/SSH/network, model, tensor, NN, GPU, optimizer, production/repository/master edit, publication or restart occurred.

The original consumer source hash equals SUMMARY:4 (`008b9258270f08b769d8e438a1cb6a36dc283cb615b118c71306a6d4df33bfb4`). This proves source identity, not historical invocation. C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\training_trace910\EXECUTION_PROVENANCE_LIMIT.json:9 explicitly says the retained cell186/chunk d8686c/exit0 metadata is a later context transcription. C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\training_trace910\EXECUTION_PROVENANCE_LIMIT.json:10 says the complete call/stdout is absent. That requested provenance check remains blocked for an unqualified execution claim.

Actual auditor command: `C:/Users/gb/AppData/Roaming/uv/python/cpython-3.12.12-windows-x86_64-none/python.exe -I -B C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\review\training_trace910/check_saved_scalars.py`; native chunk `764d98`, exit0, 1.4103835 s. C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\review\training_trace910/SCALAR_CHECK_NATIVE_RESULT.json preserves its exact tool result. Initial `python -B` failed before running the script with `No pyvenv.cfg file` (chunk ed0438, exit1); C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\review\training_trace910/INITIAL_NATIVE_FAILURE.json retains that failure. The already-installed standalone CPython was then selected, without installing packages or changing producer code.

## Deterministic checks

**7,410/7,410 predicates passed; 0 failed.** This total is 7,399 scalar/content checks plus 7 definition-correction checks and 4 final wording checks. It is not a count of independent scientific observations.

| Check category | Count |
|---|---:|
| source_hash | 1 |
| seal_hash | 1 |
| scope | 1 |
| input_set | 1 |
| report_hash | 1 |
| producer_seal | 13 |
| source_only_imports | 2 |
| intake_input_hash | 12 |
| summary_input_hash | 12 |
| receipt | 4 |
| producer_binding | 4 |
| epoch_sequence | 4 |
| best_selection | 4 |
| best_metrics | 4 |
| comparison909_selection | 4 |
| saved_step | 6710 |
| epoch_steps | 200 |
| epoch_aggregation | 200 |
| history_mean_loss | 200 |
| step_coverage | 4 |
| run_totals | 4 |
| first_selected_last | 12 |
| report_rows | 1 |
| definition_correction_targeted | 7 |
| endpoint_wording_targeted | 4 |

All 6,710 JSONL lines were read. Each of 200 epoch records matched in full: epoch, steps, summed legal and nonpositive-margin exposures, their ratio, all four unweighted batch-mean scalar averages, and copied official_fused metrics. Batch indexes are consecutive within each epoch, the 50-epoch histories have no missing/extra epochs, and the recorded overall loss means agree within 1e-12. All four selected epochs equal the maximum stored mAP using the producer latest-tie rule; saved strict receipt metrics agree with selected history within its 1e-5 tolerance. All 12 displayed records and report rounding match.

Exact all-epoch file/line maps are in C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\review\training_trace910/EPOCH_EVIDENCE.json. The unmodified first checker output is C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\review\training_trace910/DETERMINISTIC_CHECKS.json. A few coarse source/manifest/MSVR selection pointers from that checker are corrected explicitly in C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\review\training_trace910/EVIDENCE_LOCATOR_CORRECTIONS.json; the original output and executed checker source are retained.

| Run | Saved steps | Best | Total legal triplet exposures | Total global nonpositive-margin exposures |
|---|---:|---:|---:|---:|
| RGBNT201/uniform | 2649 | 8 | 11089680 | 39278 |
| RGBNT201/ambiguity | 2649 | 8 | 11089680 | 39278 |
| MSVR310/uniform | 706 | 38 | 2793000 | 75342 |
| MSVR310/ambiguity | 706 | 38 | 2793000 | 75342 |

The triplet counts are repeated training exposures, not unique identities, independent tests or query/gallery retrieval samples.

## Exact first/selected/last evidence

Each run prefix below is `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/results908/trained-model/region_use_v1_20261012_906_full_<condition>_<dataset>/`. JSONL endpoints identify the inclusive saved rows used for that epoch.

| Run | Epoch | training.json line | training_steps.jsonl first / last line | SUMMARY.json line |
|---|---:|---:|---|---:|
| RGBNT201/uniform | 1 | 87 | 1 / 53 | 34 |
| RGBNT201/uniform | 8 | 171 | 372 / 424 | 167 |
| RGBNT201/uniform | 50 | 675 | 2597 / 2649 | 965 |
| RGBNT201/ambiguity | 1 | 87 | 1 / 53 | 1052 |
| RGBNT201/ambiguity | 8 | 171 | 372 / 424 | 1185 |
| RGBNT201/ambiguity | 50 | 675 | 2597 / 2649 | 1983 |
| MSVR310/uniform | 1 | 95 | 1 / 14 | 2070 |
| MSVR310/uniform | 38 | 539 | 524 / 537 | 2773 |
| MSVR310/uniform | 50 | 683 | 693 / 706 | 3001 |
| MSVR310/ambiguity | 1 | 95 | 1 / 14 | 3088 |
| MSVR310/ambiguity | 38 | 539 | 524 / 537 | 3791 |
| MSVR310/ambiguity | 50 | 683 | 693 / 706 | 4019 |

## Findings retained and correction acknowledgments

### F1: Original correction ratio definition names the wrong output field — WARN

Original SUMMARY and executed-source definition say raw_features. The actual producer computes mean(norm(raw_fused-shared_global)/norm(shared_global)) per batch. The consumer always read the correct saved actual_scaled_correction_global_ratio_mean field, so numbers are unchanged. The explicit addendum and final report correct the definition; original source and SUMMARY retain their exact original bytes.

Evidence: C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\training_trace910\SUMMARY.json:21; C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\analyse_training_trace910.py:69; C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_role_input_detach.py:26; C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_role_input_detach.py:30; C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\global_task_role_heads.py:21; C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\global_task_role_heads.py:31; C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\training_trace910\DOCUMENT_DEFINITION_CORRECTION.json:4; C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\training_trace910\REPORT.md:30.

### F2: Growth wording requires an endpoint qualifier — WARN

RGBNT201 final correction ratios exceed E8 ratios, but uniform and ambiguity have respectively 18 and 17 adjacent declines after E8. Original continues-to-grow wording must not imply monotonicity. Final REPORT explicitly says final exceeds best and that this is a two-record comparison.

Evidence: C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\training_trace910\REPORT_PRE_ENDPOINT_WORDING.md:24; C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\training_trace910\REPORT.md:24; C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\review\training_trace910/CORRECTION_ACKNOWLEDGMENT.json:1.

### F3: Original analysis invocation is not independently verified — WARN

The available receipt is a later context transcription of retained functions cell186 / terminal chunk d8686c / exit0 / 1.2817s, not the original call or complete stdout. Source hash and exact reaggregation establish content consistency only. No original producer or renderer was rerun to manufacture provenance.

Evidence: C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\training_trace910\EXECUTION_PROVENANCE_LIMIT.json:2; C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\training_trace910\EXECUTION_PROVENANCE_LIMIT.json:9; C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\training_trace910\EXECUTION_PROVENANCE_LIMIT.json:10; C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\training_trace910\SUMMARY.json:4; C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\review\training_trace910/SCALAR_CHECK_NATIVE_RESULT.json:1.

The definition correction passed 7 targeted checks (native chunk d6608d, exit0). Original report c65f575… is retained byte-exact. Its corrected 83eaa719… version is retained byte-exact as REPORT_PRE_ENDPOINT_WORDING.md. The final endpoint wording changes only line24; all numeric rows are identical. Final REPORT SHA256: `6fa57aebb07bed453336ca2a7774c03d14df69af10e792c5c5b93e907e3703fc`. Source/SUMMARY were not rewritten. Parent-reported helper failure 9d3d12 (newline encoding mismatch) and recovery 242065 were reported as later metadata, not promoted to auditor-observed native proof.

## Integrity checks A–F

### A. Ground-truth provenance: WARN

At source level legal relations use identity labels and cross-environment positives; MSVR310 uses scene and other datasets camera. The consumer copies stored official_fused values without generating labels. Original dataset images, labels, feature binaries and evaluator execution were not revalidated in this bounded audit.

Evidence: C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\region_use_objective.py:21; C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\region_use_objective.py:23; C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_region_use_probe.py:27; C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_region_use_probe.py:107; C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_region_use_probe.py:111; C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_foundation_recipe.py:119; C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_foundation_recipe.py:224; C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\analyse_training_trace910.py:49.

### B. Score normalization: PASS

The nonpositive-margin fraction is a count divided by all legal training-triplet exposures, not retrieval accuracy. Correction/global norm ratio is explicitly a model-derived magnitude diagnostic with a model norm denominator; it is not an AP/rank score. Auxiliary loss mixes detached global and regional margins with per-query legal-weight normalization and therefore cannot alone establish improved regional relationships.

Evidence: C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\region_use_objective.py:18; C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\region_use_objective.py:24; C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\region_use_objective.py:27; C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\region_use_objective.py:30; C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\region_use_objective.py:33; C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\region_use_objective.py:35; C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_role_input_detach.py:30; C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\analyse_training_trace910.py:37; C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\analyse_training_trace910.py:47; C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\training_trace910\REPORT.md:5; C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\training_trace910\REPORT.md:22.

### C. Result existence and arithmetic: PASS

All 12 requested saved inputs exist and match both intake and SUMMARY hashes. All 200 epoch records and 12 first/selected/last records match independently recomputed scalars exactly. All four receipts and training files identify COMPLETE/full50/seed42; selected epochs are actual maximum stored mAP with latest tie rule. Receipt metrics agree with selected history within the producer tolerance; this does not certify remote binaries.

Evidence: C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903/results908/INTAKE_MANIFEST_908.json:3; C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\training_trace910\SUMMARY.json:5; C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_foundation_recipe.py:252; C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_foundation_recipe.py:296; C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_foundation_recipe.py:301; C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\review\training_trace910/EPOCH_EVIDENCE.json:1.

### D. Producer execution and live code: WARN

Source call chain is live: region_use_probe configures inner.loss_values, calls region_use_loss, and delegates train; global task calls detach logger; foundation train writes components to JSONL. Saved schemas, entry hashes and components correspond. The analysis main-level write path exists and its content is reproduced independently, but historical actual execution lacks original transcript. No neural code was imported/executed here.

Evidence: C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_region_use_probe.py:56; C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_region_use_probe.py:58; C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_region_use_probe.py:60; C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_region_use_probe.py:84; C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_region_use_probe.py:122; C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_global_task_role.py:26; C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_role_input_detach.py:22; C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_foundation_recipe.py:232; C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_foundation_recipe.py:243; C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\analyse_training_trace910.py:75; C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\training_trace910\EXECUTION_PROVENANCE_LIMIT.json:10.

### E. Scope: PASS

Qualified claim ceiling is descriptive saved-source training metrics for four seed42 endpoints on RGBNT201/MSVR310. No RGBNT100/all-three, unseen-test, statistical significance, stable benefit, teacher, gain/seed rescue, regional standalone retrieval or unique causal mechanism claim is supported. Official query/gallery was used per epoch and for best/development selection. Sections908/909 and comparison909 explicitly maintain these limits.

Evidence: C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\training_trace910\SUMMARY.json:24; C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\training_trace910\REPORT.md:3; C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\training_trace910\REPORT.md:24; C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\training_trace910\REPORT.md:26; C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\training_trace910\REPORT.md:28; C:/Users/gb/Desktop/document/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md:18368; C:/Users/gb/Desktop/document/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md:18377; C:/Users/gb/Desktop/document/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md:18386; C:/Users/gb/Desktop/document/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md:18388; C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903/comparison909/SUMMARY.json:12.

### F. Evaluation type

Stored official_fused metrics are prior real_gt-scored receipts; they were copied and compared, not re-evaluated. Global-margin sign and correction/global magnitude are model-dependent training diagnostics, classified here as synthetic_proxy for claim limits; they do not supply ground truth or retrieval accuracy. Legal identity/environment relations use source labels. No image/data/feature/checkpoint binary provenance was recertified.

## Qualified claims and blockers

- C1 — supported: All 200 saved epoch aggregates, four run totals and 12 displayed first/selected/last records match the 12 audited input files.
- C2 — supported_with_qualifier: RGBNT201 has much lower logged global-nonpositive training exposure fraction at E8 and zero at E50; final correction ratio exceeds E8 while recorded fused mAP is lower. This describes saved endpoints, not monotonicity or causality.
- C3 — supported_with_qualifier: MSVR310 retains nonpositive-margin training relations and roughly 0.005 correction/global norm ratios in these saved runs. This is not a retrieval score or diagnosis of the failure cause.
- C4 — unsupported: Historical analysis producer invocation independently verified from its original native call/stdout.
- C5 — unsupported: All-three-dataset completion, untouched test, Signal recertification, statistical/stable gain, causal failure attribution, strong teacher, region-only retrieval superiority or gain/seed justification.

One unresolved provenance blocker remains: the claim that the original historical consumer invocation has been independently verified. Qualified saved-text content use is supported. A full runtime or Signal/NN recertification would be outside this audit; absence of that work must not be converted into a PASS.

## Audited input hashes

All values below are SHA256. Historical and final report versions are listed separately in EXPERIMENT_AUDIT.json.

| Input | SHA256 |
|---|---|
| `C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\training_trace910\SUMMARY.json` | `580692a14417fc49453d5bfa90225466a4470338ab95b1054d8848c67e310122` |
| `C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\results908\INTAKE_MANIFEST_908.json` | `e756efe47fcedf654076f4d118d37d5cab61f69499fbf66663552878f0a1cc17` |
| `C:\Users\gb\.trifusion_github_publish_22c3bee\refine-logs\region_use_supervision_v1\INPUT_SEAL.json` | `d9c56b40c377d658fcc879a8db538a0c585e54219e83c7fd6755411dbaeff7ed` |
| `C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\analyse_training_trace910.py` | `008b9258270f08b769d8e438a1cb6a36dc283cb615b118c71306a6d4df33bfb4` |
| `C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\training_trace910\REPORT.md` | `6fa57aebb07bed453336ca2a7774c03d14df69af10e792c5c5b93e907e3703fc` |
| `C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\render_training_trace910.py` | `185f8c4098f43d625a9ee439843daafb3e008e485f635648606608e317cb80c1` |
| `C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\training_trace910\EXECUTION_PROVENANCE_LIMIT.json` | `794987e024bc1b6a341f4175e8b7216ec5f89eaa5da21a57f1904eec50103b87` |
| `C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\comparison909\SUMMARY.json` | `0c18f2968adae23366cb4c6dad65ad7831865df0dee65177d362f20915caf115` |
| `C:\Users\gb\Desktop\document\TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md` | `fadc0d6e47d5116dcac0bd26d43d7dfabd90c84c35c67abefb94369e0543a938` |
| `C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\region_use_objective.py` | `556876002cc7abef5a3525940781b8facdf6a1e8bfc792e5b900f16a5f72fe6d` |
| `C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_region_use_probe.py` | `176e1e88a74ede5447f57cf0e3a89c1a0847e729647d37b11fdb6c76ea715591` |
| `C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_role_input_detach.py` | `a177fd5aa4d5ba25422a49cb619062001722d6c7640891cb3da9645a8eb32fe5` |
| `C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_global_task_role.py` | `151a3350a9da7d86ce6b4b7d5160537b182792ab08727ccb5e4ab0c64bb56d5f` |
| `C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\region_use_heads.py` | `1680d04f7b26f513380509af88bfce555cbc04497162e4af10c47dcbbcc9b63e` |
| `C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\global_task_role_heads.py` | `2298763b70b48718411db52e55f32dabddade5af0e56457a85eeabab4f9fcefd` |
| `C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\role_input_detach.py` | `a3732d5ea98a9fef6039a14f34de98a23cd85a39efd83f52c8835ec59d4605b3` |
| `C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_native_research.py` | `b48725622371fc6d9671c464ce97252e2e06eba465e213ce221d84c798e52cc2` |
| `C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_native_partitioned.py` | `2c3abac37b122760895b97ccf01d4fefe2cfb8223b35b6e2e335a1a4144f72fa` |
| `C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_independent_native_evidence.py` | `2429af8d2f803d70b9d15113a430609ead28621eb4341a746c3025cc5685fe23` |
| `C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_foundation_recipe.py` | `c18042509dc33ff66eee1a6b04a723ad55685bd5d9daf218c3779703ffe20758` |
| `C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\evidence_author_heads.py` | `613b249dae9e1d62d418b94398a0d6ccbcf48e72c88b26b27dd5b9254e9c24cd` |
| `C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\independent_native_roles.py` | `1565590fdfb281fb51b43d06497cd526ffe5bbdc135727ebaa609615dc7832ff` |
| `C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\results908\trained-model\region_use_v1_20261012_906_full_uniform_RGBNT201\training.json` | `ecaecb28d3190481774c1f1bf36e4548d593b2aeffc59a11826ec324ef5251ab` |
| `C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\results908\trained-model\region_use_v1_20261012_906_full_uniform_RGBNT201\training_steps.jsonl` | `e6480c3e80b1583fb20ce9175dded4e4df6c3a7953ddc2263ee6f37ca71b7dc9` |
| `C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\results908\trained-model\region_use_v1_20261012_906_full_uniform_RGBNT201\official_metrics.json` | `9dafc0a49f18aebf5087be848f027dc07f0c750fde0cd5603bbf3f4d3bfc6829` |
| `C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\results908\trained-model\region_use_v1_20261012_906_full_ambiguity_RGBNT201\training.json` | `50297128605ab50d3058326713f3d59d7925534dd7a5841d8906437ac48e8beb` |
| `C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\results908\trained-model\region_use_v1_20261012_906_full_ambiguity_RGBNT201\training_steps.jsonl` | `2d4e4a749b88172b2a60127a84e7ec33983e40432036b41d760e04313e7c57b9` |
| `C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\results908\trained-model\region_use_v1_20261012_906_full_ambiguity_RGBNT201\official_metrics.json` | `32e62373d4b7144bb492d9c842b3b35ea6d9d254e552e54893ad18591dfcab9c` |
| `C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\results908\trained-model\region_use_v1_20261012_906_full_uniform_MSVR310\training.json` | `48ffc35cc8d58dedcf676961143b7c0f350c7c3b4cc00b4d531374c23c9505c7` |
| `C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\results908\trained-model\region_use_v1_20261012_906_full_uniform_MSVR310\training_steps.jsonl` | `4a1d218638ecdab37e1bbf4bb6465a5c17d15bed29faf2635c20b5323861e697` |
| `C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\results908\trained-model\region_use_v1_20261012_906_full_uniform_MSVR310\official_metrics.json` | `bdda341ccf8ea705f3e91a76a4ad2226fff09c29e5319a58f924518ddd81dd5d` |
| `C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\results908\trained-model\region_use_v1_20261012_906_full_ambiguity_MSVR310\training.json` | `6a14f28839197d1853426d5cef31205277babebf2a360f15e2d75536b216548d` |
| `C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\results908\trained-model\region_use_v1_20261012_906_full_ambiguity_MSVR310\training_steps.jsonl` | `2805154394e20ad830a334a77d6c205cdf26be947f5d4d6ebc8c4b3717dc9766` |
| `C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\results908\trained-model\region_use_v1_20261012_906_full_ambiguity_MSVR310\official_metrics.json` | `1c5331681c140ac3ccd86bb810bb243aef7c62696fcc60f46adac9dc0ca7fba3` |
| `C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\training_trace910\REPORT_BEFORE_DEFINITION_CORRECTION.md` | `c65f575aa2849962c6d57fa9fd71526c44c6a4ec81a38cf7a837343f484832d0` |
| `C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\training_trace910\DOCUMENT_DEFINITION_CORRECTION.json` | `b50c9813443c78bdc9346832fe110a2b25ed9163e269a76340b611c389cd5624` |
| `C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\training_trace910\DOCUMENT_ENDPOINT_WORDING_CORRECTION.json` | `b7b740aade3aecec438c66d5b1335da5ff3c9465967e8349ab8014858c4001d6` |
| `C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\training_trace910\REPORT_PRE_ENDPOINT_WORDING.md` | `83eaa719e46e037a7e8ba4a9ce56b89471f7c6a23daf811cee8eff9fe394ed08` |

Full reviewer request/response and route metadata are retained under C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\review\training_trace910\trace. Deterministic acceptance applies only to the enumerated content predicates; the semantic verdict remains same-family/provisional.
