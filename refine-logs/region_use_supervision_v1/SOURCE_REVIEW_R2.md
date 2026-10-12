# Source review R2: draft905

**Verdict: FAIL — one medium source diagnostic issue blocks accepting the new M0 activity observer.** The R1 metadata issue is fixed. No other evidenced source bug was found in the R2 diff or unchanged objective/head path. This is not a failed model experiment: no new model or real batch has been run.

The review applies `experiment-audit/SKILL.md` and `shared-references/local-codex-policy.md`. Requested route: `gpt-6-astra`, reasoning `max`. Actually attested model and effort: **unattested by this review session**. Native OpenAI Codex, same-family, provisional. This is the same reviewer following R1, not a second independent reviewer.

## Finding R2-M0-01 (medium/P2)

**Disable autocast around the isolated auxiliary VJP so it observes the production backward context.**

`tools/run_region_use_probe.py:67` calls `torch.autograd.grad` from `loss_values`. The active trainer calls that function **inside** `torch.autocast("cuda", dtype=torch.float16)` (`tools/run_foundation_recipe.py:230–232`), while its real scaled `.backward()` is **outside** the block (`:235`). The loss function's own disabled forward-autocast scope has already returned; it does not disable the subsequent VJP. The gate then makes its activity decision from those observer norms (`tools/accept_region_use_m0.py:23–27`).

This difference has a concrete precedent in this architecture. The raw existing `refine-logs/incremental_role_objective_v1/AUTOCAST_CONTEXT_RESULTS_20261007_885.json:211–330` contains two VJPs of the same retained graph/targets at scale1. All six Q/K gradients are nonunused but zero with backward autocast enabled; all six are positive with it disabled. The disabled norms are `4.7441e-6, 2.6215e-7, 1.4647e-7, 4.7928e-6, 2.3798e-7, 1.4269e-7`. The diagnostic source changes only that context (`tools/diagnose_initial_aux_autocast_context.py:48–64`). The existing checked observer disables it explicitly (`tools/run_incremental_role_objective_checked.py:31–43`).

The new gate therefore measures an auxiliary backward under a different numerical context from the actual updates. Its acceptance or rejection may reflect the observer context. The minimal fix is a local `with torch.autocast("cuda", enabled=False):` around only the isolated VJP. Keep scale256, targets, retained graph, loss math, forward autocast, original scaler and main backward unchanged. No fallback or repeatability experiment is requested.

**Evidence limit:** the historical test used repair_keep, scale1 and six Q/K targets; draft905 uses the new actual-region objective, scale256 and nine Q/K/V targets. This review does **not** claim measured draft905 zeros, a guaranteed numerical outcome after the fix, a production disconnect, or any retrieval result.

## Other reviewed behavior

| Check | Source verdict | Evidence |
|---|---|---|
| R1_metadata | FIXED | tools/run_region_use_probe.py:48-53 |
| actual_capture_and_objective | PASS_SOURCE_UNCHANGED | modeling/trifusion/region_use_heads.py:7-29; modeling/trifusion/region_use_objective.py:6-39; modeling/trifusion/independent_native_roles.py:63-74; modeling/trifusion/correspondence_evidence_readout.py:65-77 |
| current_model_builder_lifetime | PASS_SOURCE_AND_CONTROL_STUB | tools/run_region_use_probe.py:18-24; tools/run_region_use_probe.py:34-45; tools/run_region_use_probe.py:79-89; tools/run_foundation_recipe.py:193-198; tools/run_foundation_recipe.py:263-284; tools/run_independent_native_evidence.py:126-174 |
| nine_QKV_parameter_names | PASS_SOURCE_AND_CONTROL_STUB | tools/run_region_use_probe.py:62-66; modeling/trifusion/patch_memory_roles.py:19-22; modeling/trifusion/slot_competition_fp32_roles.py:10-20 |
| scale_graph_grad_ownership | PASS_SOURCE_EXCEPT_CONTEXT_FINDING | tools/run_region_use_probe.py:61-76; tools/run_foundation_recipe.py:198; tools/run_foundation_recipe.py:229-241 |
| original_loss_BN_optimizer_ownership | PASS_SOURCE_UNCHANGED | tools/run_global_task_role.py:25-31; modeling/trifusion/global_task_role_heads.py:8-32; modeling/trifusion/role_input_detach.py:5-12; modeling/trifusion/evidence_author_heads.py:63-73; tools/run_foundation_recipe.py:194-198; tools/run_foundation_recipe.py:229-241 |
| hook_and_observer_memory_lifetime | PASS_SOURCE_RUNTIME_MEMORY_UNVERIFIED | modeling/trifusion/region_use_heads.py:8-29; tools/run_region_use_probe.py:66-76; tools/run_foundation_recipe.py:221-245 |
| M0_receipt_and_text_gate | PASS_SOURCE_AND_TEXT_FIXTURES_DEPENDENT_ON_OBSERVER_FIX | tools/accept_region_use_m0.py:8-36; tools/run_foundation_recipe.py:229-284; tools/run_independent_native_evidence.py:138-174; SOURCE_ONLY_CHECKS_R2.json:acceptance_gate_in_memory_text_fixtures |
| initializer_CLI_reload_evaluation | PASS_SOURCE_UNVERIFIED_RUNTIME | tools/run_region_use_probe.py:34-53; tools/run_region_use_probe.py:92-121; tools/run_clean_clip_joint.py:37-107; tools/run_native_partitioned.py:23-30; tools/run_foundation_recipe.py:103-107; tools/run_foundation_recipe.py:174-186; tools/run_foundation_recipe.py:263-312; tools/run_correspondence_roles.py:60-107 |
| plan_and_learning_object_scope | PASS_SCOPE_WITH_EXPLICIT_LIMITS | refine-logs/region_use_supervision_v1/EXPERIMENT_PLAN.md:3-30; tools/diagnose_same_evidence_comparison.py:72-100; refine-logs/evidence_comparison_diagnosis_v1/EXPERIMENT_PLAN.md |

**R1_metadata.** auxiliary_tasks now explicitly records actual_region_identity_relation. It flows into the existing training/checkpoint/evaluation conditions. R1 finding is closed for R2 only; R1 bytes/report remain immutable.

**actual_capture_and_objective.** Byte-identical objective/head to R1. Captured cnn/transformer/mamba are the final evidence fields consumed by read_evidence, stacked without detach into Bx3x3x16x128. Nine group-local symmetric mean MaxSim scores and legal same-ID/different-environment positives plus different-ID negatives are unchanged. Global margin/weights detach, FP32 loss, sqrt128 scale, legal mass and equal eligible-query normalization remain as audited. Empty support stays zero; no training labels enter inference.

**current_model_builder_lifetime.** The active rebinding chain still calls the new builder. CURRENT_MODEL is the same instance returned for training. The fresh reload builder replaces it only after all eight loss calls/updates; reload then calls return_aux=False and does not invoke loss_values. M0_DIAGNOSTICS holds its original model separately. No stale-current-model bug is evidenced in the active CLI path. The control stub confirms current-model replacement and parameter targeting, without building a neural model.

**nine_QKV_parameter_names.** Three role-specific query, three key and three value projection Linear layers are bias=False: exactly nine weight tensors with the selected names. Prefix filtering excludes output projections/readout, shared model and author heads. Source shows these projections feed the sampled role values used downstream. Actual nonzero activity remains unverified.

**scale_graph_grad_ownership.** The observer is m0-only. autograd.grad returns separate gradients, retains the original graph and neither writes .grad nor calls an optimizer/scaler. Shared-global target must be unused; every returned non-None gradient is checked finite. Gradients are promoted to FP32 for norm computation then divided by256 into JSON scalars. Diagnostic256 is a separate constant matching the trainer initial scale; it does not mutate the GradScaler. Non-m0 formal loss still evaluates original+auxiliary exactly once. The actual first-eight scaler states and numerical gradients were not executed.

**original_loss_BN_optimizer_ownership.** Original global/raw-role losses and detached role input path are unchanged. Auxiliary gradients are confined to role evidence, with no global/head/readout/gain direct path. Fused functional author heads still detach parameters and clone buffers; real global heads own BN updates. No new parameter or buffer, parameter ownership change, optimizer/scheduler action, or new forward is introduced by the observer.

**hook_and_observer_memory_lifetime.** One hook per model; auxiliary capture resets flag and stored reference on successful forward. Observer tensors are local to loss_values, only Python scalar diagnostics survive in components; the retained graph is used by the original backward. No list of graphs accumulates. Extra VJP memory/time is not measured. There is no evidence for adding an exception fallback to this fail-stop path.

**M0_receipt_and_text_gate.** Parser matches existing training.json and eight flat training_steps.jsonl rows. M0_PASS is emitted only after the native finite-gradient, all-parameter activity, frozen/visual/camera checks and strict state/buffer reload. The post-step observer then requires eight effective updates and eight author-BN batches before adding production_m0_diagnostics. The new gate consumes those native receipts and requires any positive support/actual-region norm plus each of nine stable names having a positive norm across the eight rows. Because norms are nonnegative, any-positive is the intended cumulative-activity predicate. It hashes both input text files and refuses output overwrite. Nine in-memory text fixtures accept distributed activity and reject empty support, zero regions, one inactive QKV tensor, incomplete/native-failed receipts, global gradient present and inconsistent names. No fixture qualified a real M0. This is a gate over the trusted producer receipts, not an independent replay or generic adversarial-receipt validator.

**initializer_CLI_reload_evaluation.** CURRENT_MODEL assignment and the observer add no constructor randomness/state/parameters. The same public initializer and original model hierarchy remain active; entry hash binds the modified observer source. CLI path resolution, condition binding, strict full state reload and dataset-specific evaluation parsers remain as R1. M0 reload/inference does not need the environment field. Actual initialization state/parameter/RNG/output equality, CUDA/import behavior, full M0/reload and queue/controller/storage qualification remain pending.

**plan_and_learning_object_scope.** Plan correctly labels a new source draft, two fixed controls and pending registration/NN. Known/repeated mechanisms are architecture/readout, symmetric MaxSim, softplus margin, detached global difficulty sigmoid and a training-only auxiliary loss. Changed learning object is actual final role-value relation rather than historical keys or fused/residual vectors. It remains an intermediate score, not the deployed additive1536 score; local loss/score improvement cannot imply fused retrieval improvement. No novelty or performance guarantee is made. R2 diagnostic wording must be read subject to the context finding.

## Verification and unverified work

Read the exact R2 diff and preserved the five R2 draft files under `r2_inputs/`. All five hashes match `SOURCE_SYNTAX_905_R2.json`; objective/head bytes match R1. All 73 unchanged previously audited inputs were hash-rechecked. R1 reports/snapshots remain immutable. `SOURCE_ONLY_CHECKS_R2.json` records the actual function AST control checks and nine in-memory text fixtures. These use fake Python scalars/model placeholders, not tensor or neural execution. They demonstrate wiring and parser predicates only. No synthetic gate output was promoted as a real M0 receipt.

Review actions: 0 neural forwards, 0 GPU queries, 0 weights/tensor-cache loads, 0 optimizer updates, 0 production modifications, 0 numerical objective reruns. The existing historical context result was read as JSON only.

Real initializer state/parameter/RNG/global-output equality, actual sampled support, scale256 full-model gradients, eight effective M0 updates, strict full-state reload, runtime memory/time and controller/storage/queue qualification remain pending. Source approval would not launch NN or establish any performance improvement.

The source reuses MaxSim, soft-margin/softplus, detached difficulty weights, and the existing architecture/readout. The changed supervised object is the final role-value relation. That comparison is not the deployed additive1536 score. Improved region loss or local retrieval cannot establish fused benefit; this is not a novelty audit or guarantee.

## Audited file hashes and line coverage

Inherited R1 source coverage below was retained and all unchanged bytes reverified in R2. The R2 direct changes and contextual evidence are marked in JSON. The old local `experts/mamba.py` remains CRLF; its LF-normalized hash matches the sealed903 pin (both hashes retained in JSON). No byte-identity claim is made for that EOL-only source.

| File | SHA256 | Reviewed lines |
|---|---|---|
| `C:/Users/gb/.trifusion_github_publish_22c3bee/AGENTS.md` | `55a0361490d0bbc2b1c13d337e46c44ca0abddcb4f7287ba5f5700d8e363a1b2` | 1-105; 316-321; applicable current-priority directives |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/CURRENT_GOAL.md` | `7e69f7af9273dede4c0f3f819c1a7ecb65dd93d262275c6e8fa9c3bc8864a166` | 1-33; 121-155 (dated priority boundaries, not result re-audit) |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/region_use_objective.py` | `556876002cc7abef5a3525940781b8facdf6a1e8bfc792e5b900f16a5f72fe6d` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/region_use_heads.py` | `1680d04f7b26f513380509af88bfce555cbc04497162e4af10c47dcbbcc9b63e` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_global_task_role.py` | `151a3350a9da7d86ce6b4b7d5160537b182792ab08727ccb5e4ab0c64bb56d5f` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_role_input_detach.py` | `a177fd5aa4d5ba25422a49cb619062001722d6c7640891cb3da9645a8eb32fe5` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_native_research.py` | `b48725622371fc6d9671c464ce97252e2e06eba465e213ce221d84c798e52cc2` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_native_partitioned.py` | `2c3abac37b122760895b97ccf01d4fefe2cfb8223b35b6e2e335a1a4144f72fa` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_independent_native_evidence.py` | `2429af8d2f803d70b9d15113a430609ead28621eb4341a746c3025cc5685fe23` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_foundation_recipe.py` | `c18042509dc33ff66eee1a6b04a723ad55685bd5d9daf218c3779703ffe20758` | 1-313; parser cross-check |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_clean_clip_joint.py` | `178e9793f2cf8261eb44a278e053d5caabd4f2f37dba48a65952b2547338a746` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_visual_update_control.py` | `537cd5a4ab4e9b3501a7279e3f4857536bdda761f597d8b5fa7bfdbdbf82cbdd` | all (active imported bindings distinguished from unused legacy loop) |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_training_feature_scale.py` | `473853f58e4ed70bf123509afe2861fa8a9988631c04f0ce04f3260c55fd9081` | all (BatchOrderLoader is the active imported object) |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_incremental_role_objective.py` | `76ad1ec53ec4d87df11c12fb89fd32aeb64174505136ea59d61a184d7e3b7e0a` | all (historical mechanism/parser reference only) |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/global_task_role_heads.py` | `2298763b70b48718411db52e55f32dabddade5af0e56457a85eeabab4f9fcefd` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/role_input_detach.py` | `a3732d5ea98a9fef6039a14f34de98a23cd85a39efd83f52c8835ec59d4605b3` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/independent_native_roles.py` | `1565590fdfb281fb51b43d06497cd526ffe5bbdc135727ebaa609615dc7832ff` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/evidence_author_heads.py` | `613b249dae9e1d62d418b94398a0d6ccbcf48e72c88b26b27dd5b9254e9c24cd` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/role_global_tokens.py` | `f41ff1af1fb1b6303a9af63cf039263e2572c16b23813f92a0297f014e71dd4f` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/slot_competition_fp32_roles.py` | `810801e048bcb07ddbdef0635708b63cba7562a4876e044af34135fd412f79c8` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/slot_competition_roles.py` | `bbc3e2430f760587cf8c1d11e40bcbdf0459424e975de74b5b15a40706efbe99` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/patch_memory_roles.py` | `833c8c93363cee7a4fc74c6c458194b484dc1b8cac0f9ec58113a07ac89cb23d` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/correspondence_context_identity.py` | `eda3d1473496b33e4b69402dba7770e67d23b33b07afd41d23105b7f148d63af` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/correspondence_evidence_readout.py` | `e737a977d39ccbe09a5509118e5b5333d653f7f13381e65e84af1a13e5c28dc4` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/correspondence_roles.py` | `e2a51154b5b327341f35e453dd32a821b091a49a669b49a7c70e79e6e2f2591b` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/partitioned_evidence_clip.py` | `3bad53abd6777602e1f1ea9e7b03f10a796116b577501ee5ec0aec6e683b436c` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/experts/mamba.py` | `8b9cb420c42e4d70f8de7e4608637c81c505b97ca54228175462a7e92fdfcc83` | all; production factory 29-32 is the active path |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/aligned_data.py` | `3ea362d17660483b554cb599442b6377ace020fa114969b9bdd58906fbceedd5` | 76-94; class/return discovery |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_correspondence_roles.py` | `e50865fb923297cd61cf38b33ec2bc95154f8c503b5dfe5de9823cad3f03d7ef` | 1-108 |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/official_three_dataset_data.py` | `c3180a88d12440709319d2025928fc699ff3b7c020b11697d468453fbfc68062` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/official_three_dataset_model.py` | `e103ea0f4d6342b0dc129dc2dace0e8c0d0e6c956cfe94729ccb70d9588cf0f9` | all; ROLE_CONFIGS/sha are active imports, legacy checkpoint builder is not called |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_official_three_dataset_roles.py` | `f30bb00a0b71429e9d11dd196cd1d543202a133c663ad66e175d2f82e28d448d` | 1-150; 230-237 |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_signal_preserving_v5.py` | `e162184f68778b4991db0f97f26c5fda273b2ad2f7c8db2bbb2d53775eb717e5` | 1-210; 257-267; 523-530; 568-584 |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_signal_baseline_dev.py` | `083967f2a38267415b2992da98c2ad9429ebb793f387b45daac1dbb4eb16f365` | 1-64 |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/build_v12_complete_path_oof_targets.py` | `999356e18d489dce99042f1e7873eac15b811b64255f1aee9550c52ae1c7ee06` | imports/signatures; 235-258 |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/train_rgbnt100_signal_oof.py` | `4462b73139e034d450d955c5b4a994aaa967548e9cc503d97bb753a14ea03b22` | 1-215; 253-268 |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/train_msvr310_signal_oof.py` | `c25579d931df34481fef321558f0b9a1f9179d72a25d6047dbbfd3682d4ae60f` | 80-99; 223-238; loader/signature discovery |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/train_signal_preserving_v18.py` | `f6bdda4631d710d0b6db5fe2a8df124d8dcbd294cb54adcf3ec261d523e96cf8` | 33-49; import/signature discovery |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_same_evidence_comparison.py` | `71fe059e26744c670661cab6fb3bbaaf13da56a5567adf78b8079e2ba0db9ad5` | all; read only, not rerun |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/evidence_comparison_diagnosis_v1/EXPERIMENT_PLAN.md` | `0ea17e67772806f1f6fbbb2082c07fda4b5b380efa1da5195c17a064976e9bf4` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/evidence_comparison_diagnosis_v1/PRIMARY_EVIDENCE_COMPARISON_902.md` | `fb3760dcdedc2a631c35943a9dfebfb4e56885247333d366f5b7c5c62240f34a` | 1-9; 79-147; source-bound design comparison, no fresh literature audit |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/evidence/msvr310_history_gradient_q1_complete_20260908/independent_audit/remote_sources/comparators/Signal-cd1b0a6/solver/make_optimizer.py` | `ada1023a79fcaaed092bce0849a61db9d0253cf8974c494bfb3082d0fd35fee1` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/evidence/msvr310_history_gradient_q1_complete_20260908/independent_audit/remote_sources/comparators/Signal-cd1b0a6/layers/make_loss.py` | `7eb7ab953c2bbc52d0c298ef971d0c3ff7000e5e50756fbb5a5fd63ac4c109aa` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/evidence/msvr310_history_gradient_q1_complete_20260908/independent_audit/remote_sources/comparators/Signal-cd1b0a6/data/datasets/sampler.py` | `b77902214633a2b06a7c0a80924a737fdb5122984a857f3c6e120a8ad8cce65a` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/evidence/msvr310_history_gradient_q1_complete_20260908/independent_audit/remote_sources/comparators/Signal-cd1b0a6/data/datasets/bases.py` | `48f050a7924cebd1e51f8831ba05df78dbccc7049afb0bcf83351d9d1ee23814` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/evidence/msvr310_history_gradient_q1_complete_20260908/independent_audit/remote_sources/comparators/Signal-cd1b0a6/data/datasets/make_dataloader.py` | `54c47d798a20955bde224d9c78b42149c3c2ef008b02659141829a553194434f` | 1-90; 125-186 (collate 142-182) |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/evidence/msvr310_history_gradient_q1_complete_20260908/independent_audit/remote_sources/comparators/Signal-cd1b0a6/configs/MSVR310/Signal.yml` | `357ca3a989e943ea5348e6206c0d0f69b3c0d35ad9fbadbf68d2786439b8a915` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/evidence/msvr310_history_gradient_q1_complete_20260908/independent_audit/remote_sources/comparators/Signal-cd1b0a6/utils/metrics.py` | `91604acb7d978462c16904910a645d7d1697c15bf150da529099004c8b5eecb2` | 1-175 |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/evidence/msvr310_history_gradient_q1_complete_20260908/independent_audit/remote_sources/comparators/Signal-cd1b0a6/modeling/make_model.py` | `1a88a91d2e5055f5d338996231b16c0e13b2fa7013c74131ca663f16c2655963` | 1-90; constructor/forward function discovery |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/evidence/smooth_ap_m0_audit_20260909/snapshots/remote/root/autodl-tmp/trifusion-v2/comparators/Signal-cd1b0a6/modeling/__init__.py` | `c1962dec3f8aa828bd7237baf6edd7b4862cdbd8e9f710170f37f3d61bf1d1c0` | all |
| `D:/Program Files/UserCache/gb/codex_tmp/foundation_recipe_v1_20261002/source_intake737/comparators/Signal-cd1b0a6/configs/RGBNT201/Signal.yml` | `a13bd5169a4c710fa79efc87db909409015a0b2611086c0fbf4c1842f0cfcc44` | all |
| `D:/Program Files/UserCache/gb/codex_tmp/foundation_recipe_v1_20261002/source_intake737/comparators/Signal-cd1b0a6/configs/RGBNT100/Signal.yml` | `02641ce2941dcd50df77d379aab54650cae4a2a6b15ba1bcd43deba970b6900f` | all |
| `D:/Program Files/UserCache/gb/codex_tmp/foundation_recipe_v1_20261002/source_intake737/comparators/Signal-cd1b0a6/modeling/meta_arch.py` | `8b709d5ab4b14ba150a96bbb5e8cc7e6dbbb71dd8bd30cafd7a5cb110ee5a757` | all |
| `D:/Program Files/UserCache/gb/codex_tmp/foundation_recipe_v1_20261002/source_intake737/comparators/Signal-cd1b0a6/config/defaults.py` | `40257d349153e56e345983c99b1638d7c3d77966dfd2f5496fe03c86dc8dbd57` | 34-38; 53; 106; 125-126; 141 |
| `D:/Program Files/UserCache/gb/codex_tmp/foundation_recipe_v1_20261002/source_intake737/comparators/Signal-cd1b0a6/layers/triplet_loss.py` | `f502c44bec642bbf72ce6d0f58c9f9a0084beef9cbb451c2d2786bb13f94960a` | all |
| `D:/Program Files/UserCache/gb/codex_tmp/foundation_recipe_v1_20261002/source_intake737/comparators/Signal-cd1b0a6/layers/softmax_loss.py` | `92192d9c3ff4e2fd3920c6e9a9637e0caa1e66cffbbc0be8b4049e5dcdb0bf24` | all |
| `D:/Program Files/UserCache/gb/codex_tmp/foundation_recipe_v1_20261002/source_intake737/comparators/Signal-cd1b0a6/layers/center_loss.py` | `c2e23be80b1b3e54503c704dda45ddc4a86f23647fcc8c37c63dbac47d003d89` | all (construction only; center loss not added) |
| `D:/Program Files/UserCache/gb/codex_tmp/independent_evidence_draft/evidence_comparison903/REGION_USE_COMPONENT_CHECK_905.json` | `363d71154d6fe02f8e161474c7b64062305706b4a67c494292a7af08380fe6b1` | all |
| `D:/Program Files/UserCache/gb/codex_tmp/independent_evidence_draft/evidence_comparison903/SOURCE_SYNTAX_905.json` | `8e9399716e6f01c72057ba50ee18807b5bfed51c74b775a454acf5011b72136b` | all |
| `D:/Program Files/UserCache/gb/codex_tmp/independent_evidence_draft/evidence_comparison903/region_use_probe905.py` | `23a23a2edb59fe17a132daecad629862fca60edb8b39a6e35203bd2d73e7bb44` | all |
| `D:/Program Files/UserCache/gb/codex_tmp/independent_evidence_draft/evidence_comparison903/check_region_use_probe905.ps1` | `d4f8d1518c7524e2320d5d107d48977c342362889127a5c82b8dd75e3fdf3a28` | all |
| `D:/Program Files/UserCache/gb/codex_tmp/independent_evidence_draft/evidence_comparison903/INPUT_SEAL_903.json` | `3e6eb3be936f0f8bbd695fe6c797413613b746444768de42a41cbf0ac2b1f2ab` | schema, source hashes, relevant artifact/path fields only |
| `C:/Users/gb/.codex/skills/experiment-audit/SKILL.md` | `0ab77f219f554cbe42f22b79d780528e61c5e58169af4c2f56acf59596ab8db7` | all |
| `C:/Users/gb/.codex/skills/shared-references/local-codex-policy.md` | `b0249bfccd48ff79a5976afa7a54c64a3cfb65a5496a2105c5bb21069d197783` | all |
| `C:/Users/gb/.codex/skills/shared-references/reviewer-independence.md` | `1aa1756a99dc47009072f0ee7bf570d797bf7e2cc84439401e2a9312d73659b0` | all |
| `C:/Users/gb/.codex/skills/shared-references/experiment-integrity.md` | `22150fbdc5867e7790c88f968bccb367079c2db4c190ddce980709aea7e76f60` | all |
| `C:/Users/gb/.codex/skills/shared-references/review-tracing.md` | `7f556fb9ef8922cfd0d047e5a2d857b0c7f9408492b8c8dcec00207e76405395` | all |
| `D:/Program Files/UserCache/gb/codex_tmp/metric_feature_scale_failed_closeout766/source/logs/training_feature_scale_protocols_20261002/RGBNT201.json` | `b2409c2d992d04adcf0c78354d306afd6af7ae9507738fd4e83de480e9a9b90f` | complete records parsed as metadata only |
| `D:/Program Files/UserCache/gb/codex_tmp/metric_feature_scale_failed_closeout766/source/logs/training_feature_scale_protocols_20261002/MSVR310.json` | `015a3cafef8e36ba71da2784de6835959c0549c442624c8fb9e9b37bd8c8229d` | complete records parsed as metadata only |
| `D:/Program Files/UserCache/gb/codex_tmp/metric_feature_scale_failed_closeout766/source/logs/training_feature_scale_protocols_20261002/RGBNT100.json` | `b2c14adf947b06b27b9576a712f96192c1393ad6ed3ec1be5259b63f88300169` | complete records parsed as metadata only |
| `D:/Program Files/UserCache/gb/codex_tmp/independent_evidence_draft/evidence_comparison903/review/source905/check_source_only.py` | `80fcecf0b2989c0a902a8955334dcd75c9fb48c8ef293e9b40a5a39c167482b3` | all |
| `D:/Program Files/UserCache/gb/codex_tmp/independent_evidence_draft/evidence_comparison903/review/source905/SOURCE_ONLY_CHECKS_R1.json` | `db4afa8dd1fa6229e2337a10393cf384ee7f25be004e68c2a90ffbfb7107c163` | all |
| `D:/Program Files/UserCache/gb/codex_tmp/independent_evidence_draft/evidence_comparison903/review/source905/PROTOCOL_TEXT_CHECKS_R1.json` | `505c685de0d2f129d1b6482e4edd2e7434f40faa00d12094f84b08b8ab08ab71` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_region_use_probe.py` | `0c452a9728151e47e9cd5a463daad0c0c8c87925a0a49c5f0afdf0b20140c031` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/accept_region_use_m0.py` | `f378797bcb9c641156de215d38b9db8dc34668c5ac7dad15bf214decd442f3ac` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/region_use_supervision_v1/EXPERIMENT_PLAN.md` | `b4cfc994342dec77611ed1b8e40cce452cdc648de295e01e133f517fa65eb117` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_incremental_role_objective_checked.py` | `70121fe68b6a89830ca98835e84ace604b04986e125881eb571744dc7549c17d` | all, especially 22-44 |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_initial_aux_autocast_context.py` | `1b3f2cc6e0e1efbcb6f6c4435c98df023598c38e98b4e4ebdad55875c466e5f2` | 1-80; especially 45-74 |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/incremental_role_objective_v1/AUTOCAST_CONTEXT_RESULTS_20261007_885.json` | `500ad100eddb067ac7958d79dcb8fafcdf986e3075f7116de3fbd16eb05403ab` | all; raw actual.rows 211-330 parsed |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/incremental_role_objective_v1/CHECKED_VJP_PLAN_20261007_886.md` | `c72fda997b48b4e05264f63b1e3441fa6eab1569803611a385a157ae8131d768` | all |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_prepool_dense_correspondence.py` | `5e8f6c80844be07639415694b8749dec98cecbaf108cebcbb3f1310e690f4c04` | 68-84; context observer comparison only |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_text_semantic_prior.py` | `18fafa9a7157077d7ad22a0359de905163bb17a8eb7b04b8f54c6b6bce436c79` | 228-247; scale256 observer comparison only |
| `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/accept_incremental_m0_support.py` | `e37d93b7263e542012e5020465ccfd569166c022e7e52748bb199f6884ec2fc7` | all; text gate comparison, not a new requirement |
| `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/SOURCE_SYNTAX_905_R2.json` | `ec1e5e10993f412ccc8a9593274dfa3e3a84f0cc1cfb609c24df71a6e0286a01` | all |
| `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/review/source905/SOURCE_REVIEW_R1.json` | `2e79ef674d68cda06d4163d27537ff83513ff2a182cc7f45ccb0db3ea88c1f8a` | R1 findings/checks/hash manifest; retained audit context |
| `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/review/source905/SOURCE_REVIEW_R1.md` | `5f1a58aa824b01a431077669f7e642c5fb645e030507396ff8194c5571a5825c` | R1 findings/boundary; retained audit context |
| `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/review/source905/check_source_only_r2.py` | `16aa4b0dda1002671912a280cf52de6aac56ef511974c961bfe312ee5c1f49da` | all |
| `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/review/source905/SOURCE_ONLY_CHECKS_R2.json` | `aa6efb5f981d175c5eaec63d2e5bc2bba236777fbc3ae754d32a2fc2ae3c7a75` | all |
| `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/review/source905/R1_TO_R2.diff` | `07fab9cde7a54185b755086bdd0d70d72753c2a013b272910b8832261646478a` | all |

Final source hash recheck: 2026-10-12T10:37:22.455375+00:00

Same-family provisional source-only follow-up review. The actual serving model/effort is not independently attested. R2 FAIL concerns the new M0 observer context, not a demonstrated model/retrieval failure. No source verdict qualifies real initialization, M0, queue launch, novelty or performance.
