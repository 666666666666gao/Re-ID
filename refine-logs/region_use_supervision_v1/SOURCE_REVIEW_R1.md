# Draft905 source-only experiment audit — R1

**Verdict: WARN.** No blocking implementation bug was evidenced in the reviewed active source path; one concrete condition-metadata error needs a one-field correction. This is a fresh same-family/provisional source review, not an initialization, M0, production or performance pass.

Date: 2026-10-12T10:16:07.842029+00:00. Requested reviewer: `gpt-6-astra`, effort `max` under the installed policy. Native Codex task: `/root/review_region_use_source905`. The actual serving model and effective effort are not independently attested by this review session; requested values are not represented as observed values.

## Finding

**R1-META-01 — auxiliary_tasks is incorrectly serialized as none (low severity).** `tools/run_region_use_probe.py:45-49` extends `original_condition` without overriding its `auxiliary_tasks` value. `tools/run_independent_native_evidence.py:29-33` supplies `none`, although `tools/run_region_use_probe.py:52-57` adds the region-use loss. This condition is saved in `tools/run_foundation_recipe.py:174-185`, `200-203`, and `302-311`. `SOURCE_ONLY_CHECKS_R1.json` reproduces the false field through the actual configure/condition AST. The comparison field is correct and reload uses the same condition, so this is a provenance error, not demonstrated omission of the loss or a broken reload. Replace this one inherited field with a training-only region-use label before creating run artifacts. R1 source bytes are preserved in `r1_inputs/`.

## Active source path

`run_region_use_probe → run_global_task_role → run_role_input_detach → run_native_research → run_native_partitioned → run_independent_native_evidence → run_foundation_recipe`; initialization continues through `run_clean_clip_joint` and the pinned author constructor, while extraction/evaluation uses `run_correspondence_roles`. The imported legacy ReID-checkpoint builder is not the active clean-public constructor. Source-only AST execution confirmed rebinding to the new head/loss/builder/schema/condition, preservation of the author optimizer/training wrapper, and absolute CLI paths. No full import or neural model was executed.

## Checks

**capture — PASS_SOURCE.** The roles forward hook stacks the exact final cnn/transformer/mamba fields used next by read_evidence. It does not detach or replace the output. Each field is Bx3x16x128; stacking yields Bx3x3x16x128. The original structured readout/gain remains in use.

Evidence: `modeling/trifusion/region_use_heads.py:7-29`; `modeling/trifusion/independent_native_roles.py:63-74`; `modeling/trifusion/role_global_tokens.py:29-61`; `modeling/trifusion/correspondence_evidence_readout.py:65-77`.

**scoring — PASS_SOURCE.** Flattening only role/modality gives nine independent coordinate groups. Each BxBx16x16 interaction takes row and column maxima, averages each direction, then equally averages the nine groups. This matches the existing symmetric mean comparator, with no cross-role/modality matching. Slot permutation invariance is inherent; it is not one-to-one matching or visibility estimation.

Evidence: `modeling/trifusion/region_use_objective.py:6-13`; `tools/diagnose_same_evidence_comparison.py:72-100`.

**legal_support — PASS_SOURCE_AND_PINNED_TEXT.** Positive: same training label and different camera (201/100) or scene (MSVR). Negative: different training label. Self/same-environment positives are excluded. Empty support produces zero numerator/gradient and is excluded from eligible-query count; original author tasks remain active. Actual author sampler batches have multiple identities (201 B64/K8, MSVR B64/K4, 100 B128/K16), so a positive-support query also has negatives. No speculative one-ID branch is needed. Pinned protocol text has no basename/environment conflicts or train/evaluation identity overlap. Actual sampled-batch coverage is pending.

Evidence: `modeling/trifusion/region_use_objective.py:21-39`; `tools/run_region_use_probe.py:26-30`; `tools/run_region_use_probe.py:87-93`; `tools/official_three_dataset_data.py:8-18`; `tools/run_foundation_recipe.py:110-130`; `PROTOCOL_TEXT_CHECKS_R1.json`.

**weighting — PASS_SOURCE.** Global feature is detached before FP32 normalization and both global-margin uses. Scale is sqrt(128); global and regional margins are averaged, coefficient is 1. Legal sigmoid weights are normalized within each query, then queries are equally averaged. This is a defined training-loss normalization, not retrieval-score self-normalization. Supported FP32 sigmoid mass is positive for bounded cosine margins; the existing floor serves the observed empty-support case.

Evidence: `modeling/trifusion/region_use_objective.py:16-35`.

**ownership_and_once — PASS_SOURCE_AND_AST_WIRING.** Full rebinding reaches foundation.loss_values exactly once. It computes the same global + raw-role author losses, then adds one auxiliary scalar. Role inputs and global margin stay detached from shared features. Fused author head parameters are detached; BN buffers are cloned for fused calls, while actual author heads/BN update on the global path. The auxiliary reaches role values, not shared features, author heads, readout or gain directly. Existing author optimizer receives the same named trainable parameters once; no new parameters/optimizer/scheduler/BN are introduced. Future trajectories naturally differ after role updates.

Evidence: `tools/run_region_use_probe.py:18-22`; `tools/run_region_use_probe.py:52-70`; `tools/run_global_task_role.py:25-31`; `modeling/trifusion/global_task_role_heads.py:8-32`; `modeling/trifusion/role_input_detach.py:5-12`; `modeling/trifusion/evidence_author_heads.py:63-73`; `tools/run_foundation_recipe.py:229-241`.

**hook_lifetime — PASS_SOURCE_AND_PYTHON_CONTROL_STUB.** One hook is registered per head object, not per step. Capture is enabled only for return_aux=True; after a successful call the flag is reset and the stored tensor reference cleared. The returned output owns the graph for the usual backward lifetime; no batch list or persistent tensor state accumulates. Ordinary inference does not capture. No exception/retry fallback is warranted by the fail-stop entry.

Evidence: `modeling/trifusion/region_use_heads.py:10-29`; `tools/run_foundation_recipe.py:221-245`.

**initialization — PASS_SOURCE_UNVERIFIED_RUNTIME.** The new subclass adds only ordinary Python flags/reference/hook after the unchanged constructor: no parameter/buffer registration or RNG draw. Same public-CLIP/camera/head construction, seeds, role replacements, state hashes and partitioning are used. Thus equality is expected by source; actual original-vs-new state/parameter/RNG/global-output comparison has not been executed and remains a pre-registration requirement. No weight was opened in this review.

Evidence: `modeling/trifusion/region_use_heads.py:8-12`; `tools/run_clean_clip_joint.py:37-107`; `tools/run_independent_native_evidence.py:36-92`; `tools/run_native_partitioned.py:23-30`; `tools/run_region_use_probe.py:33-42`.

**cli_binding_reload_eval — PASS_SOURCE_AND_AST_PARSER.** Parser creates semantic variant/recipe, seed42,50 epochs and resolves all five paths before chdir. The comparison is included in initializer and condition bindings, with new schema propagated to foundation. Full state_dict is saved and strict-loaded; M0 calls no-aux inference on its small reload batch, so no environment field is needed there. RGBNT201 uses its six-field eval parser; MSVR/100 use the five-field train collator and wrapped metadata parser. Inference ignores label/environment fields and stays normalized1536. Best mAP selection and same-checkpoint CMC evaluation are unchanged. Full CUDA/import/M0/state-buffer/runtime equality is not attested by these AST checks.

Evidence: `tools/run_region_use_probe.py:73-106`; `tools/run_foundation_recipe.py:103-107`; `tools/run_foundation_recipe.py:174-186`; `tools/run_foundation_recipe.py:263-278`; `tools/run_foundation_recipe.py:287-312`; `tools/run_correspondence_roles.py:60-107`; `tools/train_signal_preserving_v18.py:33-49`; `tools/train_rgbnt100_signal_oof.py:99-123`; `tools/train_msvr310_signal_oof.py:89-99`.

**scope_and_learning_object — PASS_SCOPE_WITH_LIMITS.** Known/repeated mechanisms: existing architecture/readout, symmetric MaxSim, sigmoid difficulty weights, soft-margin/softplus triplet loss, and training-only loss addition. The changed supervised object is the actual final role-value comparison, rather than keys or only the fused/residual vector. The score is not the deployed additive1536 cosine, and MaxSim is indifferent to slot rearrangements to which the structured readout can be sensitive. Reduced auxiliary loss or better local retrieval cannot establish fused benefit. Plan states that limitation and makes no new-method/SOTA guarantee. This is not a novelty search.

Evidence: `refine-logs/region_use_supervision_v1/EXPERIMENT_PLAN.md:3-28`; `tools/run_incremental_role_objective.py:62-81`; `tools/diagnose_same_evidence_comparison.py:72-100`.

## Evidence and claim limits

- The two submitted CPU/syntax receipts exist. I independently verified their current three source hashes and exact function AST identity with `region_use_probe905.py`. The supplied gradient numbers were not numerically rerun. In particular, the synthetic test establishes a nonzero total region-input gradient, not per-role/per-parameter activity in the real model.
- The independent check in `SOURCE_ONLY_CHECKS_R1.json` uses the real Python definitions with opaque/scalar interface stubs. It proves control-flow wiring and parser/hook cleanup properties only. It imports no torch and uses no tensors, weights or devices.
- `PROTOCOL_TEXT_CHECKS_R1.json` parses byte-exact903-pinned metadata snapshots for all three training protocols. It confirms basename/environment consistency and train/evaluation identity separation. Full-train cross-environment support is not sampled-batch support.
- The author source checkout is absent at its canonical local path. Relevant optimizer/loss/data/constructor/evaluator/config files were read from historical local text snapshots with exact SHA256 equality to the sealed903 source map; there was no remote runtime query. This verifies the pinned source, not the current server files.
- Full-train metadata supplies legal cross-environment positive support for 51/171 RGBNT201 identities, 60/155 MSVR310 identities and 50/50 RGBNT100 identities. Sampled-batch eligibility may be lower. This is a support limitation of the fixed target, not an implementation defect or a performance prediction; no sampler change is requested.
- Local `modeling/trifusion/experts/mamba.py` uses CRLF. Its actual SHA256 is `8b9cb420c42e4d70f8de7e4608637c81c505b97ca54228175462a7e92fdfcc83`; its LF-normalized SHA256 is `c516c7ad937e5eee6a4ed1e3ec33c2afe3522b751d296bd2e4910e4f27a20ee5`, exactly the sealed903 pin. The manifest marks this as line-ending equivalence, not byte identity.
- Real initial state/parameter/RNG/global-output equality, actual warm imports, eight effective updates, isolated auxiliary activity, full runtime buffer/state reload, storage/controller qualification and all retrieval performance remain UNVERIFIED. The plan explicitly leaves these steps pending; that absence is not a source-draft failure.
- The known comparator/weight/loss primitives are repeated mechanisms. The changed learning object is the actual final region-value comparison. No proof of complementary additive evidence, task consistency, novelty, broader stability or SOTA follows from source or synthetic checks.
- Review actions: zero NN forwards, GPU queries, model/weight/tensor-cache loads, optimizer steps, production modifications, or experiment launches. Only private text audit artifacts were written.

## Experiment-integrity checklist

A: PASS_SOURCE — protocol-derived labels/environments; B: PASS_SOURCE — no retrieval metric self-normalization; C: PASS_QUALIFIED_TEXT_ONLY — source-bound synthetic receipts, no claimed real result; D: PASS_SOURCE_AND_AST_WIRING — active loss path confirmed; E: PASS_DRAFT_BOUNDARY — scope is correctly limited; F: present evidence is simulation-only synthetic objective checking, with future real-GT retrieval unverified.

## Reviewed input SHA256 manifest

The JSON report contains reviewed line coverage and pinned-source aliases for every entry below. Hashing an entire file does not imply every inactive legacy function was semantically audited.

- `C:\Users\gb\.trifusion_github_publish_22c3bee\AGENTS.md` — `55a0361490d0bbc2b1c13d337e46c44ca0abddcb4f7287ba5f5700d8e363a1b2`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\refine-logs\CURRENT_GOAL.md` — `7e69f7af9273dede4c0f3f819c1a7ecb65dd93d262275c6e8fa9c3bc8864a166`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\region_use_objective.py` — `556876002cc7abef5a3525940781b8facdf6a1e8bfc792e5b900f16a5f72fe6d`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\region_use_heads.py` — `1680d04f7b26f513380509af88bfce555cbc04497162e4af10c47dcbbcc9b63e`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_region_use_probe.py` — `7a122fa7c4ef4dae2fe5432b76f6de9633e8ce6e767f3243477ca504ee82f53f`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\refine-logs\region_use_supervision_v1\EXPERIMENT_PLAN.md` — `1bba47aa81db7b03523411bb0ae9063f44db96de38e0ade6da832da92c760bc9`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_global_task_role.py` — `151a3350a9da7d86ce6b4b7d5160537b182792ab08727ccb5e4ab0c64bb56d5f`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_role_input_detach.py` — `a177fd5aa4d5ba25422a49cb619062001722d6c7640891cb3da9645a8eb32fe5`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_native_research.py` — `b48725622371fc6d9671c464ce97252e2e06eba465e213ce221d84c798e52cc2`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_native_partitioned.py` — `2c3abac37b122760895b97ccf01d4fefe2cfb8223b35b6e2e335a1a4144f72fa`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_independent_native_evidence.py` — `2429af8d2f803d70b9d15113a430609ead28621eb4341a746c3025cc5685fe23`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_foundation_recipe.py` — `c18042509dc33ff66eee1a6b04a723ad55685bd5d9daf218c3779703ffe20758`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_clean_clip_joint.py` — `178e9793f2cf8261eb44a278e053d5caabd4f2f37dba48a65952b2547338a746`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_visual_update_control.py` — `537cd5a4ab4e9b3501a7279e3f4857536bdda761f597d8b5fa7bfdbdbf82cbdd`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_training_feature_scale.py` — `473853f58e4ed70bf123509afe2861fa8a9988631c04f0ce04f3260c55fd9081`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_incremental_role_objective.py` — `76ad1ec53ec4d87df11c12fb89fd32aeb64174505136ea59d61a184d7e3b7e0a`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\global_task_role_heads.py` — `2298763b70b48718411db52e55f32dabddade5af0e56457a85eeabab4f9fcefd`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\role_input_detach.py` — `a3732d5ea98a9fef6039a14f34de98a23cd85a39efd83f52c8835ec59d4605b3`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\independent_native_roles.py` — `1565590fdfb281fb51b43d06497cd526ffe5bbdc135727ebaa609615dc7832ff`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\evidence_author_heads.py` — `613b249dae9e1d62d418b94398a0d6ccbcf48e72c88b26b27dd5b9254e9c24cd`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\role_global_tokens.py` — `f41ff1af1fb1b6303a9af63cf039263e2572c16b23813f92a0297f014e71dd4f`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\slot_competition_fp32_roles.py` — `810801e048bcb07ddbdef0635708b63cba7562a4876e044af34135fd412f79c8`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\slot_competition_roles.py` — `bbc3e2430f760587cf8c1d11e40bcbdf0459424e975de74b5b15a40706efbe99`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\patch_memory_roles.py` — `833c8c93363cee7a4fc74c6c458194b484dc1b8cac0f9ec58113a07ac89cb23d`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\correspondence_context_identity.py` — `eda3d1473496b33e4b69402dba7770e67d23b33b07afd41d23105b7f148d63af`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\correspondence_evidence_readout.py` — `e737a977d39ccbe09a5509118e5b5333d653f7f13381e65e84af1a13e5c28dc4`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\correspondence_roles.py` — `e2a51154b5b327341f35e453dd32a821b091a49a669b49a7c70e79e6e2f2591b`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\partitioned_evidence_clip.py` — `3bad53abd6777602e1f1ea9e7b03f10a796116b577501ee5ec0aec6e683b436c`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\experts\mamba.py` — `8b9cb420c42e4d70f8de7e4608637c81c505b97ca54228175462a7e92fdfcc83`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\modeling\trifusion\aligned_data.py` — `3ea362d17660483b554cb599442b6377ace020fa114969b9bdd58906fbceedd5`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_correspondence_roles.py` — `e50865fb923297cd61cf38b33ec2bc95154f8c503b5dfe5de9823cad3f03d7ef`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\tools\official_three_dataset_data.py` — `c3180a88d12440709319d2025928fc699ff3b7c020b11697d468453fbfc68062`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\tools\official_three_dataset_model.py` — `e103ea0f4d6342b0dc129dc2dace0e8c0d0e6c956cfe94729ccb70d9588cf0f9`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_official_three_dataset_roles.py` — `f30bb00a0b71429e9d11dd196cd1d543202a133c663ad66e175d2f82e28d448d`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_signal_preserving_v5.py` — `e162184f68778b4991db0f97f26c5fda273b2ad2f7c8db2bbb2d53775eb717e5`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\tools\run_signal_baseline_dev.py` — `083967f2a38267415b2992da98c2ad9429ebb793f387b45daac1dbb4eb16f365`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\tools\build_v12_complete_path_oof_targets.py` — `999356e18d489dce99042f1e7873eac15b811b64255f1aee9550c52ae1c7ee06`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\tools\train_rgbnt100_signal_oof.py` — `4462b73139e034d450d955c5b4a994aaa967548e9cc503d97bb753a14ea03b22`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\tools\train_msvr310_signal_oof.py` — `c25579d931df34481fef321558f0b9a1f9179d72a25d6047dbbfd3682d4ae60f`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\tools\train_signal_preserving_v18.py` — `f6bdda4631d710d0b6db5fe2a8df124d8dcbd294cb54adcf3ec261d523e96cf8`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\tools\diagnose_same_evidence_comparison.py` — `71fe059e26744c670661cab6fb3bbaaf13da56a5567adf78b8079e2ba0db9ad5`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\refine-logs\evidence_comparison_diagnosis_v1\EXPERIMENT_PLAN.md` — `0ea17e67772806f1f6fbbb2082c07fda4b5b380efa1da5195c17a064976e9bf4`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\refine-logs\evidence_comparison_diagnosis_v1\PRIMARY_EVIDENCE_COMPARISON_902.md` — `fb3760dcdedc2a631c35943a9dfebfb4e56885247333d366f5b7c5c62240f34a`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\evidence\msvr310_history_gradient_q1_complete_20260908\independent_audit\remote_sources\comparators\Signal-cd1b0a6\solver\make_optimizer.py` — `ada1023a79fcaaed092bce0849a61db9d0253cf8974c494bfb3082d0fd35fee1`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\evidence\msvr310_history_gradient_q1_complete_20260908\independent_audit\remote_sources\comparators\Signal-cd1b0a6\layers\make_loss.py` — `7eb7ab953c2bbc52d0c298ef971d0c3ff7000e5e50756fbb5a5fd63ac4c109aa`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\evidence\msvr310_history_gradient_q1_complete_20260908\independent_audit\remote_sources\comparators\Signal-cd1b0a6\data\datasets\sampler.py` — `b77902214633a2b06a7c0a80924a737fdb5122984a857f3c6e120a8ad8cce65a`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\evidence\msvr310_history_gradient_q1_complete_20260908\independent_audit\remote_sources\comparators\Signal-cd1b0a6\data\datasets\bases.py` — `48f050a7924cebd1e51f8831ba05df78dbccc7049afb0bcf83351d9d1ee23814`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\evidence\msvr310_history_gradient_q1_complete_20260908\independent_audit\remote_sources\comparators\Signal-cd1b0a6\data\datasets\make_dataloader.py` — `54c47d798a20955bde224d9c78b42149c3c2ef008b02659141829a553194434f`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\evidence\msvr310_history_gradient_q1_complete_20260908\independent_audit\remote_sources\comparators\Signal-cd1b0a6\configs\MSVR310\Signal.yml` — `357ca3a989e943ea5348e6206c0d0f69b3c0d35ad9fbadbf68d2786439b8a915`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\evidence\msvr310_history_gradient_q1_complete_20260908\independent_audit\remote_sources\comparators\Signal-cd1b0a6\utils\metrics.py` — `91604acb7d978462c16904910a645d7d1697c15bf150da529099004c8b5eecb2`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\evidence\msvr310_history_gradient_q1_complete_20260908\independent_audit\remote_sources\comparators\Signal-cd1b0a6\modeling\make_model.py` — `1a88a91d2e5055f5d338996231b16c0e13b2fa7013c74131ca663f16c2655963`
- `C:\Users\gb\.trifusion_github_publish_22c3bee\evidence\smooth_ap_m0_audit_20260909\snapshots\remote\root\autodl-tmp\trifusion-v2\comparators\Signal-cd1b0a6\modeling\__init__.py` — `c1962dec3f8aa828bd7237baf6edd7b4862cdbd8e9f710170f37f3d61bf1d1c0`
- `D:\Program Files\UserCache\gb\codex_tmp\foundation_recipe_v1_20261002\source_intake737\comparators\Signal-cd1b0a6\configs\RGBNT201\Signal.yml` — `a13bd5169a4c710fa79efc87db909409015a0b2611086c0fbf4c1842f0cfcc44`
- `D:\Program Files\UserCache\gb\codex_tmp\foundation_recipe_v1_20261002\source_intake737\comparators\Signal-cd1b0a6\configs\RGBNT100\Signal.yml` — `02641ce2941dcd50df77d379aab54650cae4a2a6b15ba1bcd43deba970b6900f`
- `D:\Program Files\UserCache\gb\codex_tmp\foundation_recipe_v1_20261002\source_intake737\comparators\Signal-cd1b0a6\modeling\meta_arch.py` — `8b709d5ab4b14ba150a96bbb5e8cc7e6dbbb71dd8bd30cafd7a5cb110ee5a757`
- `D:\Program Files\UserCache\gb\codex_tmp\foundation_recipe_v1_20261002\source_intake737\comparators\Signal-cd1b0a6\config\defaults.py` — `40257d349153e56e345983c99b1638d7c3d77966dfd2f5496fe03c86dc8dbd57`
- `D:\Program Files\UserCache\gb\codex_tmp\foundation_recipe_v1_20261002\source_intake737\comparators\Signal-cd1b0a6\layers\triplet_loss.py` — `f502c44bec642bbf72ce6d0f58c9f9a0084beef9cbb451c2d2786bb13f94960a`
- `D:\Program Files\UserCache\gb\codex_tmp\foundation_recipe_v1_20261002\source_intake737\comparators\Signal-cd1b0a6\layers\softmax_loss.py` — `92192d9c3ff4e2fd3920c6e9a9637e0caa1e66cffbbc0be8b4049e5dcdb0bf24`
- `D:\Program Files\UserCache\gb\codex_tmp\foundation_recipe_v1_20261002\source_intake737\comparators\Signal-cd1b0a6\layers\center_loss.py` — `c2e23be80b1b3e54503c704dda45ddc4a86f23647fcc8c37c63dbac47d003d89`
- `D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\REGION_USE_COMPONENT_CHECK_905.json` — `363d71154d6fe02f8e161474c7b64062305706b4a67c494292a7af08380fe6b1`
- `D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\SOURCE_SYNTAX_905.json` — `8e9399716e6f01c72057ba50ee18807b5bfed51c74b775a454acf5011b72136b`
- `D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\region_use_probe905.py` — `23a23a2edb59fe17a132daecad629862fca60edb8b39a6e35203bd2d73e7bb44`
- `D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\check_region_use_probe905.ps1` — `d4f8d1518c7524e2320d5d107d48977c342362889127a5c82b8dd75e3fdf3a28`
- `D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\INPUT_SEAL_903.json` — `3e6eb3be936f0f8bbd695fe6c797413613b746444768de42a41cbf0ac2b1f2ab`
- `C:\Users\gb\.codex\skills\experiment-audit\SKILL.md` — `0ab77f219f554cbe42f22b79d780528e61c5e58169af4c2f56acf59596ab8db7`
- `C:\Users\gb\.codex\skills\shared-references\local-codex-policy.md` — `b0249bfccd48ff79a5976afa7a54c64a3cfb65a5496a2105c5bb21069d197783`
- `C:\Users\gb\.codex\skills\shared-references\reviewer-independence.md` — `1aa1756a99dc47009072f0ee7bf570d797bf7e2cc84439401e2a9312d73659b0`
- `C:\Users\gb\.codex\skills\shared-references\experiment-integrity.md` — `22150fbdc5867e7790c88f968bccb367079c2db4c190ddce980709aea7e76f60`
- `C:\Users\gb\.codex\skills\shared-references\review-tracing.md` — `7f556fb9ef8922cfd0d047e5a2d857b0c7f9408492b8c8dcec00207e76405395`
- `D:\Program Files\UserCache\gb\codex_tmp\metric_feature_scale_failed_closeout766\source\logs\training_feature_scale_protocols_20261002\RGBNT201.json` — `b2409c2d992d04adcf0c78354d306afd6af7ae9507738fd4e83de480e9a9b90f`
- `D:\Program Files\UserCache\gb\codex_tmp\metric_feature_scale_failed_closeout766\source\logs\training_feature_scale_protocols_20261002\MSVR310.json` — `015a3cafef8e36ba71da2784de6835959c0549c442624c8fb9e9b37bd8c8229d`
- `D:\Program Files\UserCache\gb\codex_tmp\metric_feature_scale_failed_closeout766\source\logs\training_feature_scale_protocols_20261002\RGBNT100.json` — `b2c14adf947b06b27b9576a712f96192c1393ad6ed3ec1be5259b63f88300169`
- `D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\review\source905\check_source_only.py` — `80fcecf0b2989c0a902a8955334dcd75c9fb48c8ef293e9b40a5a39c167482b3`
- `D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\review\source905\SOURCE_ONLY_CHECKS_R1.json` — `db4afa8dd1fa6229e2337a10393cf384ee7f25be004e68c2a90ffbfb7107c163`
- `D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\review\source905\PROTOCOL_TEXT_CHECKS_R1.json` — `505c685de0d2f129d1b6482e4edd2e7434f40faa00d12094f84b08b8ab08ab71`
