# E924 current-source audit

**WARN — `source_ready_for_prepare_m0: true`; `unresolved_blocking_source_defects: []`.** This is readiness to collect new prepare/M0 evidence, not completed M0, formal-training, performance or novelty approval. Actual reviewer model/effort **UNATTESTED**; same-family/provisional. All source references below are relative to `C:/Users/gb/.trifusion_github_publish_22c3bee`.

## A–F

| Check | Finding |
|---|---|
| A Provenance — PASS | Labels/cameras come from protocol training records and the native batch (`tools/official_three_dataset_data.py:8`; `tools/run_signal_preserving_v5.py:568`; new wrapper `tools/run_global_conditioned_evidence.py:24`). Same-ID/different-camera positives and different-ID negatives define legal relations; positive-margin support is model-derived (`modeling/trifusion/global_relation_inputs_diagnostic.py:35`). Inference consumes no identity label (`modeling/trifusion/global_conditioned_evidence.py:89`; `tools/train_signal_preserving_v18.py:44`). Actual inputs/support remain unverified. |
| B Normalization — PASS | L2/attention softmax, support-count averaging and division by the same VJP scale256 are model/proxy operations, not self-normalized retrieval metrics (`modeling/trifusion/global_relation_inputs_diagnostic.py:42`; `modeling/trifusion/global_conditioned_evidence.py:28`, `:63`). Camera AP uses real-label ranks and query averaging (`tools/train_rgbnt100_signal_oof.py:253`). No new measured score exists. |
| C Existence/status — PASS | The inspected plan says unexecuted; all four new milestones are TODO (`refine-logs/global_conditioned_evidence_20261013_924/EXPERIMENT_PLAN.md:3`, `:44`; `EXPERIMENT_TRACKER.md:7`). Versioned plans are not runtime evidence. Historical scores and upstream screening are outside this new-result audit. |
| D Actual call path — PASS | New constructor, wrapper, batch adapter and loss callbacks are reachable through configure/build; current-state/optimizer compatibility has no confirmed source blocker. Detailed ownership tracing follows. |
| E Scope/cost — WARN | Eight source-training M0 updates per arm, no M0 retrieval evaluation (`tools/run_foundation_recipe.py:211`). Both arms have one exposure VJP plus an M0-only isolation `autograd.grad`; formal training has one exposure VJP. Actual B64/K8 coverage, two-card AMP feasibility, timing/memory/disk and realized mask activity remain pending. |
| F Classification — PASS | Labels/future retrieval: `real_gt` lineage. Sensitivity/support: **`synthetic_proxy` with real-label conditioning**. Prepare/M0: engineering verification. Neither gradients nor spatial exclusion prove semantic disjointness, causal importance or efficacy (`modeling/trifusion/global_relation_inputs_diagnostic.py:35`; plan `:12`, `:22`; `refine-logs/evidence_learning_gap_20261013_920/PRIMARY_SOURCE_SCREENING.md:3`). |

## Constructor, ownership and current M0 diagnostics

Static alias tracing resolves `inner` to `tools.run_independent_native_evidence`. The final `clean.control.GlobalTokenTriFusion` and semantic type guard both refer to `GlobalConditionedEvidence`; the head wrapper is `GlobalTaskRoleHeads`. The captured build chain is new entry → global-task → detached-input → native-research → partitioned → independent-native → clean constructor. Evidence: `tools/run_global_conditioned_evidence.py:16`, `:70`; `tools/run_global_task_role.py:39`; `tools/run_role_input_detach.py:34`; `tools/run_native_research.py:22`; `tools/run_native_partitioned.py:54`; `tools/run_independent_native_evidence.py:36`, `:177`; `tools/run_clean_clip_joint.py:79`. Captured original function references prevent recursion.

The role replacement inherits the same constructor, restores CPU RNG, and strictly loads inherited state; mask/exposure/generator/capture flags add no registered parameter (`modeling/trifusion/global_conditioned_evidence.py:41`). Optimizer construction checks duplicate-free, exact trainable coverage (`modeling/trifusion/evidence_author_heads.py:63`). Partitioning preserves parameter identities/state (`modeling/trifusion/partitioned_evidence_clip.py:17`; `tools/run_native_partitioned.py:25`). Original-semantic and paired **numerical** equality remain runtime facts.

`CURRENT_MODEL` is assigned to the actual built wrapper at `tools/run_global_conditioned_evidence.py:34`. Foundation training calls this bound build before forward/loss (`tools/run_foundation_recipe.py:193`, `:231`). Its later overwrite during fresh reload occurs after the training loop and isolation calls (`:268`), so it does not misbind the declared single-model M0.

The ordinary total loss resolves to global-task global+fused author objectives. The M0 diagnostic deliberately calls **`previous.previous.loss_values` = `run_role_input_detach.loss_values`**, whose captured original path computes the fused author task from the already-produced `output['heads']` (`tools/run_global_conditioned_evidence.py:54`; `tools/run_global_task_role.py:25`; `tools/run_role_input_detach.py:22`; `tools/run_independent_native_evidence.py:118`). It repeats loss arithmetic, not model/head forward. It requests gradients to `shared_global` and the current wrapper's trainable `evidence_model.backbone.*` parameters, requires all to be `None`, and records the checked count (`tools/run_global_conditioned_evidence.py:60`). `retain_graph=True` preserves the subsequent ordinary backward.

This matches the detach boundary (`modeling/trifusion/role_input_detach.py:5`) and detached fused-global/current-head path (`modeling/trifusion/global_task_role_heads.py:8`, `:20`). A separate M0-only assertion verifies all evidence-model parameter `.grad` are absent immediately after the exposure VJP (`modeling/trifusion/global_conditioned_evidence.py:85`), following optimizer zeroing (`tools/run_foundation_recipe.py:229`). These checks are now reachable source, **not executed passes**. Both arms incur them; they are absent from formal train mode.

## Mask, zero support, AMP and reload

The target VJP uses the three modality inputs to zero-based `resblocks[3]`; support decisions and norm-based masks are detached (`modeling/trifusion/global_relation_inputs_diagnostic.py:11`, `:30`, `:49`; `modeling/trifusion/global_conditioned_evidence.py:62`). No supported relation yields a connected zero target. Zero total sensitivity means no exclusion; otherwise exactly32/128 positions per active sample/modality are excluded in both arms. Random ranking uses its own seeded CUDA generator (`tools/run_global_conditioned_evidence.py:37`; model `:68`). This matches VJP and mask-count policy, not measured equal wall time or guaranteed identical realized masks across evolving runs.

Dynamic content is removed after token-wise projection and before CNN mixing, then those positions leave softmax (`modeling/trifusion/global_conditioned_evidence.py:17`, `:25`; `modeling/trifusion/role_global_tokens.py:31`). Contextualized remaining patches, global-conditioned queries and global residual remain (`modeling/trifusion/role_global_tokens.py:30`; new model `:92`). The no-semantic-isolation boundary is accurate.

Target/attention FP32 does not make the upstream FP16-autocast training VJP FP32-equivalent (`tools/run_foundation_recipe.py:230`); scale256 cannot certify sensitivity ranking/coverage. Inference skips VJP/masking and uses full reads (`modeling/trifusion/global_conditioned_evidence.py:89`). Strict state loading and the two-source-sample reload check exist (`tools/run_foundation_recipe.py:180`, `:263`), with `atol=rtol=1e-5`, not bitwise/full-gallery parity. Generator state is transient and unused by inference; training-resume equivalence is not claimed.

## Required runtime facts, separate from source blockers

- Bind actual protocol/public CLIP/Signal source/config/module bytes, native B64/K8 and camera/label provenance; witness original-semantic and paired parameter/buffer equality and optimizer ownership. The historical local Signal snapshot moves cameras to the embedding device (`evidence/smooth_ap_source_coverage_independent_audit_20260920/snapshots/signal_meta_arch.py:102`), but does not attest the next live import.
- Preserve eight effective updates and actual passing ownership records per arm. At least one M0 step per arm must have supported relations and active masking; check `excluded_patches == 32 * active_sample_modalities`, including zero-support records. **Generic inherited `M0_PASS` does not assert this new activity requirement** (`tools/run_foundation_recipe.py:263`, `:279`; `tools/run_independent_native_evidence.py:146`). The plan itself requires it before training (`EXPERIMENT_PLAN.md:32`).
- Observe finite AMP/ordinary-update gradients, label-free unmasked strict reload, actual two-card cost/peak memory and disk headroom. M0 includes extra isolation work, so its total step time is not a direct formal-training estimate. Preserve failures. No fact in this list has been established by this source audit.

## Pins and review boundaries

Current audited SHA256:

- Model: `f87a7d26ad357cb81e37b41676379dfc3fd1e8b3a3bdcf907ab0eb3a7eba6e0d`
- Runner: `a35c5005740c26dcb624e1ef0c69756092b70fe8141b57219272028d77931587`
- Plan and `EXPERIMENT_PLAN_20261013_110423.md`: `cd9241f06b00d90b5d6eb6004e6b3123d1ccd11ad6a15a49b72a45eb27db48f6`
- Tracker: `7758d893dc3587690e9db98f5b3dcdc528ace5c82897c94917fe0c4daec55557`

Full 46-file hashes and 41 Python AST results are in `SOURCE_AUDIT.json` and `STATIC_CHECKS_FINAL.json`. Initial hashes remain in `STATIC_CHECKS.json`. Source drift was caught by the final consistency assertion (chunk `00dfbe`); no stale verdict was written. The earlier observation that explicit ownership diagnostics were absent is superseded by the current source. An earlier PATH Python failure (`No pyvenv.cfg file`, chunk `e7a73c`) is preserved; checks subsequently used the already-installed isolated Python executable. No installation occurred.

No SSH/network/GPU/torch import/model execution/training/checkpoint inference or project-file changes occurred. Mandatory workspace rules caused extra reads of `SOUL.md`, `USER.md`, `memory/2026-10-13.md`, `memory/2026-10-12.md`, project `AGENTS.md`, and the audit skill/references. Some combined output was truncated; no note was used as current runtime evidence. `MEMORY.md` was not read. Review is therefore not fully context-isolated; actual model/effort remain UNATTESTED. Future official-mAP checkpoint selection is outside prepare/M0 approval and does not establish selection-independent performance.
