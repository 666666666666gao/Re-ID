# Global relation input sensitivity 922 — source audit

Observed at: 2026-10-13T00:14:16.2708552Z

**Overall: WARN. Source-ready for the exact read-only pilot; no confirmed blocking source defects (0). No runtime or performance PASS.**

The conclusion applies to the retained prior-sealed execution tree plus the two reviewed new modules, the fixed RGBNT201/uniform/E8 checkpoint, one B16/K2 batch, and the reviewed one-launch wrapper. At the observation time the synchronization receipt was absent; publication, real remote admission, and execution were still pending.

Requested reviewer: `gpt-6-astra` / `max`. Actual model and effort: **UNATTESTED**. `review_independence: same-family`; `acceptance_status: provisional`. Agent: `/root/audit_global_relation_inputs922`.

## Method and limits

Read the supplied primary artifacts directly and followed the same-project dependencies needed to resolve constructor, data, graph and launcher behavior. Used only source reads/search, JSON parsing, SHA256 hashing, normalized text comparison and local existence checks. No Python process, project/author import, model construction, GPU query, SSH call, probe, forward, VJP, optimizer update or project source modification occurred. Reports are written only in the requested review directory.

Context disclosure: inherited mandatory session instructions caused reads of SOUL.md, USER.md and current/prior daily notes. Those notes are not scientific evidence for this report. MEMORY.md and unrelated project source were not read; this report does not claim perfectly clean-room context.

PASS below means a source/static-data check passed within its stated scope. Runtime assertions are proposed checks, not observed successful results. No existing experiment history or failure was rewritten.

## Reviewed byte pins

| Artifact | SHA256 |
|---|---|
| `tools/diagnose_global_relation_inputs.py` | `97cb00ff2fe99a3b13ea14ac344067c8a854ce63eba3a0f5c6ecf19f3e03e9ce` |
| `modeling/trifusion/global_relation_inputs_diagnostic.py` | `5d962766520a616942902aeac5a7e19f83a3dadc39797174d010c58b2cd98ffb` |
| `refine-logs/global_relation_input_sensitivity_20261013_922/EXPERIMENT_PLAN.md` | `120f9ac71f92f2dff073cef50dafbbe0fe6b4e4bfb6b53d34bd9585d93ab09d7` |
| `refine-logs/global_relation_input_sensitivity_20261013_922/INPUT_SEAL.json` | `44d92bae4fb7c8a7b7a50af22724d9620ee56a993d163f6b058e17df44b48632` |
| `launch_global_inputs922.py` | `fdb624c0c46cf1f7aef6cb3f17ed1ad60ae2405e36da7de5502dfcd8030bc485` |
| `readonly_diagnosis918/INPUT_SEAL.json` | `8e5c37e57c088a8ab53344266195bbc7183ccd27a94d0f0910ff9c20998b4b41` |

The JSON report contains all 45 inspected/hashed primary and supplementary artifact paths. The prior seal contains 465 source pins and 90 artifact pins; this review did not hash all remote entries or open remote checkpoint binaries.

## Findings

| Check | Status |
|---|---|
| A. Dataset labels, legal relationships and ordering | PASS — source_and_static_json_only |
| B. Normalization and meaning of saved raw values | PASS — source_semantics_only |
| C. Result existence and unexecuted status | WARN — source_and_local_existence_only |
| D. Called graph, gradient ancestry, detach, BN and parameter gradients | PASS — source_semantics_only |
| E. One-best/one-batch scope and cost claims | WARN — source_and_arithmetic_only |
| F. Sensitivity-proxy classification and claim ceiling | PASS — source_semantics_only |
| G. Constructor, schema, checkpoint and seal compatibility | PASS — source_and_local_hash_only |
| H. Temporary hook lifecycle | PASS — source_semantics_only |
| I. Launcher admission, command, logs, exits and one-launch semantics | PASS — source_semantics_only |

### A. Dataset labels, legal relationships and ordering: PASS

Labels and cameras originate in the pinned RGBNT201 source-training protocol, and record selection is availability/path based rather than score based.

- records_for uses row.label for train and row.camera, with paths resolved under dataset_root. select_records takes the numerically first eight labels having at least two cameras, then the first two sorted cameras and lexicographically smallest path tuple for each. It never consumes a model score.
- Static parsing of the pinned protocol found 3,951 train records, 171 label-map entries, zero label-map inconsistencies, zero three-modality/basename inconsistencies, and 51 labels with at least two cameras. The selected labels are 40,52,63,64,66,67,68,70, corresponding to dataset identities 60,159,191,192,194,195,196,198. These are static expected inputs, not executed model inputs.
- The eval loader rebuilds ImageDataset(records, transform) with shuffle=False. Its author collator preserves input order, returns labels in raw[1] and cameras in raw[3], and assigns image channels in RGB, NI, TI order. The probe checks loaded label and camera order and hashes the 48 selected source image paths before and after the proposed forward.
- Same-identity/different-camera positives and different-identity negatives are label-derived. For this selected B16/K2 batch there should be 16 directed positive pairs and 224 legal directed triplets before model-dependent support filtering. Actual image readability, image bytes and resulting support remain unverified.

Evidence: [tools/official_three_dataset_data.py:8–18](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/official_three_dataset_data.py:8); [tools/diagnose_global_relation_inputs.py:25–39](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py:25); [tools/diagnose_global_relation_inputs.py:81–92](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py:81); [tools/diagnose_global_relation_inputs.py:111](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py:111); [tools/train_signal_preserving_v18.py:33–49](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/train_signal_preserving_v18.py:33); [tools/build_v12_complete_path_oof_targets.py:362–381](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/build_v12_complete_path_oof_targets.py:362); [sealed_source_snapshot911/comparators/Signal-cd1b0a6/data/datasets/bases.py:98–107](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/sealed_source_snapshot911/comparators/Signal-cd1b0a6/data/datasets/bases.py:98); [sealed_source_snapshot911/comparators/Signal-cd1b0a6/data/datasets/make_dataloader.py:164–182](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/sealed_source_snapshot911/comparators/Signal-cd1b0a6/data/datasets/make_dataloader.py:164); [modeling/trifusion/global_relation_inputs_diagnostic.py:43–57](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/global_relation_inputs_diagnostic.py:43); [logs/training_feature_scale_protocols_20261002/RGBNT201.json:2–12](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/training_feature_scale_protocols_20261002/RGBNT201.json:2); [logs/training_feature_scale_protocols_20261002/RGBNT201.json:12848–12859](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/training_feature_scale_protocols_20261002/RGBNT201.json:12848).

### B. Normalization and meaning of saved raw values: PASS

The objective and saved maps have explicit mathematical meanings; no performance score is normalized by its own maximum.

- The target is the mean of cosine margins s(anchor,positive)-s(anchor,negative) over legal relations whose current margin is strictly positive. L2 feature normalization defines cosine similarity. Dividing by the count of selected relations defines this conditional mean; it is not accuracy or a calibrated confidence.
- The support mask is derived from margin.detach()>0. Its selection does not receive gradients, whereas normalized features and margins remain differentiable. A no-support batch produces a connected zero in the helper and is rejected by the caller; it is not replaced with another batch or a zero mask.
- The two saved arrays are the signed channel sum of activation times target gradient and the per-token gradient L2 norm, with no min-max scaling, thresholding, absolute-value replacement of the signed map, or fitted coefficient.
- For a hidden token a scaled to (1-epsilon)a, the target change is approximately -epsilon*(a dot grad), with support held fixed. Positive saved values therefore indicate a local decrease under scaling down. This is not the finite effect of complete deletion.
- Because the active support set is model-dependent, target magnitudes cannot be interpreted as comparable retrieval quality across models or batches. COMPLETE also does not itself assert nonzero sensitivity: zero finite gradients are allowed and must be interpreted using the raw arrays/statistics.

Evidence: [modeling/trifusion/global_relation_inputs_diagnostic.py:35–57](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/global_relation_inputs_diagnostic.py:35); [modeling/trifusion/global_relation_inputs_diagnostic.py:60–75](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/global_relation_inputs_diagnostic.py:60); [tools/diagnose_global_relation_inputs.py:42–46](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py:42); [tools/diagnose_global_relation_inputs.py:100–108](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py:100); [tools/diagnose_global_relation_inputs.py:117–128](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py:117); [refine-logs/global_relation_input_sensitivity_20261013_922/EXPERIMENT_PLAN.md:19–29](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/global_relation_input_sensitivity_20261013_922/EXPERIMENT_PLAN.md:19).

### C. Result existence and unexecuted status: WARN

The artifacts honestly describe a draft. This review contains no pilot runtime result and does not verify remote completion.

- The plan explicitly states that no model call, attribution array or performance result exists. The new seal records SOURCE_DRAFT_NOT_EXECUTED. The older SOURCE_PATHS artifact records zero forwards and runtime_autograd_verified=false; it is not runtime evidence for this pilot.
- At the recorded local observation time, GLOBAL_INPUTS_SYNC_922.json, GLOBAL_INPUTS_LAUNCH_COMMAND_922.json, GLOBAL_INPUTS_LAUNCH_EXIT_922.json and the local proposed DIAGNOSIS.json/SENSITIVITY.pt paths were absent. The new INPUT_SEAL.json now exists and was directly read and hashed. Remote paths were not queried.
- The executable emits status COMPLETE only after its gradient/state/input guards and tensor save; genuine process success still requires the supervisor's child exit_code==0. Source literals such as state_unchanged=True and vjp_calls=1 are intended report fields, not observed results.
- The prior seal's E8 checkpoint and six historical endpoint rows are identity/provenance inputs. Their historical retrieval numbers are not new measurements, and this source audit does not revalidate their remote binaries.

Evidence: [refine-logs/global_relation_input_sensitivity_20261013_922/EXPERIMENT_PLAN.md:3–5](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/global_relation_input_sensitivity_20261013_922/EXPERIMENT_PLAN.md:3); [refine-logs/global_relation_input_sensitivity_20261013_922/INPUT_SEAL.json:2–21](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/global_relation_input_sensitivity_20261013_922/INPUT_SEAL.json:2); [refine-logs/evidence_information_paths_20261013_921/SOURCE_PATHS.json:269–275](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/evidence_information_paths_20261013_921/SOURCE_PATHS.json:269); [tools/diagnose_global_relation_inputs.py:105–142](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py:105); [launch_global_inputs922.py:44–55](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/launch_global_inputs922.py:44); [readonly_diagnosis918/INPUT_SEAL.json:623–650](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/readonly_diagnosis918/INPUT_SEAL.json:623).

### D. Called graph, gradient ancestry, detach, BN and parameter gradients: PASS

The selected pre-block tensors are ancestors of the raw global CLS features; the later stacked stages tensor is a separate descendant branch.

- The probe calls producer.configure(), foundation.build/load(), model.eval(), then only model.evidence_model.backbone. It does not invoke the role/head forward, training loop, loss functions, official scorer or optimizer.
- The partitioned backbone encodes RGB, NI and TI sequentially. It installs existing output-adapter hooks at blocks 3,7,11, passes output+mean(adapter deltas) onward, and creates the stages stack only after all modality forwards. The global return is the concatenation of the three encoder CLS features, not a computation from the stages stack.
- The diagnostic pre-hooks capture the actual complete [129,B,768] inputs at blocks 3,7,11 without replacing or detaching them. The author's block performs attention before its tokenwise MLP; final layer norm/projection and CLS selection follow. Thus pre-block patch inputs can affect the final CLS, including those at block 11. This does not establish a measured nonzero gradient.
- One torch.autograd.grad requests all nine captured tensors plus the later stages tensor. allow_unused=True permits the stages control to return None; each of the nine real input gradients must separately be non-None and finite. Detach occurs only in the support-selection decision and after the gradients are obtained for serialization.
- DetachedSemanticTriFusion.role_evidence and detached author-head paths are not called by this backbone-only route. The model's eval override sends signal to eval as well; the visual path uses LayerNorm and no author BN head is called. The before/after state_dict digest covers parameters and registered buffers, and all parameter .grad fields are explicitly required to remain None.
- autograd.grad returning input derivatives is not optimizer backpropagation into parameter .grad buffers. The source correctly checks the intended side effects but only the proposed runtime can establish that these guards actually pass.

Evidence: [tools/diagnose_global_relation_inputs.py:73–110](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py:73); [modeling/trifusion/global_relation_inputs_diagnostic.py:16–32](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/global_relation_inputs_diagnostic.py:16); [modeling/trifusion/partitioned_evidence_clip.py:36–61](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/partitioned_evidence_clip.py:36); [sealed_source_snapshot911/comparators/Signal-cd1b0a6/modeling/clip/model.py:223–231](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/sealed_source_snapshot911/comparators/Signal-cd1b0a6/modeling/clip/model.py:223); [sealed_source_snapshot911/comparators/Signal-cd1b0a6/modeling/clip/model.py:447–488](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/sealed_source_snapshot911/comparators/Signal-cd1b0a6/modeling/clip/model.py:447); [sealed_source_snapshot911/comparators/Signal-cd1b0a6/modeling/meta_arch.py:96–112](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/sealed_source_snapshot911/comparators/Signal-cd1b0a6/modeling/meta_arch.py:96); [modeling/trifusion/role_input_detach.py:5–7](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/role_input_detach.py:5); [modeling/trifusion/global_task_role_heads.py:16–32](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/global_task_role_heads.py:16); [modeling/trifusion/evidence_author_heads.py:42–47](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/evidence_author_heads.py:42); [tools/run_signal_preserving_v5.py:523–528](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_signal_preserving_v5.py:523).

### E. One-best/one-batch scope and cost claims: WARN

The design is correctly bounded, but timing and resource feasibility are unmeasured and the reported GPU peak excludes construction peaks.

- The runner fixes RGBNT201/uniform, seed 42, the original E8 best, and one canonical B16/K2 source-training batch under the existing eval transform. There is no loop over additional checkpoints, datasets, batches or seeds. Availability-selected cross-camera examples do not estimate coverage in actual B64/K8 augmented training batches.
- One backbone call means three modality encoder passes through twelve blocks each, with the existing shared adapters and stage snapshots. It is not one image encoder pass; it is also not nine separate forwards or nine VJP calls.
- The stated nine-input size is arithmetically correct: 9*129*16*768=14,266,368 float elements, or 57,065,472 bytes (54.421875 MiB); nine returned gradients have the same logical size. This excludes the retained full forward graph, adapter deltas, three-role stage snapshots/stacks, model parameters, CUDA workspaces and allocator reserves.
- The saved two [3,3,16,128] FP32 arrays contain 36,864 floats (147,456 raw payload bytes), consistent with the plan's small-output expectation. No actual serialization size was measured.
- Peak counters reset at lines 93-94 after build, load and batch preparation. The reported peak_memory is the peak from that reset onward, including allocations live at reset and prior allocator reservations; it is not the process-wide construction/loading peak. elapsed_seconds begins before seal checks and includes setup and repeated hashing.
- The 1-3 minute estimate, CUDA memory admission and hardware feasibility remain unverified. Rehashing the prior seal's 465 source and 90 artifact entries before and after is broader I/O than the single neural batch, as the plan acknowledges.

Evidence: [tools/diagnose_global_relation_inputs.py:50–105](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py:50); [tools/diagnose_global_relation_inputs.py:112–137](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py:112); [modeling/trifusion/partitioned_evidence_clip.py:40–61](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/partitioned_evidence_clip.py:40); [refine-logs/global_relation_input_sensitivity_20261013_922/EXPERIMENT_PLAN.md:11–17](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/global_relation_input_sensitivity_20261013_922/EXPERIMENT_PLAN.md:11); [refine-logs/global_relation_input_sensitivity_20261013_922/EXPERIMENT_PLAN.md:23–33](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/global_relation_input_sensitivity_20261013_922/EXPERIMENT_PLAN.md:23); [tools/diagnose_region_use_best.py:18–30](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_region_use_best.py:18).

### F. Sensitivity-proxy classification and claim ceiling: PASS

Classify this as a source-label-conditioned model-sensitivity proxy, not ground-truth attribution or retrieval evaluation.

- Identity and camera relationships use real dataset metadata. The sensitivity target/support and patch maps are model-derived quantities, and there is no external ground truth for which hidden tokens are useful or causal. For the audit taxonomy, the attribution evidence is synthetic_proxy with real_gt relationship labels, not a self-supervised or performance evaluation.
- The batch target includes each sample's anchor, positive and negative roles. The gradient is not query-only attribution. Intermediate patch-token coordinates can already contain mixed context and are not physical-part annotations.
- A finite result supports only the availability of this particular local derivative on one selected model/batch. Nonzero map entries, if actually observed, would establish local sensitivity there. Neither observation establishes full-deletion effects, semantic complementarity, mask benefit, causal image-part attribution, training support or improved retrieval.
- The plan and output boundary explicitly keep these limits and do not authorize a new teacher, mask, model, training experiment or performance claim.

Evidence: [modeling/trifusion/global_relation_inputs_diagnostic.py:1–4](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/global_relation_inputs_diagnostic.py:1); [modeling/trifusion/global_relation_inputs_diagnostic.py:35–40](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/global_relation_inputs_diagnostic.py:35); [modeling/trifusion/global_relation_inputs_diagnostic.py:60–64](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/global_relation_inputs_diagnostic.py:60); [tools/diagnose_global_relation_inputs.py:138](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py:138); [refine-logs/global_relation_input_sensitivity_20261013_922/EXPERIMENT_PLAN.md:19–25](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/global_relation_input_sensitivity_20261013_922/EXPERIMENT_PLAN.md:19); [refine-logs/global_relation_input_sensitivity_20261013_922/EXPERIMENT_PLAN.md:43](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/global_relation_input_sensitivity_20261013_922/EXPERIMENT_PLAN.md:43).

### G. Constructor, schema, checkpoint and seal compatibility: PASS

The reused configure chain reaches the correct RegionUseHeads/DetachedSemanticTriFusion constructor and original schema, then performs strict binding and state checks.

- Importing diagnose_region_use_best leads through diagnose_same_evidence_comparison and diagnose_global_task_role_best; that wrapper aliases its entry.runner to the same run_correspondence_roles module used by foundation. The probe's direct import of trifusion is reachable after the reused import chain adds ROOT/modeling to sys.path.
- run_region_use_probe.configure overrides the head class to RegionUseHeads, while run_role_input_detach.configure selects DetachedSemanticTriFusion. The research and partition wrappers propagate the current schema/build into run_independent_native_evidence.configure and foundation. The final condition includes the original region-use condition, and the constructor binding is compared both with the initialization witness and the sealed selected row.
- foundation.load requires checkpoint schema, dataset, condition and protocol hash equality and load_state_dict(strict=True); the probe further requires E8. Prior require_inputs hashes the selected checkpoint along with the historical artifact set before and after. There is no new best selection.
- The new seal exactly binds both reviewed new modules and the provided prior seal. It declares the same dataset/comparison/epoch/batch/devices as the source. The primary artifact hash table in this report distinguishes the reviewed bytes from future changes.
- The relevant local state.py copy differs from the old seal by CRLF/LF bytes only; normalized text equality was checked. The sealed snapshot state.py has the expected old SHA and the same MODALITY_ORDER=(RGB,NI,TI). Source readiness refers to the retained pinned execution tree plus the two new modules, not a wholesale upload of the local checkout.
- No dynamic constructor, checkpoint load or dependency import was executed in this audit. Their actual acceptance remains a runtime obligation.

Evidence: [tools/diagnose_region_use_best.py:13–15](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_region_use_best.py:13); [tools/diagnose_same_evidence_comparison.py:18–20](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_same_evidence_comparison.py:18); [tools/diagnose_global_task_role_best.py:7–14](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_task_role_best.py:7); [tools/run_visual_update_control.py:20–23](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_visual_update_control.py:20); [tools/run_region_use_probe.py:34–45](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_region_use_probe.py:34); [tools/run_region_use_probe.py:80–90](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_region_use_probe.py:80); [tools/run_role_input_detach.py:34–41](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_role_input_detach.py:34); [tools/run_native_partitioned.py:23–30](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_native_partitioned.py:23); [tools/run_native_partitioned.py:54–60](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_native_partitioned.py:54); [tools/run_independent_native_evidence.py:36–92](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_independent_native_evidence.py:36); [tools/run_independent_native_evidence.py:177–184](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_independent_native_evidence.py:177); [tools/run_foundation_recipe.py:103–107](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_foundation_recipe.py:103); [tools/run_foundation_recipe.py:180–186](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_foundation_recipe.py:180); [tools/diagnose_global_relation_inputs.py:51–77](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py:51); [refine-logs/global_relation_input_sensitivity_20261013_922/INPUT_SEAL.json:4–20](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/global_relation_input_sensitivity_20261013_922/INPUT_SEAL.json:4); [sealed_source_snapshot911/modeling/trifusion/state.py:13–14](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/sealed_source_snapshot911/modeling/trifusion/state.py:13); [readonly_diagnosis918/INPUT_SEAL.json:546–547](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/readonly_diagnosis918/INPUT_SEAL.json:546).

### H. Temporary hook lifecycle: PASS

For the single intended use, both diagnostic pre-hooks and the backbone's temporary adapter-output hooks are removed on success and on forward-body exceptions.

- GlobalInputTap captures without returning a replacement tensor and removes its handles in __exit__. The context wraps the backbone forward and ends before the VJP. Removing a module hook does not detach the already captured computation graph.
- The partitioned backbone removes its temporary forward hooks in a finally block. The two device-transfer pre-hooks are part of the pre-existing partitioned constructor and correctly remain in place. The registered RegionUseHeads hook does not run because roles are never forwarded.
- No reusable-context, concurrent-forward or hook-registration-failure guarantee is claimed. No extra defensive branch is needed for unsupported usage absent evidence that it is in scope.

Evidence: [modeling/trifusion/global_relation_inputs_diagnostic.py:9–32](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/global_relation_inputs_diagnostic.py:9); [tools/diagnose_global_relation_inputs.py:95–105](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py:95); [modeling/trifusion/partitioned_evidence_clip.py:28–33](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/partitioned_evidence_clip.py:28); [modeling/trifusion/partitioned_evidence_clip.py:38–57](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/partitioned_evidence_clip.py:38); [modeling/trifusion/region_use_heads.py:7–17](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/region_use_heads.py:7).

### I. Launcher admission, command, logs, exits and one-launch semantics: PASS

The launcher implements a single child launch with explicit GPU0/1 admission, pinned remote inputs and genuine child/transport exit recording.

- The local launcher requires a GLOBAL_INPUTS922_STATIC_EXACT synchronization receipt, embeds its source/seal pins and then checks the remote seal/source hashes. That receipt is a launch dependency, not supplied runtime evidence; it was absent at the recorded observation time.
- The remote code checks that both admin and output directories are absent, reads memory.used only for physical GPUs 0/1, requires both exact indices and <500 MiB usage, and requires >=500 MiB free disk. It does not query power, temperature, GPU2/3 or port 2025.
- admin.mkdir() reserves the single run before Popen. The sole child command uses the existing tri_reid Python, -B, the reviewed diagnostic, the new seal and the unique output directory, with CUDA_VISIBLE_DEVICES=0,1 and four OMP/MKL threads. There is no retry or candidate loop.
- The child stdout and stderr share STDOUT.log. START records the child PID, command and admission facts; child.wait() supplies the actual exit code written to remote EXIT.json, and only an exit-zero child result is read. The local SSH return code and transcripts are written separately after subprocess.run returns.
- A local timeout or missing local receipt must not be interpreted as a child failure or justification to launch again: capture_output buffers the SSH transcript until the remote process terminates. The plan correctly calls for joining the original handle and preserving failures.
- This review did not execute the launcher, establish the SSH connection, perform admission, create a remote directory or verify the synchronization receipt. A start/spawn/transport failure is not a successful diagnostic.

Evidence: [launch_global_inputs922.py:8–12](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/launch_global_inputs922.py:8); [launch_global_inputs922.py:18–37](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/launch_global_inputs922.py:18); [launch_global_inputs922.py:38–55](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/launch_global_inputs922.py:38); [launch_global_inputs922.py:57–69](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/launch_global_inputs922.py:57); [refine-logs/global_relation_input_sensitivity_20261013_922/EXPERIMENT_PLAN.md:33–41](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/global_relation_input_sensitivity_20261013_922/EXPERIMENT_PLAN.md:33).

## Specific static expectations

The pinned protocol has 51 labels with at least two source cameras. The canonical eight labels are `40, 52, 63, 64, 66, 67, 68, 70`, mapping to original identities `60, 159, 191, 192, 194, 195, 196, 198`. All 3,951 training records agree with the label map and the three modality basenames. The selected batch implies 16 directed cross-camera positive pairs and 224 legal directed triplets. These are static JSON-derived expectations; the model-dependent supported count is unknown.

Expected map order is depth `[3,7,11]`, modality `[RGB,NI,TI]`, selected batch order, then 128 non-CLS tokens. The expected shape is `[3,3,16,128]`. The raw two-array payload is 147,456 bytes. A finite zero map is still finite and is not proof of nonzero sensitivity.

## Suspicions ruled out

- Missing ENVIRONMENTS population is not a defect for this fixed RGBNT201 pilot: the RGBNT201 branch calls image_batch directly. The vehicle path's requirement does not apply. Evidence: [tools/run_correspondence_roles.py:60–63](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_correspondence_roles.py:60); [tools/run_region_use_probe.py:23–31](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_region_use_probe.py:23).
- The template eval loader's doubled record list does not double this batch: only its transform is reused, and the returned loader is reconstructed from the undoubled selected records. Evidence: [tools/train_signal_preserving_v18.py:33–41](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/train_signal_preserving_v18.py:33); [tools/build_v12_complete_path_oof_targets.py:362–381](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/build_v12_complete_path_oof_targets.py:362).

## Open requirements and claim boundary

**Blocking source defects: none identified. No source repair is requested by this audit.**

Before launch, the executor must finish byte-exact publication and provide the concrete synchronization receipt for these reviewed bytes. The reviewed launcher's existing remote source/seal, original-input, GPU and disk assertions remain required. Do not replace pinned old sources with a broad local checkout upload: local state.py differs from its seal in line-ending bytes, while its sealed snapshot and normalized text were verified.

After the one permitted launch, inspect the real child exit record and actual saved artifacts. Retain any failure; do not infer success from source readiness or relaunch because a local wait timed out. Label GPU peak_memory as measured after the post-construction reset, and treat the 1–3 minute estimate as unverified.

An actual successful pilot would establish only engineering access to the specified sensitivity on this particular source-label batch. Nonzero sensitivity, unchanged model state, actual cost, and process success each need their actual runtime evidence. No result here supports mask efficacy, complete-deletion causality, semantic complementarity, new training, retrieval improvement or the broader three-dataset goal.
