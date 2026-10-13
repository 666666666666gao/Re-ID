**Global relation input sensitivity §922 — runtime artifact audit**

**Verdict: WARN.** The collected pilot is provisionally supported as one bounded finite-sensitivity engineering observation. The result files and consistent zero-exit receipts exist; canonical record order and all 90 raw-array summary comparisons pass. There are **0 confirmed blocking evidence defects for this bounded observation**. The stored attribution is **not ready to be treated as a validated evidence-selection signal**. No retrieval, training or performance PASS is issued.

Same-family review; acceptance status **provisional**. Requested reviewer: `gpt-6-astra / max`; actual model and effort: **UNATTESTED**. Generated at 2026-10-13T02:41:45.885884+00:00. This continues the prior source reviewer and inherits its disclosed context caveat; it is not a clean-room or cross-family review.

The listed artifacts were read directly. This follow-up used local standard-library source/JSON/hash/ZIP/pickle-disassembly/struct checks only. No SSH, GPU query, torch/project import, model forward, checkpoint inference, training or new probe was run by this reviewer. Only the two runtime reports are written; project files and prior source audits are unchanged.

| Check | Verdict | Scope |
|---|---|---|
| A — Dataset identity/camera provenance and canonical ordering | PASS | The 16 saved input records exactly match an independent standard-library reconstruction from the pinned RGBNT201 source-training protocol. |
| B — Normalization, target and raw-output meaning | WARN | The saved quantities have a defined local derivative meaning, but the last-block radial derivative is strongly suppressed and is not a validated evidence-selection score. |
| C — Result existence, source identity, receipts and claim values | WARN | The actual collected result and terminal receipts exist and agree byte-for-byte with their recorded size/SHA claims; absolute cross-host chronology remains unresolved. |
| D — Actual call linkage, gradient ancestry, state effects and hooks | PASS | The bounded result is consistent with the reviewed backbone-only/VJP route and recorded unchanged state; the full original derivatives and model-state bytes are not reconstructible from the saved scalars. |
| E — One-best/one-batch scope, execution admission and cost | WARN | The evidence supports one recorded RGBNT201/uniform/E8 B16/K2 FP32 eval pilot. GPU peaks cover the postconstruction measurement window and timing is not a forward/VJP benchmark. |
| F — Dataset labels versus model-sensitivity proxy and claim ceiling | PASS | Identity/camera metadata are real dataset labels; support filtering, cosine margins and patch sensitivities are model-derived diagnostic proxies. The artifact is not attribution ground truth or a retrieval/training result. |

**A. Dataset identity/camera provenance and canonical ordering — PASS**

The 16 saved input records exactly match an independent standard-library reconstruction from the pinned RGBNT201 source-training protocol.

The protocol has 3,951 train records and 171 identity-to-label entries. All labels agree with the registered identity map; every training record uses RGB, NI, TI paths with a common basename. Fifty-one labels have at least two cameras.

The selection reproduces the numerically first eight eligible labels, the first two sorted cameras, and the lexicographically first path tuple per camera. Labels are [40,40,52,52,63,63,64,64,66,66,67,67,68,68,70,70]; dataset identities are [60,159,191,192,194,195,196,198], each twice. Cameras are [0,1,2,3,0,1,0,1,0,1,0,1,0,1,0,1].

The result's 48 image-pin keys exactly equal the selected paths; each reported image SHA has valid SHA256 syntax. Actual remote image bytes and their visual annotation truth were not locally available or independently rehashed by this reviewer.

Independent label/camera enumeration yields 16 directed cross-camera positive pairs and 224 legal directed positive/negative triplets, matching the JSON. The reported 224 supported triplets and all 16 involved samples additionally depend on model scores; they are not established solely by metadata.

The source's evaluation loader preserves order with shuffle=False; the runner checks labels and camera order. Canonical modality order is RGB, NI, TI in the exact sealed state.py, not the differently line-ended working copy.

Evidence:

- [C:/Users/gb/.trifusion_github_publish_22c3bee/logs/training_feature_scale_protocols_20261002/RGBNT201.json:2–12](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/training_feature_scale_protocols_20261002/RGBNT201.json:2)
- [C:/Users/gb/.trifusion_github_publish_22c3bee/logs/training_feature_scale_protocols_20261002/RGBNT201.json:12848–12859](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/training_feature_scale_protocols_20261002/RGBNT201.json:12848)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/readonly_diagnosis918/INPUT_SEAL.json:254](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/readonly_diagnosis918/INPUT_SEAL.json:254)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/readonly_diagnosis918/INPUT_SEAL.json:565–590](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/readonly_diagnosis918/INPUT_SEAL.json:565)
- [C:/Users/gb/.trifusion_github_publish_22c3bee/tools/official_three_dataset_data.py:8–18](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/official_three_dataset_data.py:8)
- [C:/Users/gb/.trifusion_github_publish_22c3bee/tools/official_three_dataset_data.py:28–39](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/official_three_dataset_data.py:28)
- [C:/Users/gb/.trifusion_github_publish_22c3bee/tools/train_signal_preserving_v18.py:33–49](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/train_signal_preserving_v18.py:33)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/sealed_source_snapshot911/modeling/trifusion/state.py:13–14](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/sealed_source_snapshot911/modeling/trifusion/state.py:13)
- [C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py:25–39](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py:25)
- [C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py:81–92](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py:81)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/DIAGNOSIS.json:9–228](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/DIAGNOSIS.json:9)

**B. Normalization, target and raw-output meaning — WARN**

The saved quantities have a defined local derivative meaning, but the last-block radial derivative is strongly suppressed and is not a validated evidence-selection score.

The source normalizes each concatenated 1536-D global vector with F.normalize in FP32, then computes pairwise cosine scores. It does not separately normalize each modality or scale attribution by its maximum.

The target is the arithmetic mean of positive cosine margins over legal triplets with margin>0. The support decision is detached; the margin remains differentiable. The recorded target is 0.39639848470687866 and support is 224/224. Global vectors, score matrices and individual margins are not saved, so this audit cannot independently recompute the target or model-dependent support.

For one captured patch vector x, A=x dot grad(D,x). Under x -> (1-epsilon)x with other captured coordinates fixed, D(x)-D((1-epsilon)x)=epsilon*A+O(epsilon^2), locally within the support regime. A is signed and may be negative. Norm values are Euclidean norms of the 768-channel gradient, not confidence, probabilities or attribution ground truth.

All nine groups contain nonzero gradient norms. The last-block mean absolute A is only 1.3753e-9 to 1.8869e-9; its ratio to the corresponding block-7 value is 3.9141e-5 to 5.0882e-5. Its gradient norms remain 0.1746 to 0.2143 of block-7 means.

Source-derived explanation: at the final block, a patch affects final CLS through pre-LayerNorm attention. The patch residual, later MLP, output adapters, final LayerNorm/projection are tokenwise; final CLS selection has no later patch aggregation. Ideal epsilon=0 LayerNorm is invariant to positive scalar rescaling of a token, suppressing this radial derivative while allowing nonzero derivatives in other directions.

For finite LayerNorm epsilon, LN(a*x)=gamma*a*(x-mean(x))/sqrt(a^2*variance(x)+epsilon)+beta; its radial derivative at a=1 is gamma*(x-mean(x))*epsilon/(variance(x)+epsilon)^(3/2). Floating-point reduction cancellation may also affect x dot gradient. The saved data do not contain full x/gradient vectors or activation variances, so the relative contributions of epsilon and rounding cannot be determined. The tiny values must not be relabeled exact zeros or proven pure roundoff.

One exact saved A=0 occurs at depth 11, NI, batch index 1, patch index 21 (zero-based), with gradient norm 7.003858627285808e-05. This is a concrete example of zero saved radial-sensitivity scalar coexisting with a nonzero vector-gradient norm. No new perturbation experiment was run.

Evidence:

- [C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/global_relation_inputs_diagnostic.py:35–57](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/global_relation_inputs_diagnostic.py:35)
- [C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/global_relation_inputs_diagnostic.py:60–75](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/global_relation_inputs_diagnostic.py:60)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/DIAGNOSIS.json:224–247](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/DIAGNOSIS.json:224)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/DIAGNOSIS.json:282–328](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/DIAGNOSIS.json:282)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/DIAGNOSIS.json:353–399](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/DIAGNOSIS.json:353)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/sealed_source_snapshot911/comparators/Signal-cd1b0a6/modeling/clip/model.py:154–179](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/sealed_source_snapshot911/comparators/Signal-cd1b0a6/modeling/clip/model.py:154)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/sealed_source_snapshot911/comparators/Signal-cd1b0a6/modeling/clip/model.py:223–231](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/sealed_source_snapshot911/comparators/Signal-cd1b0a6/modeling/clip/model.py:223)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/sealed_source_snapshot911/comparators/Signal-cd1b0a6/modeling/clip/model.py:478–488](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/sealed_source_snapshot911/comparators/Signal-cd1b0a6/modeling/clip/model.py:478)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/sealed_source_snapshot911/comparators/Signal-cd1b0a6/modeling/meta_arch.py:107–112](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/sealed_source_snapshot911/comparators/Signal-cd1b0a6/modeling/meta_arch.py:107)
- [C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/correspondence_roles.py:33–37](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/correspondence_roles.py:33)
- [C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/partitioned_evidence_clip.py:40–61](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/partitioned_evidence_clip.py:40)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/SENSITIVITY.pt](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/SENSITIVITY.pt), ZIP member `SENSITIVITY/data/0`
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/SENSITIVITY.pt](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/SENSITIVITY.pt), ZIP member `SENSITIVITY/data/1`

**C. Result existence, source identity, receipts and claim values — WARN**

The actual collected result and terminal receipts exist and agree byte-for-byte with their recorded size/SHA claims; absolute cross-host chronology remains unresolved.

The two new modules, plan, seal and launcher match their prior source-audit pins. The new seal matches the synchronization receipt and binds the local prior seal exactly. The private SOURCE_AUDIT.json matches the admission SHA and the published-JSON SHA recorded by synchronization. All 45 admission source hashes equal the prior source-audit inventory.

The five collected result/administrative files match every byte-count and SHA claim in COLLECTION_RECEIPT.json. SENSITIVITY.pt also matches DIAGNOSIS.json: 148976 bytes, SHA256 cd30afbc30b112ada58b3554fb28cc10dc1ffe8a82294fba89a58324929efe65. ZIP CRC checks pass. Reported extrema and nonzero counts match the raw storage exactly; all mean and absolute-mean checks match within explicitly recorded FP32 reduction tolerances.

One recorded child PID 3097314 and supervisor PID 3097311 appear consistently in START, EXIT and the launch transcript. Commands match. Remote child, local launcher/transport and collection native exit records all report 0. The diagnostic's final log line and launch COMPLETE record agree with DIAGNOSIS.json on their shared fields.

Remote START is 2026-10-13T10:27:33.045227+08:00 and remote EXIT is 10:28:34.088348+08:00 (61.043121 seconds). The local launch-exit timestamp is 10:28:24.023631+08:00, 10.064717 seconds earlier than the remote child EXIT despite the reviewed synchronous wait. Cross-host clock offset/synchronization evidence is absent. Clock skew is a possible explanation, not an established repair; exact cross-host ordering is unverified.

The plan and new seal retain their pre-run SOURCE_DRAFT_NOT_EXECUTED status and the original source audit retains its pending-runtime conclusions. Their hashes are unchanged. These are frozen historical inputs, not current claims that no diagnostic now exists. This runtime report adds evidence without rewriting them.

Remote SHA admission, old-tree preservation and child_absent are recorded assertions/receipts, not independently repeated remote observations. No remote SSH, process query or source rehash was performed during this audit.

Evidence:

- [C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/global_relation_input_sensitivity_20261013_922/INPUT_SEAL.json:4–20](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/global_relation_input_sensitivity_20261013_922/INPUT_SEAL.json:4)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/GLOBAL_INPUTS_SYNC_922.json:2–23](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/GLOBAL_INPUTS_SYNC_922.json:2)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/GLOBAL_INPUTS_SOURCE_ADMISSION_922.json:3–58](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/GLOBAL_INPUTS_SOURCE_ADMISSION_922.json:3)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/review_global_inputs922/SOURCE_AUDIT.json:15–29](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/review_global_inputs922/SOURCE_AUDIT.json:15)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/review_global_inputs922/SOURCE_AUDIT.json:675–948](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/review_global_inputs922/SOURCE_AUDIT.json:675)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/COLLECTION_RECEIPT.json:3–35](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/COLLECTION_RECEIPT.json:3)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/DIAGNOSIS.json:426–429](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/DIAGNOSIS.json:426)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/START.json:2–28](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/START.json:2)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/EXIT.json:2–16](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/EXIT.json:2)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/GLOBAL_INPUTS_LAUNCH_922.stdout:1–3](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/GLOBAL_INPUTS_LAUNCH_922.stdout:1)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/GLOBAL_INPUTS_LAUNCH_EXIT_922.json:2–3](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/GLOBAL_INPUTS_LAUNCH_EXIT_922.json:2)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/STDOUT.log:16](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/STDOUT.log:16)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/launch_global_inputs922.py:44–66](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/launch_global_inputs922.py:44)
- [C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/global_relation_input_sensitivity_20261013_922/EXPERIMENT_PLAN.md:3](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/global_relation_input_sensitivity_20261013_922/EXPERIMENT_PLAN.md:3)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/SENSITIVITY.pt](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/SENSITIVITY.pt), ZIP member `SENSITIVITY/data.pkl`, pickle byte offsets 0–299

**D. Actual call linkage, gradient ancestry, state effects and hooks — PASS**

The bounded result is consistent with the reviewed backbone-only/VJP route and recorded unchanged state; the full original derivatives and model-state bytes are not reconstructible from the saved scalars.

The runner invokes configure, the original constructor/binding checks, strict checkpoint load, epoch-8 check and model.eval, then calls model.evidence_model.backbone once. Foundation load checks schema/dataset/condition/protocol and uses strict=True. There is no optimizer, training entry, role/head forward or official query/gallery scorer in this diagnostic route.

The partitioned backbone performs three modality encoder calls internally. The real pre-block tensors at indices 3,7,11 are ancestors of the final global CLS vectors. Its post-forward stages stack is constructed after the modality forwards and is not an ancestor of the global target.

One autograd.grad call requests the nine actual inputs plus the poststack control. The code asserts that the control gradient is None and each real gradient is non-None and finite. The recorded poststack_gradient_absent=true and all saved positive gradient norms are consistent with that contract. This is stronger than the earlier source-only expectation, but the original graph and full gradient vectors were not archived for independent replay.

Tap capture does not detach or replace inputs. The support decision alone is detached before differentiation; activation/gradient values are detached only when computing saved scalars. Requesting input derivatives with autograd.grad does not itself accumulate parameter .grad buffers.

DIAGNOSIS records equal before/after state hashes (993ae4c14324db537999293d7a6f4ff31dbc32bee03e5b189969398ce2415dcb), unchanged-state=true and parameter-gradients-absent=true. The hash helper covers named state_dict tensor bytes, including registered buffers; parameter .grad absence is checked separately. These assertions support no state-value update during the measured probe after construction/load, not a claim that model construction/load performs no assignments.

The source uses eval mode, the visual path uses LayerNorm, and no BN head forward occurs. Temporary diagnostic pre-hooks are removed by __exit__, and existing output-adapter hooks by finally. Actual hook counts and runtime mode flags were not serialized.

The command has -B and no -O. Full inherited environment/sys.flags was not archived; this review treats guard execution as source-linked, receipt-supported evidence rather than an independent attestation of every assertion. No runtime optimizer trace or full pre/post model tensor dump is available.

Evidence:

- [C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py:51–110](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py:51)
- [C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_foundation_recipe.py:103–107](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_foundation_recipe.py:103)
- [C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_foundation_recipe.py:180–186](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_foundation_recipe.py:180)
- [C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_region_use_probe.py:34–45](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_region_use_probe.py:34)
- [C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_region_use_probe.py:80–90](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_region_use_probe.py:80)
- [C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_region_use_best.py:18–29](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_region_use_best.py:18)
- [C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/global_relation_inputs_diagnostic.py:16–32](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/global_relation_inputs_diagnostic.py:16)
- [C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/global_relation_inputs_diagnostic.py:43–73](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/global_relation_inputs_diagnostic.py:43)
- [C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/partitioned_evidence_clip.py:36–61](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/partitioned_evidence_clip.py:36)
- [C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_signal_preserving_v5.py:523–528](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_signal_preserving_v5.py:523)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/DIAGNOSIS.json:248–257](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/DIAGNOSIS.json:248)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/DIAGNOSIS.json:400–425](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/DIAGNOSIS.json:400)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/START.json:5–14](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/START.json:5)

**E. One-best/one-batch scope, execution admission and cost — WARN**

The evidence supports one recorded RGBNT201/uniform/E8 B16/K2 FP32 eval pilot. GPU peaks cover the postconstruction measurement window and timing is not a forward/VJP benchmark.

Exactly one recorded fixed checkpoint and one canonical source-training batch are used. No best search, alternative batch, fresh seed, new checkpoint, mask or retrieval evaluation appears in this route. The inherited training binding is B64/K8; the diagnostic B16/K2 does not establish training-batch support or memory requirements.

The reviewed launcher checks new-source/seal pins, absent admin/output directories, physical GPU0/1 admission and at least 500 MiB free disk; it contains one Popen and one wait, with no retry loop. START records GPU used memory 18 MiB on each selected GPU and 2935975936 free bytes. CUDA_VISIBLE_DEVICES=0,1 is recorded; OMP/MKL thread settings are established by launcher source, not separately echoed in START. No GPU model, current torch build, precision backend settings or full process environment was collected in these runtime artifacts.

DIAGNOSIS reports nine float32 input tensors and [cuda:1]*3+[cuda:0]*6 placement. Saved arrays are independently confirmed CPU FP32. The source disables autocast for the target and makes no autocast call around the backbone. FP32 tensor dtype does not independently certify every kernel's internal arithmetic or exclude TF32; no such broader numerical claim is accepted.

Recorded max allocated/reserved bytes are GPU0 3351473152/3657433088 and GPU1 2772115456/3053453312. The runner resets peak counters after model construction, checkpoint load, batch creation and the before-state hash. Peaks therefore exclude earlier construction/load highs and include resident tensors/cache remaining at reset; they are not incremental attribution-only allocations or whole-process/device high-water marks.

Recorded elapsed_seconds is 58.14403727487661 from run() entry through checks and tensor serialization, before final JSON output. It includes construction/load, input hashing and other overhead, but excludes earlier module import and final process teardown. Same-host child wall duration is 61.043121 seconds. Neither is a separately synchronized forward/VJP timing measurement.

One backbone call means three sequential modality encoder passes, not one image encoder pass or nine separate model forwards. All nine derivatives are requested in one VJP. The plan's 54.4 MiB input-element arithmetic is a logical tensor-size count, not measured total VRAM.

The artifacts show one recorded child; they do not prove no unrelated job or uncollected attempt ever existed. No claim of scalability, representative batch performance, general GPU feasibility or whole-training memory sufficiency is supported.

Evidence:

- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/readonly_diagnosis918/INPUT_SEAL.json:565–590](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/readonly_diagnosis918/INPUT_SEAL.json:565)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/readonly_diagnosis918/INPUT_SEAL.json:623–643](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/readonly_diagnosis918/INPUT_SEAL.json:623)
- [C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/global_relation_input_sensitivity_20261013_922/EXPERIMENT_PLAN.md:11–41](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/global_relation_input_sensitivity_20261013_922/EXPERIMENT_PLAN.md:11)
- [C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/global_relation_input_sensitivity_20261013_922/INPUT_SEAL.json:10–20](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/global_relation_input_sensitivity_20261013_922/INPUT_SEAL.json:10)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/launch_global_inputs922.py:18–55](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/launch_global_inputs922.py:18)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/START.json:5–28](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/START.json:5)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/EXIT.json:2–16](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/EXIT.json:2)
- [C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py:50](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py:50)
- [C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py:63–98](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py:63)
- [C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py:115–142](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py:115)
- [C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/partitioned_evidence_clip.py:51–61](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/partitioned_evidence_clip.py:51)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/DIAGNOSIS.json:221–223](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/DIAGNOSIS.json:221)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/DIAGNOSIS.json:248–257](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/DIAGNOSIS.json:248)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/DIAGNOSIS.json:400–441](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/DIAGNOSIS.json:400)

**F. Dataset labels versus model-sensitivity proxy and claim ceiling — PASS**

Identity/camera metadata are real dataset labels; support filtering, cosine margins and patch sensitivities are model-derived diagnostic proxies. The artifact is not attribution ground truth or a retrieval/training result.

Classification is source-label-conditioned model-sensitivity proxy (synthetic_proxy at the diagnostic/evaluation level), with real_gt identity/camera metadata as relationship inputs. This does not mean the input images or dataset identities are synthetic.

The result conditionally supports finite, nonzero gradient norms at the nine actual block/modality inputs of this model on this batch and the recorded non-ancestor poststack control. It does not establish useful masks, full-deletion effects, causal image parts, semantic complementarity, confidence calibration, role-specific evidence, training support, or any retrieval improvement.

The batch target aggregates anchor, positive and negative participation. Its selected triplets all have positive margins; it is not an isolated query attribution or a diagnosis of incorrect relationships. Directed triplets built from 16 samples are correlated, not 224 independent experimental units.

Selection uses only the first eight of 51 eligible labels from 171 labels. No random/cohort robustness, rank stability, perturbation outcome, held-out validation, threshold validation, alternate-best comparison or training run was supplied or performed.

Raw per-layer/modality scales differ and the last-block radial derivative is structurally constrained. Pooling, min-max normalizing or thresholding these values into an evidence-selection rule would be an unvalidated new method. Norm positivity establishes a differentiable path, not a useful radial ranking.

The project's full three-dataset Signal objective remains unmet in the plan and collection/sync metadata. Historic mAP values in the prior seal are not new results from this diagnostic.

Evidence:

- [C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/global_relation_inputs_diagnostic.py:1–4](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/global_relation_inputs_diagnostic.py:1)
- [C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/global_relation_inputs_diagnostic.py:35–75](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/global_relation_inputs_diagnostic.py:35)
- [C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py:25–39](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py:25)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/DIAGNOSIS.json:224–257](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/DIAGNOSIS.json:224)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/DIAGNOSIS.json:442](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/DIAGNOSIS.json:442)
- [C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/global_relation_input_sensitivity_20261013_922/EXPERIMENT_PLAN.md:17–29](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/global_relation_input_sensitivity_20261013_922/EXPERIMENT_PLAN.md:17)
- [C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/global_relation_input_sensitivity_20261013_922/EXPERIMENT_PLAN.md:43](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/global_relation_input_sensitivity_20261013_922/EXPERIMENT_PLAN.md:43)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/readonly_diagnosis918/INPUT_SEAL.json:623–629](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/readonly_diagnosis918/INPUT_SEAL.json:623)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/COLLECTION_RECEIPT.json:35](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/COLLECTION_RECEIPT.json:35)
- [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/GLOBAL_INPUTS_SYNC_922.json:31](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/GLOBAL_INPUTS_SYNC_922.json:31)

**Raw tensor verification**

The binary artifact is 148976 bytes, SHA256 `cd30afbc30b112ada58b3554fb28cc10dc1ffe8a82294fba89a58324929efe65`. All ZIP CRCs passed. Both tensors are contiguous, little-endian CPU FloatStorage (FP32), shape `[3,3,16,128]`, stride `[6144,2048,128,1]`, offset 0, `requires_grad=false`. Symbolic pickle metadata was inspected without unpickling or invoking any pickle GLOBAL. Tensor order is depth `[3,7,11]`, modality `[RGB,NI,TI]`, batch, patch.

Each storage contains 18,432 floats (73,728 bytes). The signed storage has 18,432 finite values: 8,679 positive, 9,752 negative, one zero. The norm storage has 18,432 finite and strictly positive values. This verifies saved scalar arrays, not the original full 768-channel vectors or autograd graph.

The table uses independent float64 `math.fsum` averages of stored FP32 scalars. Extrema and nonzero counts match the JSON exactly (54 comparisons). Means and absolute means pass the explicit absolute tolerance `8 * 2^-24 * recomputed_abs_mean` (36 comparisons). This allows FP32 reduction rounding; it is not attribution normalization or a claim of bit-identical float64 sums. Full reported/recomputed values, differences and tolerances are in RUNTIME_AUDIT.json.

| Block index | Modality | Mean absolute signed scalar | Mean gradient norm | Signed nonzero / 2048 |
|---|---|---:|---:|---:|
| 3 | RGB | 7.91340699e-5 | 3.76041429e-4 | 2048 |
| 3 | NI | 8.09791661e-5 | 3.86807243e-4 | 2048 |
| 3 | TI | 9.51135309e-5 | 3.84010385e-4 | 2048 |
| 7 | RGB | 3.35584833e-5 | 2.36358680e-4 | 2048 |
| 7 | NI | 3.51370435e-5 | 2.52060138e-4 | 2048 |
| 7 | TI | 4.14496114e-5 | 2.62839738e-4 | 2048 |
| 11 | RGB | 1.70750728e-9 | 4.69881772e-5 | 2048 |
| 11 | NI | 1.37530643e-9 | 4.40042263e-5 | 2047 |
| 11 | TI | 1.88685615e-9 | 5.63137042e-5 | 2048 |

| Modality | Block 11 / block 7 mean abs signed scalar | Block 11 / block 7 mean gradient norm |
|---|---:|---:|
| RGB | 5.08815393e-5 | 0.19880030 |
| NI | 3.91412109e-5 | 0.17457828 |
| TI | 4.55216848e-5 | 0.21425110 |

These ratios describe this saved batch; they are not calibrated cross-depth importances. The sole exact stored zero is NI at block 11, batch 1, patch 21 (all zero-based); its saved gradient norm is 7.003858627285808e-05. The source-derived LayerNorm explanation in check B leaves the finite-epsilon/roundoff decomposition unresolved.

Binary evidence: [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/SENSITIVITY.pt](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/SENSITIVITY.pt), ZIP member `SENSITIVITY/data.pkl`, pickle byte offsets 0–299; storages `SENSITIVITY/data/0` and `SENSITIVITY/data/1`. Text evidence: [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/DIAGNOSIS.json:258–399](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/DIAGNOSIS.json:258).

**Recorded command and cost**

```text
/data/gaob/Re-ID/conda-envs/tri_reid/bin/python -B tools/diagnose_global_relation_inputs.py --seal /data/gaob/Re-ID/Trifusion/refine-logs/global_relation_input_sensitivity_20261013_922/INPUT_SEAL.json --output-dir /data/gaob/Re-ID/Trifusion/logs/global_relation_input_sensitivity_20261013_922
```

START records CUDA_VISIBLE_DEVICES=0,1, child 3097314, supervisor 3097311, 18 MiB in use on each selected GPU and 2935975936 free disk bytes. Launcher source sets OMP_NUM_THREADS=4 and MKL_NUM_THREADS=4; they are not separately echoed in START.

| GPU | Max allocated bytes | Max reserved bytes | Allocated / reserved GiB |
|---|---:|---:|---:|
| 0 | 3351473152 | 3657433088 | 3.121303 / 3.406250 |
| 1 | 2772115456 | 3053453312 | 2.581734 / 2.843750 |

Peak counters were reset after construction/load and batch creation; resident allocations remain included, earlier highs are excluded. Diagnostic elapsed time is 58.14403727487661 s; same-remote-host START-to-EXIT is 61.043121 s. Local launch exit precedes remote EXIT by 10.064717 s. No clock correction or inferred retry has been applied. Evidence: [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/START.json:2–28](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/START.json:2); [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/EXIT.json:2–16](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/EXIT.json:2); [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/GLOBAL_INPUTS_LAUNCH_EXIT_922.json:2–3](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/GLOBAL_INPUTS_LAUNCH_EXIT_922.json:2); [C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py:93–94](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py:93); [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/DIAGNOSIS.json:431–441](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/DIAGNOSIS.json:431).

**Unresolved evidence limits**

No confirmed blocking defect prevents accepting the declared bounded observation. These limits block stronger claims:

- **L1 — WARN:** Final-block radial-scaling sensitivity is strongly suppressed by the pre-LayerNorm/CLS topology; the tiny nonzero values lack a stored-vector error budget or stability validation. Blocks: validated evidence-selection signal.
- **L2 — WARN:** Global vectors, individual margins, full activations/gradients and before/after model tensors were not archived. Only their scalar summaries, hashes and recorded assertions are available. Blocks: independent reconstruction of target, full VJP and state preservation.
- **L3 — WARN:** Local transport exit precedes recorded remote child exit by 10.064717 seconds. Clock synchronization/offset evidence is absent; no timestamp has been repaired. Blocks: exact absolute cross-host event ordering.
- **L4 — WARN:** One B16/K2 eval batch and postconstruction PyTorch peaks do not measure whole-process construction memory, B64/K8 training memory, kernel-only cost or representative dataset support. Blocks: general cost or training support.
- **L5 — WARN:** Remote image/checkpoint bytes, process disappearance, all 465 remote source bytes, full environment/assertion flags, torch build and internal precision were not independently observed in this local audit. Blocks: independent remote-state and environment attestation.

**Supported claims**

- Collected diagnostic/tensor/START/EXIT/STDOUT files match receipt SHA256 and byte sizes; the saved tensor shapes, finite counts and 90 summary comparisons pass. Direct local byte, metadata and scalar verification.
- The saved canonical input records exactly match the pinned source-training protocol; legal pair/triplet counts are 16 and 224. Independent metadata computation without model execution.
- The raw signed tensor has 18431 nonzero values out of 18432, and the raw norm tensor has 18432 strictly positive finite values. Direct struct decoding of saved FP32 storage; no inference about cause or usefulness.

**Qualified claims**

- One fixed RGBNT201/uniform/E8 B16/K2 backbone-only/VJP diagnostic completed with recorded exit code 0 and unchanged model-state digest/absent parameter gradients. Source-linked receipts and guarded source support it; no model replay, full-state reconstruction, runtime call trace or environment/sys.flags attestation was performed.
- Target 0.39639848470687866 and model-supported count 224/224 were reported. Cross-record fields agree, but saved global vectors/margins are absent, preventing independent numerical recomputation.
- Elapsed diagnostic time 58.14403727487661 s and the two recorded postconstruction GPU allocation/reservation peaks describe this run's instrumentation. No whole-process peak, incremental-only memory, isolated VJP benchmark or training/generalization cost claim.
- Final-block radial sensitivity is strongly suppressed consistently with the final pre-LN attention/CLS path. Source-derived structural explanation, not an empirical separation of finite-epsilon and roundoff effects; exact numerical stability is unverified.

**Unsupported claims**

- Retrieval benefit, mAP/Rank-1 improvement, superiority over Signal or completion of the three-dataset goal.
- Validated evidence masks, reliable radial patch ranking, confidence/probability, semantic complementarity or causal body-part attribution.
- Full-deletion/intervention effects from the local derivative; actual benefit of any masking/reweighting rule.
- Query-only contribution, role-specific C/T/M evidence or support of an identity-learning loss/training teacher.
- Representative source population, B64/K8 training support, multi-batch/seed/best robustness or scalability.
- Exact cross-host chronological ordering, hardware/library/precision attestation, whole-process GPU peak or isolated VJP cost.
- Independent recomputation of the original model target, gradients, state digests or remote input image/checkpoint SHA from this saved artifact.

**Source identity and historical status**

The two modules, plan, launcher and seal match their source-review pins. The seal binds the prior seal exactly. Synchronization/admission bind the same source-audit JSON, and all 45 admission source pins equal the prior audit inventory. Current hashes were independently recomputed for the scientific files listed below; this was not a new remote source-tree sweep. The retained line-ending caveat for working state.py remains in admission; exact sealed snapshot bytes were used for modality order.

The plan, new seal and SOURCE_AUDIT remain frozen pre-execution records. Their pending/unexecuted statements are preserved as historical context. The collected runtime artifacts establish present result existence without editing that history. Published/private source-audit Markdown byte equality is not asserted; the exact source-audit JSON binding was verified.

**All scientific input/source files actually used**

Hashes cover complete file bytes. Text references use 1-based lines. The binary artifact uses ZIP member/byte-offset evidence instead of fabricated text lines.

| File | Bytes | Lines | SHA256 |
|---|---:|---:|---|
| [C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/global_relation_inputs_diagnostic.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/global_relation_inputs_diagnostic.py) | 3275 | 75 | `5d962766520a616942902aeac5a7e19f83a3dadc39797174d010c58b2cd98ffb` |
| [C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_global_relation_inputs.py) | 8379 | 149 | `97cb00ff2fe99a3b13ea14ac344067c8a854ce63eba3a0f5c6ecf19f3e03e9ce` |
| [C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/global_relation_input_sensitivity_20261013_922/EXPERIMENT_PLAN.md](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/global_relation_input_sensitivity_20261013_922/EXPERIMENT_PLAN.md) | 5104 | 43 | `120f9ac71f92f2dff073cef50dafbbe0fe6b4e4bfb6b53d34bd9585d93ab09d7` |
| [C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/global_relation_input_sensitivity_20261013_922/INPUT_SEAL.json](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/global_relation_input_sensitivity_20261013_922/INPUT_SEAL.json) | 980 | 22 | `44d92bae4fb7c8a7b7a50af22724d9620ee56a993d163f6b058e17df44b48632` |
| [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/DIAGNOSIS.json](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/DIAGNOSIS.json) | 18103 | 443 | `e8d2ad0b5fbfa59cd07278d4052dce7755af40adce68fdb146f28539daa5730b` |
| [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/SENSITIVITY.pt](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/SENSITIVITY.pt) | 148976 | binary | `cd30afbc30b112ada58b3554fb28cc10dc1ffe8a82294fba89a58324929efe65` |
| [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/START.json](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/START.json) | 837 | 29 | `e77c6e24849e5575693e47daef5897f3846a1b92e391a2e5891f3eb5f5ab8fa3` |
| [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/EXIT.json](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/EXIT.json) | 612 | 17 | `109428df0d123bcca9682bb9252c77f0c0dedb2ea2b0ead0a68db855706d9e7f` |
| [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/STDOUT.log](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/STDOUT.log) | 1584 | 16 | `83b50a3f4235b7b01a44f8ceefc3566492dafd59d289c30b96779f3b5a5ef48f` |
| [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/COLLECTION_RECEIPT.json](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/global_inputs_results922/COLLECTION_RECEIPT.json) | 1500 | 36 | `e679c3ff179774f083d9f5d40f5c1dec557cccc660bf91b094eb012e3ee3d979` |
| [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/GLOBAL_INPUTS_LAUNCH_EXIT_922.json](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/GLOBAL_INPUTS_LAUNCH_EXIT_922.json) | 69 | 4 | `5ba02b6ffa66e5e4a00702121c6d548fed822613bfdde302d1bc5280daf3c503` |
| [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/GLOBAL_INPUTS_LAUNCH_922.stdout](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/GLOBAL_INPUTS_LAUNCH_922.stdout) | 1678 | 3 | `1a2f07192545cc5c72d4a8f9eb0bf3bd0eeb9676fdb6cd6a30db510c500569d6` |
| [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/GLOBAL_INPUTS_SYNC_922.json](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/GLOBAL_INPUTS_SYNC_922.json) | 2657 | 37 | `cd178d85aaefe5c27f407e75cd883bad08a97961fbb9da6ed6a19c871b4cbb7c` |
| [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/GLOBAL_INPUTS_SOURCE_ADMISSION_922.json](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/GLOBAL_INPUTS_SOURCE_ADMISSION_922.json) | 9085 | 68 | `b619842e88ec6ebc02839ab8ca3137cb888ceec8c62cbbc5135bda8bc5bd5bcf` |
| [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/readonly_diagnosis918/INPUT_SEAL.json](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/readonly_diagnosis918/INPUT_SEAL.json) | 116016 | 1151 | `8e5c37e57c088a8ab53344266195bbc7183ccd27a94d0f0910ff9c20998b4b41` |
| [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/review_global_inputs922/SOURCE_AUDIT.json](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/review_global_inputs922/SOURCE_AUDIT.json) | 56988 | 976 | `8ce2e6de4b2d1f8e9921645e1aed6edf7170bcdc4e9aadbb607d1d68a486c55b` |
| [C:/Users/gb/.trifusion_github_publish_22c3bee/logs/training_feature_scale_protocols_20261002/RGBNT201.json](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/training_feature_scale_protocols_20261002/RGBNT201.json) | 1958131 | 78317 | `b2409c2d992d04adcf0c78354d306afd6af7ae9507738fd4e83de480e9a9b90f` |
| [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/launch_global_inputs922.py](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/launch_global_inputs922.py) | 4028 | 69 | `fdb624c0c46cf1f7aef6cb3f17ed1ad60ae2405e36da7de5502dfcd8030bc485` |
| [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/sealed_source_snapshot911/comparators/Signal-cd1b0a6/modeling/clip/model.py](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/sealed_source_snapshot911/comparators/Signal-cd1b0a6/modeling/clip/model.py) | 33586 | 729 | `2c2ca81815830fad2ab13bb8bb4ba0d41f0625477af6505815f0982d96df4e73` |
| [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/sealed_source_snapshot911/comparators/Signal-cd1b0a6/modeling/meta_arch.py](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/sealed_source_snapshot911/comparators/Signal-cd1b0a6/modeling/meta_arch.py) | 5518 | 125 | `8b709d5ab4b14ba150a96bbb5e8cc7e6dbbb71dd8bd30cafd7a5cb110ee5a757` |
| [C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/sealed_source_snapshot911/modeling/trifusion/state.py](C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/sealed_source_snapshot911/modeling/trifusion/state.py) | 4854 | 136 | `5fe89c4885507e74ca1833cc18822a4b57e4c2d6fa9a6c6b2e127d1e74b70db9` |
| [C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/partitioned_evidence_clip.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/partitioned_evidence_clip.py) | 3614 | 76 | `3bad53abd6777602e1f1ea9e7b03f10a796116b577501ee5ec0aec6e683b436c` |
| [C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/correspondence_roles.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/correspondence_roles.py) | 15524 | 328 | `e2a51154b5b327341f35e453dd32a821b091a49a669b49a7c70e79e6e2f2591b` |
| [C:/Users/gb/.trifusion_github_publish_22c3bee/tools/official_three_dataset_data.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/official_three_dataset_data.py) | 1885 | 39 | `c3180a88d12440709319d2025928fc699ff3b7c020b11697d468453fbfc68062` |
| [C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_region_use_best.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_region_use_best.py) | 9471 | 152 | `bc73f6f2d8dc399048dca58ba25d99c059f6cb80bae35e5ab01b130c5b438ff4` |
| [C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_foundation_recipe.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_foundation_recipe.py) | 18242 | 341 | `c18042509dc33ff66eee1a6b04a723ad55685bd5d9daf218c3779703ffe20758` |
| [C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_region_use_probe.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_region_use_probe.py) | 5920 | 126 | `176e1e88a74ede5447f57cf0e3a89c1a0847e729647d37b11fdb6c76ea715591` |
| [C:/Users/gb/.trifusion_github_publish_22c3bee/tools/train_signal_preserving_v18.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/train_signal_preserving_v18.py) | 17540 | 310 | `f6bdda4631d710d0b6db5fe2a8df124d8dcbd294cb54adcf3ec261d523e96cf8` |
| [C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_clean_clip_joint.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_clean_clip_joint.py) | 9804 | 183 | `178e9793f2cf8261eb44a278e053d5caabd4f2f37dba48a65952b2547338a746` |
| [C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_signal_preserving_v5.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_signal_preserving_v5.py) | 65190 | 1682 | `e162184f68778b4991db0f97f26c5fda273b2ad2f7c8db2bbb2d53775eb717e5` |

**Local review tooling failures preserved**

- first standard-library archive inspection: exit 1. E:/Scripts/python.exe failed immediately with 'No pyvenv.cfg file'. Switched to the existing base interpreter E:/python.exe; no package installation or environment modification.
- receipt/protocol checker: exit 1. Reviewer script used key positive_pairs instead of actual legal_positive_pairs and raised KeyError. Corrected only the in-memory reviewer script after reading the concrete JSON schema; diagnostic files were unchanged.
- AST metadata output handling: tool-side JSON parse failure. Tool output budget truncated a verbose successful local metadata result before functions JSON.parse. Repeated the read-only inspection with concise output; no source or result repair.
- in-memory report assembly: tool-side TypeError before any file write. Reviewer addressed a nonexistent detail-array index while refining wording. Corrected the in-memory report assembly; no evidence/source file was modified.

A PowerShell range-display helper initially omitted single-range excerpts; those excerpts were reread directly using standard-library file reads.

Chinese plan text initially rendered incorrectly through terminal output; it was reread as UTF-8, ASCII-escaped in JSON and decoded for direct inspection. No file encoding was modified.

These are reviewer-tooling/display events, not pilot execution failures. No diagnostic, source or historical failure was repaired or replaced.
