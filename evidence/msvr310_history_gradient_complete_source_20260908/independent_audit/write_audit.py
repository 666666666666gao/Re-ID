from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json

O=Path(__file__).parent
d=json.loads((O/'independent_checks.json').read_bytes())
claim_checks=json.loads((O/'independent_claim_checks.json').read_bytes())
R='C:/Users/gb/.trifusion_github_publish_22c3bee'
S='C:/Users/gb/.codex_tmp/history_gradient_complete_source_20260908'
P='C:/Users/gb/.codex_tmp/history_gradient_complete_processing_20260908'
A='C:/Users/gb/.codex_tmp/history_gradient_complete_analysis_20260908'
F='C:/Users/gb/.codex_tmp/history_gradient_complete_figures_20260908'

checks={
 'A':dict(title='Ground-truth provenance and fixed-state binding',status='WARN',details='All 1032 record mappings, source folds, contiguous local labels including class zero, registered sample orders, inherited pixel SHA strings and nine state digest references pass. Raw images, checkpoint tensors, upstream Signal/Mamba binaries and GPU state hashing were not independently re-executed. Five inherited local files differ from their runtime SHA only by CRLF versus LF.',evidence=['R/tools/build_msvr310_train_oof_protocol.py:12-55','R/tools/train_msvr310_signal_oof.py:68-99','R/tools/probe_msvr_history_candidate_gradients.py:134-146','R/tools/train_msvr_instance_memory.py:94-110','S/source/summary.json:2-14']),
 'B':dict(title='Normalization and parameter-block alignment',status='PASS',details='Fixed 64-anchor mean, margin 0.3, identity-based masks, standard vector norms/cosines, lambda=1, role-aligned 42/54/93 encoder tensors, and explicit undefined handling. All 26190 comparison identities, 5238 gradient sums and all plotted means pass independently. Total loss components and raw gradients remain runtime-only.',evidence=['R/tools/msvr_instance_memory.py:40-80','R/tools/probe_msvr_history_candidate_gradients.py:26-80','R/tools/run_signal_preserving_v5.py:99-142','R/configs/MSVR310/TriFusion-source-style-paired-v1-r2.json:59-67','R/tools/plot_msvr_history_candidate_gradients.py:108-126']),
 'C':dict(title='Artifact existence and numbers',status='WARN',details='All 25 intake files (53366938 bytes), nine raw text streams/receipts, terminal bindings, 5238 CSV rows and 108 plotted cells pass. Nine distance binaries (174752256 bytes) are absent locally. The tracker is stale (RUNNING/NOT_STARTED); terminal receipts establish completed source/CPU processing. No finished new narrative was supplied, so no narrative acceptance is implied.',evidence=['S/intake_manifest.json:1-131','S/pipeline.json:1-128','R/refine-logs/msvr310_history_candidate_gradient_v1/EXPERIMENT_TRACKER.md:10-15','P/completion.json:1-9','R/tools/verify_msvr_history_gradient_all_statistics.py:69-101']),
 'D':dict(title='Actual calls and gradient implementation',status='WARN',details='Launcher, probe, CPU verifier, postcheck, intake, analysis and plotting have code paths and terminal receipts. Chain decomposition and replay source are correct on inspected paths. Direct full-graph proof is one historical group at step 67 per source state only; repeats use the same graph. No independent model backward, multiple-group full graph, raw upstream vector or 14-loss reconstruction is available.',evidence=['R/tools/run_msvr_history_candidate_gradients.py:24-49','R/tools/probe_msvr_history_candidate_gradients.py:84-106','R/tools/probe_msvr_history_candidate_gradients.py:193-248','R/tools/msvr_freshness_probe.py:43-71','C:/Users/gb/.codex_tmp/finish_history_gradient_source_20260908.py:65-103']),
 'E':dict(title='Registered scope, branch coverage and cost',status='PASS',details='All 9 states x260 batches, all source records, 1746 history batches and all text rows are covered. Queue expiry, latest-record dedup, current-record exclusion and recorded zero-upstream group skips are exercised. Capacity eviction is unexercised; max queue 408 <512. Costs and source-only/seed42-only/fixed-weight boundaries are retained; no task-conflict, optimizer-update, generalization or performance claim is supported.',evidence=['R/refine-logs/msvr310_history_candidate_gradient_v1/EXPERIMENT_PLAN.md:9-28','R/tools/msvr_instance_memory.py:8-37','R/tools/probe_msvr_history_candidate_gradients.py:149-248','R/docs/MSVR310_PARTIAL_VS_TOTAL_METRIC_GRADIENT_2026-09-08.md:7-45']),
 'F':dict(title='Measurement classification',status='PASS',details='Dataset-label-based fixed-state mechanism diagnostic (real_gt mechanism, not retrieval); synthetic T0 chain-rule proxy; runtime gradient/state witnesses; independently replayed text/queue/aggregate checks; remote NumPy matrix witnesses. No measured retrieval or optimizer result is created by this diagnostic.',evidence=['R/tools/probe_msvr_history_candidate_gradients.py:109-131','R/tools/verify_msvr_history_candidate_gradients.py:103-106','R/tools/analyze_msvr_history_gradient_text.py:119-128','R/tools/plot_msvr_history_candidate_gradients.py:129-138'])
}
limitations=[
 'The nine distances.f32 files, original images, six role checkpoints, source Signal/CLIP weights, frozen fields, RNG snapshots and parameter-gradient arrays are not independently available in the local audit package. Distance shape/offset/count and recorded remote hashes are verified; matrix values and GPU state digests are not regenerated.',
 'The remote all-statistics postcheck records 30420 checks over 43688064 float elements. Locally, five metadata-only statistics were independently derived at every batch (11700 checks), all 117 state/statistic sums were checked, and matrix-dependent counts/losses remain remote NumPy witnesses.',
 'Each direct-graph proof uses only the first single stored group at step 67. Multi-group accumulation is supported by the inspected chain-rule implementation and per-group coordinate-equality assertions, not a saved direct multi-group parameter-gradient comparison.',
 'Same-graph repeated VJPs estimate repeat-backward numerical noise, not independent-forward/training noise, bias, estimator uncertainty or robustness across seeds.',
 'G denotes the gradient of all 14 existing task losses restricted to one encoder role block. It excludes trainable classifier/neck blocks from the displayed vector. The 14 components are not recorded separately here, so neither the weighted total scalar nor its gradient can be reconstructed independently from this text package.',
 'The skip count verifies the consistency of recorded group choices; absence of upstream arrays prevents independently proving that each omitted group has exactly zero upstream. Capacity eviction and undefined/zero role-gradient branches are not exercised in the complete source run.',
 'The runtime environment receipt identifies the Signal commit/diff, not full Python/PyTorch/CUDA/Mamba binary identities. The five inherited CRLF/LF differences are byte-level portability limits, not source-equivalence failures.',
 'EXPERIMENT_TRACKER.md is stale at the audited input snapshot. A finished new source narrative was not supplied. The executor must update the tracker and pass the final narrative for a separate claim/number check.',
 'No new training, inference, retrieval, image access or scientific-file edit was performed by this audit. This same-family fresh-context review is provisional, not cross-family acceptance.'
]
actions=[
 dict(priority='before_public_claims',action='Update the source tracker from the verified terminal pipeline and retain the fixed-state/source-only status. Submit the completed narrative for a separate claim-number check; do not label this audit as narrative acceptance.'),
 dict(priority='claim_qualification',action='Preserve runtime-only gradient/state provenance, remote matrix-check attribution, all 194 history batches per state, role-restricted G, no optimizer/retrieval gains, and the single-group direct-proof limit.'),
 dict(priority='if_stronger_reproducibility_is_required',action='Provide the nine saved distance binaries to a separate read-only checker, and a pinned runtime/weight package before claiming independently reproduced matrices or model derivatives. This audit does not launch such work.'),
 dict(priority='if_stronger_gradient_validation_is_required',action='Separately register a multiple-history-group direct comparison and broader numerical reference before claiming general grouped-backpropagation precision. Same-graph noise must not be treated as independent-run uncertainty.'),
 dict(priority='future_training_contract_only',action='The broad nonzero candidate contribution can motivate a separately registered fixed paired gradient-scope hypothesis; it cannot itself approve training, establish the cause of prior failure, justify PCGrad, or imply generalization gains.')
]

lines=[
'# Independent audit: complete fixed-state historical-candidate gradient diagnostic',
'',
'**Overall verdict: WARN. Deterministic text, queue, binding and plot-number checks PASS.** No fabricated ground truth, score manipulation, missing text row, numerical mismatch or scientific-code change was found in the inspected package. The result is an adequately covered fixed-state source mechanism diagnostic. It is not independently reproduced model backpropagation and does not establish optimizer or retrieval benefit.',
'',
'Date: 2026-09-08. Reviewer: fresh-context native Codex auditor (`/root/audit_msvr_history_gradient_source`), model **gpt-6-astra**, reasoning effort **max**, same model family; **provisional** assurance. The reviewer wrote only independent analysis and review artifacts in this new audit directory. No experiment code/config/contract/result/tracker was edited; no training, inference, image forward or remote command was launched.',
'',
'Paths in evidence references: `R` = `'+R+'`; `S` = `'+S+'`; `P` = `'+P+'`; `A` = `'+A+'`; `F` = `'+F+'`. Each `file:line` below is relative to the indicated root. `steps.jsonl` line numbers are batch numbers (1-260), so lines 67-260 contain all history rows in each of the nine streams.',
'',
'## A-F verdicts',
'',
'| Check | Status | Finding |','|---|---|---|']
for k,v in checks.items():lines.append(f"| {k}. {v['title']} | {v['status']} | {v['details']} |")
lines += [
'',
'## A. Provenance and source identity',
'',
'The actual diagnostic reads the frozen protocol through `records_for(..., True)`, not labels predicted by a model. The protocol builder consumes the SHA-bound label inventory and decodes identity/camera/scene from the image names (`R/tools/build_msvr310_train_oof_protocol.py:12-55`). The independent checker read the complete original inventory and independently reconstructed all **1,032** record mappings, all modality paths, identity-scene membership, the three deterministic source/heldout partitions, and all contiguous source-label maps. The dataset parser uses identity `name[0:4]`, camera `name[11]` and scene `name[6:9]` (`R/data/datasets/msvr310.py:67-87`); the diagnostic uses the protocol rather than this parser’s unrelated stale docstring.',
'',
'| Fold | Source identities | Source records | Source scenes | True identity mapped to local class 0 | Class-0 current exposures per state |','|---|---:|---:|---:|---:|---:|']
for f in d['fold_scope']:
 s=next(s for s in d['states'] if s['fold']==f['fold']);lines.append(f"| {f['fold']} | {f['source_identities']} | {f['source_records']} | {f['source_scenes']} | {f['class_zero_identity']} | {s['class_zero_exposures']} |")
lines += [
'',
'All 2,340 sampled batches have 64 positions, 8 identities x8 instances, real protocol identity/scene mappings, no heldout record, and exact equality to the registered batch-index schedule. All three per-batch pixel SHA strings match the inherited complete Q1 `memory_steps.jsonl` record for the appropriate endpoint. All source records are observed in every state. Class zero is a legal local classifier label; it is not a background class. Cross-entropy has no custom zero-ignore argument, and triplet masks use identity equality (`R/modeling/trifusion/signal_preserving_v8.py:696-741`; `R/tools/train_msvr310_signal_oof.py:68-99`). Scene labels contribute descriptive statistics only, never the negative identity label.',
'',
'The three states are rebuilt role initialization, sealed control endpoint and sealed fresh-memory endpoint. Current state digests match their inherited initialization/final references for all nine states; all six checkpoint SHA references match the current configuration. The inherited Q1 summary and CPU files are locally present and hash to the fixed values in the new configuration. Actual model construction/reload checks Signal source/heldout identity binding, strict state loading, aliases and final state digests (`R/tools/train_msvr310_trifusion_oof.py:27-55`; `R/tools/train_msvr_instance_memory.py:94-110`; `R/tools/probe_msvr_history_candidate_gradients.py:134-146,262-275`). These comparisons verify references and recorded witnesses; this auditor did not load the underlying tensors or images.',
'',
'The source summary records HEAD `d682042c5da218091f31665f81c185902fb6872a`, while the launcher was bound to `eebaaa07708e8f5e5d3e05d246afe5fe5f7abfc6` (`S/source/summary.json:5`; `S/pipeline.json:4`). An independent `git diff --name-only` found six changed documentation/tracker/evidence files and **no scientific source/config/plan change**. Of **66 transitive project SHA references**, **61** match local raw bytes; the remaining five are the inherited `criterion.py`, `state.py`, `builder.py`, `experts/mamba.py`, and `experts/semantic_residual.py`. Their LF-normalized bytes match the registered runtime SHA exactly. Their local and remote hashes are explicitly distinguished in `R/evidence/msvr310_style_readiness_20260907/msvr310_style_prelaunch_source_bytes_20260907.json:55-78`. No source or input was normalized or modified for execution by this audit.',
'',
'## B. Losses, gradients and normalization',
'',
'The expanded scalar is the hinge mean over the **current 64 anchors**: normalized current/current and current/history Euclidean distances, same-identity positives excluding the same current position, different-identity negatives, margin **0.3**. The current peers remain differentiable; only historical candidates are stopped in the original objective. History is normalized when stored and replayed. The differentiable-history form uses the same max/min/hinge scalar, with history as an independent leaf for its upstream derivative (`R/tools/msvr_instance_memory.py:40-80`; `R/tools/probe_msvr_history_candidate_gradients.py:26-40,185-200`). No division by a favorable model score or own maximum occurs. Vector L2 normalization is a feature/gradient definition, not a performance rescaling. The ratio panel’s maximum controls only its color range; its printed numbers remain raw mean ratios (`R/tools/plot_msvr_history_candidate_gradients.py:75-99`).',
'',
'The chain rule is `g_U = J_U^T dL/dU`, `g_V = J_V^T dL/dV`, and their sum for this same scalar objective. The raw triplet derivatives are unweighted; the displayed task perturbation is `G + lambda g_V` with **lambda=1** from `TRIPLET_FUSED`. `G` comes from all **14 existing task losses**: fused, three branches and three residuals, each with ID and triplet losses. Existing weights are 0.25/1 for fused ID/triplet, 1/12 and 0.25 for branch ID/triplet, and 1/12 and 0.25 for residual ID/triplet (`R/tools/run_signal_preserving_v5.py:99-142`; `R/configs/MSVR310/TriFusion-source-style-paired-v1-r2.json:59-67`).',
'',
'All compared vectors within a role use exactly the same parameter ordering. The inherited trainable-name bindings and code partition give **CNN42 / Transformer54 / Mamba93 encoder tensors**, totaling **189**, in every state. The 14 trainable classifier/neck tensors are outside these displayed encoder blocks. Thus “total” means all-task gradient **restricted to that role**, not the whole-model gradient. Gradients missing from `autograd.grad` are represented by correctly aligned zero tensors; cosines are `None` when either norm is zero (`R/tools/probe_msvr_history_candidate_gradients.py:53-80,154-157`). No undefined value is filled with zero in the plot. None of the 20,952 displayed metric contributions is undefined in this run; the absent-gradient branch is not thereby validated.',
'',
'The independent checker rederived all **26,190** pairwise norm/cosine closures, all **5,238** gradient-sum norm identities and cosine identities, `||g_U+g_V-g_U||=||g_V||`, and `||G+lambda g_V-G||=lambda||g_V||`. Maximum relative sum-norm closure error is **2.02738058622413e-10**, exactly the published maximum. Maximum pairwise comparison closure is **2.3062076436885516e-15**. These are consistency checks of recorded scalar summaries; they do not prove the underlying vector direction independently.',
'',
'The author CSV contains raw current/history norms, ratios, three cosines and repeat-noise values for every role/history batch. The independent evidence additionally records quantiles/min/max/means of `||g_U||`, `||g_V||`, `||g_U+g_V||`, `||G||`, `||G+g_V||`, the two combined/current norm ratios and relative repeat-noise ratios for all 27 state/role combinations (`independent_checks.json`, `norm_distributions`). The 14 per-component loss values are not saved by this diagnostic, so its total loss scalar and total gradient remain runtime witnesses.',
'',
'## C. Complete files, text rows, CPU checks and figure values',
'',
'All **25** manifest-listed text/log files, totaling **53,366,938 bytes**, match their exact intake size and SHA; all nine `receipt.json` objects equal their copies in the final summary. The source pipeline has all five stage exit codes zero and terminal `COMPLETE_VERIFIED_SOURCE_ONLY`. Source runtime logs contain the exact nine-state order and **180** epoch records, each state reaching epoch20/batch260. Processing receipts bind the postchecker, intake, analysis and plot scripts to the actual output. The static tracker remains `source RUNNING / source CPU NOT_STARTED` (`R/refine-logs/msvr310_history_candidate_gradient_v1/EXPERIMENT_TRACKER.md:10-15`) despite these newer terminal records; it needs an executor update, not a rerun.',
'',
'| Independently checked item | Coverage | Result |','|---|---:|---|',
'| Raw steps and full queue metadata | 2,340 rows, all nine states | PASS |',
'| Current identity/scene + source schedule + inherited pixel SHA strings | All 2,340 batches | PASS |',
'| Five metadata-only statistics | 11,700 batch/statistic checks | PASS |',
'| Per-state sums of all 13 statistics | 117 sums:99 integer,18 float | PASS; max float difference2.8312206268310547e-7 |',
'| Saved matrix layout derived from metadata | 43,688,064 float32 positions | PASS for offsets/counts/receipt bytes |',
'| Role comparison finite/zero/cosine/norm identities | 26,190 comparisons | PASS |',
'| Current+history sum identities | 5,238 role/history rows | PASS |',
'| CPU role arrays | 20,952 numeric entries | Exact equality |',
'| CSV | All5,238 rows x14 columns | Exact scalar/string values |',
'| Plot means | 108 cells from20,952 contributions | Exact agreement to receipt; aggregate max difference2.220446049250313e-16 |',
'',
'The remote NumPy CPU/postcheck receipts report **43,688,064 distance elements** and **30,420 = 2,340x13 statistic checks** (`S/source/cpu_verification.json:5-6`; `P/all_statistics_postcheck.json:9-11`; checker formulas at `R/tools/verify_msvr_history_gradient_all_statistics.py:69-101`). The local package omits all nine `distances.f32` files (**174,752,256 bytes**). Consequently, the auditor independently reconstructs five statistics from text metadata (memory count, positive pairs, negative pairs, cross-scene positive pairs, maximum age), checks all 13 text sums and bounds, and verifies the matrix-layout arithmetic, but does **not** claim to rerun the distance-dependent eight statistics or verify binary matrix values. Those have valid recorded remote-check provenance only.',
'',
'The PNG has verified dimensions **2106x1494**, its hash matches the plot receipt, and the auditor visually inspected the complete four-panel figure for clipping, labeling, per-cell counts and scope. All108 numeric cells follow the independently checked point list. The PDF bytes/hash are verified but no separate PDF render inspection was performed. No new source narrative was supplied to this review: artifact-number integrity is established within the limits above, while final narrative wording remains pending a separate check.',
'',
'## D. Live call path, direct proof, replay and numerical limits',
'',
'The launcher calls T0 -> preflight -> preflightCPU -> source -> sourceCPU sequentially, returning immediately on a nonzero stage exit (`R/tools/run_msvr_history_candidate_gradients.py:24-49`). The new complete source probe explicitly invokes its mining scalar, total criterion, current/repeat/total gradients, leaf upstream, grouped candidate VJP, first direct graph, role summaries and state-hash checks. The postprocessor waits for the original pipeline terminal before running the separately SHA-bound full13-statistic NumPy script, then the text intake, analysis and plotting (`C:/Users/gb/.codex_tmp/finish_history_gradient_source_20260908.py:50-103`). These functions have actual terminal outputs; they are not phantom helper code. The local T0 receipt is a **synthetic proxy** only. It records equality of old/new scalar/current gradient and a linear-map VJP; this auditor inspected but did not execute T0.',
'',
'For historical VJPs, frozen anchor/reference/Signal fields and role-entry CPU/CUDA RNG are captured from each original B64 group. The record-to-position map keeps the final occurrence of each duplicate record, matching latest-view queue replacement (`R/tools/msvr_freshness_probe.py:48-60`; `R/tools/msvr_instance_memory.py:29-37`). Every selected group is reencoded under `fork_rng`; selected coordinates must be bitwise equal to the stored history. Coefficients are the detached history leaf upstream and have unique selected positions. Each group’s VJP is accumulated in FP32; the same graph is differentiated a second time to record repeat noise. Groups with exactly zero upstream are skipped (`R/tools/probe_msvr_history_candidate_gradients.py:43-106`). This is an implementation of the chain rule; it does not add historical anchors.',
'',
'The replay path is encoder+fusion only, so it bypasses the ID BatchNorm necks. Encoder role paths use LayerNorm; the frozen Signal remains eval; the complete current path’s saved buffers are restored after each batch. Global CPU/CUDA RNG equality is asserted after the historical work and all parameter `.grad` fields must remain `None`; initial/final full-state digests match (`R/modeling/trifusion/signal_preserving_v8.py:74-77,264-339,607-612,637-653`; `R/tools/probe_msvr_history_candidate_gradients.py:201-239`). State SHA checks are runtime witnesses. The current implementation has no stateful role BatchNorm path, so lack of a separate per-VJP buffer snapshot is not evidence of an observed buffer bug.',
'',
'Historical representation/upstream computation is FP32 outside autocast; role encoding uses AMP FP16, fixed scale256 and FP32 accumulated parameter VJPs. Comparison dot products and squared norms are summed in double precision. Activation checkpointing recomputes tail-block work inside backwards (`R/modeling/trifusion/signal_preserving_v8.py:356-368`). This matters for compute accounting and numeric reproducibility; neither a raw record-forward count nor same-graph repeat noise is a full FLOP or precision certificate.',
'',
'All nine full-graph proofs occur at **step67**, the first historical batch, and use **one historical B64 group**. Their recorded relative L2 errors, independently rederived from the norm receipts, span **9.4408731215117e-10 to6.105141938845793e-5**, all below the registered **0.005** tolerance. The highest is fold2/control (`S/source/fold_2_control/receipt.json:719-731`). Direct loss equality is asserted at runtime. This does not establish a full direct graph for later multi-group batches. Maximum same-graph current/history repeat noise norms are **3.6817184090060954e-5 /2.4339534831289294e-5**; their largest noise/norm ratios are **0.00017700467135181247 /0.0001262319004779407**. These measurements characterize repeated backward operations at a fixed graph; no independent-forward or independent-training uncertainty is measured.',
'',
'## E. Registered coverage, exercised branches and compute',
'',
'Every state contains20x13=260 source batches, including66 empty-history batches followed by194 history batches. The queue is first populated after zero-based step65, and first read with history at one-based step67. Age1-8 means stored-view batch recency, not parameter-update age: every state remains fixed. The complete source observations are **149,760 current-record exposures**, **111,744 history-anchor exposures**, **532,866 historical candidate-record exposures**, and **45,368 recorded nonzero historical-upstream record exposures**. These are repeated exposures, not independent examples. Source fold sets overlap as expected from OOF construction; nine states are not nine independent seeds.',
'',
'| Queue / VJP event | All-state count | Interpretation |','|---|---:|---|',
'| Age-expired records |58,041 | Exercised in1,674 batches |',
'| Duplicate current-record positions |51,948 | Latest within-group view mapping exercised |',
'| Current-history record exclusions |11,871 | Historical copies of current records removed |',
'| Available historical groups |13,716 | Groups in selected memory metadata |',
'| Selected VJP groups |12,498 | Recorded selections, metadata-consistent |',
'| Zero-upstream group skips |1,218 | Across901 batches; upstream arrays unavailable |',
'| Capacity evictions |0 | Capacity branch unexercised; max queue408<512 |',
'',
'Per-fold maximum selected history records are356/372/362; maximum post-update queue sizes are391/408/406; maximum selected age8 in every state. The recorded zero-upstream skip branch is exercised, but this reviewer cannot prove the skipped upstream vectors are exactly zero without the absent vectors. All5,238 role/history rows have a nonzero historical gradient above the corresponding repeated-backward noise and a negative `cos(g_U,g_V)`; therefore zero/undefined role-gradient handling was not exercised by this source run.',
'',
'Explicit extra role record forwards equal **12,498x64 +9x64 direct proofs +9x64 zero-update equality checks =801,024**. The current path adds149,760 record exposures. This count denotes full three-role encoder/fusion record calls, not one count per role, and does not include internal activation-checkpoint recomputation or quantify the two backwards per selected history group. Peak recorded allocation spans **16,080.9473-16,087.1704MiB**. The source process lasted **23,330.50696s =6.48070h**, exceeding the original1-3h estimate; per-state measured times sum23,276.40114s. This is costly diagnostic evidence, not free memory learning. Runtime calls and repeat/direct checking costs must not be presented as a measured deployment-training cost.',
'',
'The negative partial-gradient cosine is not evidence of harmful multi-task conflict. `g_U` and `g_V` differentiate different inputs of the **same** scalar objective. Their cancellation can occur in distance-invariant coordinate directions. The mathematical note correctly presents a shared orthogonal rotation only as an example; no rotation generator or rotation-dominated mechanism is fitted/measured here (`R/docs/MSVR310_PARTIAL_VS_TOTAL_METRIC_GRADIENT_2026-09-08.md:7-45`). The data do not license a PCGrad conclusion, a causal account of the prior Q1 failure, a full symmetric large-batch objective, or a new-identity retrieval prediction.',
'',
'## Complete state/role numerical means',
'',
'Every row below uses all194 history batches for that state/role. These are arithmetic means of batch-wise ratios/cosines, not ratios of pooled norms, cosine of a mean gradient, or independent-sample estimates. Initial/control/fresh_memory name fixed checkpoints, not changing coordinates within this probe. Full min/max/quantiles and norm distributions are in `independent_checks.json`.',
'',
'| Fold | State | Role | mean norm(gV)/norm(gU) | mean cos(U,V) | mean cos(U,U+V) | mean cos(G,G+V) |','|---|---|---|---:|---:|---:|---:|']
csv_rows=[]
for s in d['states']:
 for role,a in s['roles'].items():
  row=dict(fold=s['fold'],state=s['state'],role=role,n=194,history_to_current_ratio=a['history_to_current_ratio']['mean'],current_history_cosine=a['current_history_cosine']['mean'],current_both_cosine=a['current_both_cosine']['mean'],task_total_both_cosine=a['task_total_both_cosine']['mean']);csv_rows.append(row)
  lines.append(f"| {s['fold']} | {s['state']} | {role} | {row['history_to_current_ratio']:.9f} | {row['current_history_cosine']:.9f} | {row['current_both_cosine']:.9f} | {row['task_total_both_cosine']:.9f} |")
lines += [
'',
'## F. Measurement classification and claim boundaries',
'',
'| Measurement | Classification | Claim ceiling |','|---|---|---|',
'| Source identity/scene masks and label maps | `real_gt`, dataset-label-based mechanism diagnostic | Real source-label relationships only; not heldout or official retrieval accuracy |',
'| T0 random tensors / linear map | `synthetic_proxy` | Synthetic scalar/gradient/VJP consistency |',
'| Current/history/task gradient norms and cosines, state equality, pixels, reencoding equality, memory upstream support | `runtime_model_witness` supplement | Recorded fixed-state derivatives/engineering behavior with declared precision limits |',
'| Saved matrix mining/loss checks | `remote_numpy_matrix_witness` supplement | Remote independent-implementation checking of saved matrices; no model recomputation and no local binary replay |',
'| Full text identity/queue/shape/scalar algebra/CSV/plot checks by this reviewer | `independent_deterministic_text_check` supplement | Local reproducible arithmetic and provenance consistency across every row |',
'| Orthogonal rotation note | Standard mathematical example | Possible cancellation mechanism; not an identified empirical explanation |',
'| Optimizer updates / heldout or official retrieval gains | `not_measured` | No performance, causality, generalization or training-update claim |',
'',
'Supported qualified claim: **Across all registered fixed-source history batches of all nine seed42 states, the recorded candidate-side role-encoder gradient is nonzero above same-graph repeat noise, opposes the current-side partial gradient, and changes the resulting role-restricted triplet/task-gradient direction. Complete independent text arithmetic and queue replay are consistent with those recorded summaries.**',
'',
'Unsupported extrapolations: independently reproduced model gradients; direct proof for every multi-group batch; harmful task conflict; rotation-dominated drift; missing history derivatives as the established cause of failed retrieval; an optimizer-direction or parameter-step effect; a predicted heldout mAP gain; state/seed independence; a new original VJP method; or automatic authorization to train. Nonzero contribution may justify proposing one separately registered fixed comparison of gradient scope. This audit neither registers nor starts that comparison.',
'',
'## Open limitations and concrete actions',
'']
lines += [f'{i+1}. {x}' for i,x in enumerate(limitations)]
lines += ['', 'Required executor handoff actions:', '']+[f'- **{x["priority"]}**: {x["action"]}' for x in actions]
lines += [
'',
'## Reproducibility and primary input hashes',
'',
'Run the independent, stdlib-only checker with the available local interpreter:',
'',
'```powershell',
"& 'C:\\Users\\gb\\AppData\\Roaming\\uv\\python\\cpython-3.13-windows-x86_64-none\\python.exe' 'C:\\Users\\gb\\.codex_tmp\\history_gradient_source_independent_audit_20260908\\independent_checks.py'",
'```',
'',
'It reads all source inputs without importing project/model/author-analysis code, performs the counts described above, and writes only this audit directory. `independent_checks.json` records the per-category check counts, every state’s statistics/coverage, all27 role distributions and complete input SHA inventory. The inventory has126 file entries including the checker itself; it is not126 independent experimental replicates. `audited_input_hashes.json` provides the same binding inventory separately. The five historical newline-equivalent references are retained explicitly rather than falsely reported as exact local-byte matches.',
'',
'| Primary input | SHA256 |','|---|---|']
for p in [R+'/configs/MSVR310/TriFusion-history-candidate-gradient-v1.json',R+'/refine-logs/msvr310_history_candidate_gradient_v1/EXPERIMENT_PLAN.md',R+'/protocols/msvr310_train_oof_v1.json',S+'/pipeline.json',S+'/source/summary.json',S+'/source/cpu_verification.json',P+'/all_statistics_postcheck.json',A+'/complete_text_reaggregation.json',A+'/all_role_history_steps.csv',F+'/plot_receipt.json']:
 lines.append('| '+p+' | `'+d['input_hashes'][str(Path(p))]['sha256']+'` |')
lines += ['', 'Freshness of this audit: exact source artifacts and hashes above were inspected after terminal completion. The unfinished tracker and absence of a new narrative are disclosed. Final prose should undergo the promised follow-up claim-number check.']

# Replace initial intake-only limitations with the completed additional claim review.
changes={
 'The tracker is stale (RUNNING/NOT_STARTED); terminal receipts establish completed source/CPU processing. No finished new narrative was supplied, so no narrative acceptance is implied.':'The updated tracker correctly records source/CPU completion. The completed narrative, all four tables and mirrored/PDF evidence also pass the independent checks recorded below.',
 'The tracker is stale (RUNNING/NOT_STARTED); terminal receipts establish completed source/CPU processing.':'The updated tracker correctly records source/CPU completion.',
 'The static tracker remains `source RUNNING / source CPU NOT_STARTED` (`R/refine-logs/msvr310_history_candidate_gradient_v1/EXPERIMENT_TRACKER.md:10-15`) despite these newer terminal records; it needs an executor update, not a rerun.':'The tracker was initially stale; the additional reviewed revision now correctly records source COMPLETE, source CPU PASS, and postprocessing completion (`R/refine-logs/msvr310_history_candidate_gradient_v1/EXPERIMENT_TRACKER.md:10-15`). Earlier running observations are explicitly retained as historical.',
 'The PDF bytes/hash are verified but no separate PDF render inspection was performed. No new source narrative was supplied to this review: artifact-number integrity is established within the limits above, while final narrative wording remains pending a separate check.':'The subsequent independent PDF check verifies all108 mean values and108 counts at their proper grid positions, confirms8 embedded images and0 out-of-page spans, reproduces the exact PDF render hash, and includes a reviewer visual inspection. The newly supplied source narrative has also been checked in full as detailed below.',
 'EXPERIMENT_TRACKER.md is stale at the audited input snapshot. A finished new source narrative was not supplied. The executor must update the tracker and pass the final narrative for a separate claim/number check.':'The initial stale tracker was corrected before final audit. The completed narrative and tracker revisions are preserved as byte-exact audit snapshots. The narrative line112 learning-direction wording is mildly ambiguous; prefer recorded role-block gradient direction, consistent with its line33 AdamW limitation.',
 'Update the source tracker from the verified terminal pipeline and retain the fixed-state/source-only status. Submit the completed narrative for a separate claim-number check; do not label this audit as narrative acceptance.':'Keep the completed tracker and fixed-state/source-only status. When appending audit closure, preserve the reviewed report revision and exact hashes; label this native same-family review provisional. Prefer recorded role-block gradient direction over learning direction in the final mechanism statement.',
 'Freshness of this audit: exact source artifacts and hashes above were inspected after terminal completion. The unfinished tracker and absence of a new narrative are disclosed. Final prose should undergo the promised follow-up claim-number check.':'Freshness of this audit: exact source artifacts and hashes above were inspected after terminal completion. The additional completed report, updated tracker, mirrors and PDF were independently checked before this final verdict; their reviewed revisions and hashes are preserved in this audit directory.'
}
for old,new in changes.items():
 lines=[line.replace(old,new) for line in lines]
 for key,check in checks.items():check['details']=check['details'].replace(old,new)
 limitations=[x.replace(old,new) for x in limitations]
 for action in actions:action['action']=action['action'].replace(old,new)
claim_section=['','## Additional completed narrative, mirrored evidence and PDF audit','',
 'The executor supplied `R/results/MSVR310_HISTORY_CANDIDATE_GRADIENT_SOURCE_2026-09-08.md` and its complete mirrored evidence after the initial source inspection. This reviewer independently checked the new artifacts against the original complete raw rows; the executor\'s report generator was inspected but not executed or imported. **The additional claim-number check PASSes.** No claim in the finished report is treated as measured retrieval improvement.',
 '',
 '- All49 copied raw/analysis/processing/figure files are byte-identical to their original audit-intake counterparts, and three copied driver/intake/PDF-verifier scripts are byte-identical to their actual sources.',
 '- All four report tables were independently reconstructed:49 data rows,371 cells and322 numerical tokens (including fold labels and split count cells). All36 explicit prose numeric checks pass, including process IDs/timestamps, complete scope, total exposures, GPU memory, direct-proof error, plotted counts and queue totals.',
 '- All global `report_numbers.json` means/minima/maxima/negative counts match direct reaggregation of all5,238 raw role/history rows. All10 queue totals and4 resource numbers match the independent primary checks.',
 '- The23 negative `cos(g_U,g_U+g_V)` rows are real and explicitly preserved: control Transformer7/3/11 across folds0/1/2, control Mamba1 in fold0, and fresh-memory Transformer1 in fold2. Every task-total cosine is positive. The complete source therefore does not support the old preflight all-positive current-versus-combined generalization (`R/results/MSVR310_HISTORY_CANDIDATE_GRADIENT_SOURCE_2026-09-08.md:39-74`).',
 '- The narrative retains the real-identity/current64-anchor definition, fixed-state rather than training-trajectory meaning, role-restricted G, lambda1, runtime gradient/state/14-component limits, single-group direct proof, unexercised capacity, repeated-exposure counts and no-optimizer/no-retrieval qualifiers (`R/results/MSVR310_HISTORY_CANDIDATE_GRADIENT_SOURCE_2026-09-08.md:13-15,29-35,76-78,96-116`).',
 '- The prior Q1 -0.07158042pp and two0/5 gate statements are checked as inherited sealed-result references, not remeasured in this source diagnosis. No broad claim of zero heldout artifact access is made; only no new heldout/official image forward is asserted (report line15).',
 '- An independent PyMuPDF1.28.2 reader matched all108 means and108 n labels to the raw-derived row/column position, a stronger check than value-multiset equality. It found one842.4000x597.6000pt page,8 embedded image objects, no out-of-page text spans, and reproduced exactly the supplied render SHA. The reviewer viewed that newly rendered page. This is a mixed vector/raster PDF, as correctly disclosed (report line106).',
 '- Chronology now agrees: source exit11:53:48.389849, CPU exit11:53:52.275165, postprocessing finish11:55:58.161377 Beijing. The tracker\'s current rows10-15 are complete/pass and preserve old running observations under Historical observations.',
 '',
 'One optional precision edit remains: report line112 says “实际改变学习方向”. Its detailed definition at line33 already excludes actual AdamW directions, so this is not an integrity failure; “实际改变所记录的角色参数块梯度方向” would remove residual ambiguity.',
 '',
 '| Descriptive global metric over all5,238 role/history rows | Mean | Minimum | Maximum | Negative |','|---|---:|---:|---:|---:|']
for field,s in claim_checks['global_statistics'].items():claim_section.append(f"| {field} | {s['mean']:.9f} | {s['minimum']:.9f} | {s['maximum']:.9f} | {s['negative']} |")
claim_section += ['',f"Audited completed report SHA256: `{claim_checks['report_sha256']}`. Audited updated tracker SHA256: `{claim_checks['tracker_sha256']}`. Full additional evidence is in `independent_claim_checks.json`. The corresponding unmodified bytes are retained as `reviewed_source_report.md` and `reviewed_tracker.md`; later status-only publication changes must not overwrite these snapshots."]
lines += claim_section
resolved={
 'The narrative line112 learning-direction wording is mildly ambiguous; prefer recorded role-block gradient direction, consistent with its line33 AdamW limitation.':'The optional line112 wording edit was resolved before final closure: it now explicitly says recorded role-block gradient direction, consistent with its line33 AdamW limitation.',
 'Prefer recorded role-block gradient direction over learning direction in the final mechanism statement.':'Retain the corrected recorded role-block gradient-direction statement.',
 'One optional precision edit remains: report line112 says “实际改变学习方向”. Its detailed definition at line33 already excludes actual AdamW directions, so this is not an integrity failure; “实际改变所记录的角色参数块梯度方向” would remove residual ambiguity.':'The optional precision edit is resolved: final report line112 now says “实际改变所记录的角色参数块梯度方向”. The final publication revision changes only report lines3/9/112/118 and tracker status/closure text; no scientific number or underlying source artifact changed. Both prepublication and final reviewed snapshots/digests are preserved.',
 'The corresponding unmodified bytes are retained as `reviewed_source_report.md` and `reviewed_tracker.md`; later status-only publication changes must not overwrite these snapshots.':'The final reviewed bytes are retained as `reviewed_source_report.md` and `reviewed_tracker.md`; the initial completed narrative and tracker revisions remain in `reviewed_source_report_prepublication.md` and `reviewed_tracker_prepublication.md`, and the earlier audit revision in the correspondingly named prepublication audit files. Later status-only publication changes must not overwrite these snapshots.'
}
for old,new in resolved.items():
 lines=[line.replace(old,new) for line in lines];limitations=[x.replace(old,new) for x in limitations]
 for action in actions:action['action']=action['action'].replace(old,new)
for source,target in [(Path(R)/'results/MSVR310_HISTORY_CANDIDATE_GRADIENT_SOURCE_2026-09-08.md','reviewed_source_report.md'),(Path(R)/'refine-logs/msvr310_history_candidate_gradient_v1/EXPERIMENT_TRACKER.md','reviewed_tracker.md')]:
 b=source.read_bytes();expected=claim_checks['report_sha256'] if target=='reviewed_source_report.md' else claim_checks['tracker_sha256'];assert hashlib.sha256(b).hexdigest()==expected
 (O/target).write_bytes(b)
body='\n'.join(lines)+'\n'
(O/'EXPERIMENT_AUDIT.md').write_text(body,encoding='utf-8')
(O/'reviewer_full_response.md').write_text(body,encoding='utf-8')
with (O/'verified_state_role_means.csv').open('w',newline='',encoding='utf-8') as stream:
 w=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
all_inputs=dict(d['input_hashes']);all_inputs.update(claim_checks['input_hashes'])
(O/'audited_input_hashes.json').write_text(json.dumps(all_inputs,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
report=dict(audit_skill='experiment-audit',verdict='WARN',reason_code='runtime_binary_provenance_and_single_group_direct_proof_limits',summary=lines[2],reviewer='native Codex fresh-context auditor',reviewer_model='gpt-6-astra',reviewer_reasoning='max',reviewer_family='openai',review_independence='same-family',acceptance_status='provisional',agent_id='/root/audit_msvr_history_gradient_source',date='2026-09-08',generated_at=datetime.now(timezone.utc).isoformat(),overall_verdict='warn',integrity_status='warn',deterministic_checks_status='pass',scientific_qualification=None,narrative_claim_check='PASS_COMPLETE_NUMBERS_AND_SCOPE_WITH_MINOR_WORDING_NOTE',checks=checks,limitations=limitations,action_items=actions,totals=d['totals'],audited_input_hashes=all_inputs,audited_input_hashes_path=str(O/'audited_input_hashes.json'),evidence_path=str(O/'independent_checks.json'),claim_evidence_path=str(O/'independent_claim_checks.json'),reviewed_report_sha256=claim_checks['report_sha256'],reviewed_tracker_sha256=claim_checks['tracker_sha256'],read_only_scope=dict(model_forwards=0,model_backwards=0,optimizer_updates=0,remote_commands=0,scientific_files_modified=0),claims=[dict(id='fixed_state_source_candidate_derivative_contribution',impact='supported_with_runtime_witness_qualifier'),dict(id='harmful_task_conflict_or_causal_failure_mechanism',impact='unsupported'),dict(id='optimizer_or_generalization_gain',impact='not_measured'),dict(id='full_multi_group_direct_gradient_reproduction',impact='unsupported'),dict(id='final_narrative_numbers',impact='pass_complete_numeric_check')])
(O/'EXPERIMENT_AUDIT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
report['narrative_claim_check']='PASS_COMPLETE_NUMBERS_AND_SCOPE_WORDING_PRECISION_RESOLVED'
report['reviewed_prepublication_snapshots']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in O.glob('*prepublication*')}
(O/'EXPERIMENT_AUDIT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
(O/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
receipts={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in O.iterdir() if p.is_file() and p.name!='audit_artifact_sha256.json'}
(O/'audit_artifact_sha256.json').write_text(json.dumps(receipts,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipts,indent=2))
