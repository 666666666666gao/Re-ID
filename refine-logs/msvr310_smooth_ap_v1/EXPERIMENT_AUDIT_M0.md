# MSVR310 Smooth-AP M0 independent engineering integrity audit

**Overall verdict: WARN. Engineering status: PASS_WITH_LIMITS. Deterministic verification: PASS for the explicitly enumerated checks.**

The completed, registered M0 is supported by complete saved artifacts and independently reconstructed arithmetic. I found no integrity FAIL in the checked identity provenance, objective definition, saved metrics, queue, update coverage or checkpoint binding. The WARN concerns the ceiling of the evidence: image forwards, historical feature/parameter vectors, optimizer trajectories and output-reload tensors were not independently regenerated or retained as replayable arrays. Their assertions remain runtime witnesses. This audit gives no verdict about Q1 retrieval, scientific qualification or the overall research goal.

**Reviewer:** fresh Codex `gpt-6-astra`, reasoning `max`; canonical reviewer `/root/audit_msvr_smooth_ap_m0`. `review_independence: same-family`; `acceptance_status: provisional`. Deterministic arithmetic is separately accepted within its stated scope. No cross-family review or independent backend identity attestation is claimed.

**Date:** 2026-09-09, Asia/Shanghai. Request and run were registered on 2026-09-08. The reviewer read the listed primary artifacts directly. Workspace instructions and prior audit bindings were not treated as evidence that this M0 passed.

## Scope, roots and evidence

In file:line references below, `R/` means `/root/autodl-tmp/trifusion-v2/TriFusion-ReID/`, and `M/` means `/root/trifusion-storage/artifacts/msvr310_smooth_ap_v1_seed42_2e947a4/`. Their text snapshots are under this audit directory's `snapshots/remote/`, retaining the absolute remote path beneath that folder. `A/` means this audit directory, `C:/Users/gb/.codex_tmp/smooth_ap_m0_independent_audit_20260908/`. `L/` means `C:/Users/gb/.codex_tmp/smooth_ap_m0_complete_20260908/`.

The audit consumed M0/T0 artifacts, source checkpoint and label metadata, completed antecedent files required by recursive configuration, and code. It did not read any current Q1 retrieval file, official-test image or image content. Four remote checks ran with CUDA hidden, NumPy/BLAS and PyTorch limited to 2 threads, no model construction or model forward/backward, no optimizer, no remote writes and no process mutation. Pure loss autograd on CPU distance tensors is explicitly distinguished from a model backward.

## Deterministic results

| Independently checked object | Coverage / result |
|---|---|
| Registered recursive bindings | 178 hash comparisons; all match the remote bytes |
| Audit input hash inventory | 440 distinct local/remote path entries; includes duplicate content at different paths |
| Remote text snapshots | 232 snapshots; all final snapshot bytes match the source hash |
| Source metadata | 1,032 aligned triplets, 3,096 existing nonempty modality paths, 155 identities; filename identity/camera/scene and manifest exactly agree |
| Complete registered source sampler | All 780 batches reproduced independently, 260 per fold |
| T0 memory admission | All 780 queue rows/ages reproduced; first history at registered step 67; fold maxima 356/372/362 candidates |
| M0 length | 6 capacity endpoints × 8 updates + 2 fixed-batch endpoints × 100 = 248 updates |
| Saved AP coverage | All 15,872 anchors; maximum saved-FP32 versus independent-Float64 AP error **2.3559432371644817e-7** |
| Saved distance coverage | All 4,945,920 values, four spaces, exact offsets and EOF; finite, nonnegative, in the unit-distance range |
| Distance derivatives | 1,236,480 fused-distance entries per objective per dtype; analytic Smooth-AP Jacobian versus deployed loss autograd max error **3.122502256758253e-17** in Float64 and **1.7623613799214177e-8** in Float32; hard gradients exact |
| Scalars / other branch Triplets | Maximum hard/AP/basic scalar discrepancy **8.65389654380877e-8**; three original full-branch Triplet discrepancies at most **1.227017492055893e-7** |
| Weighted objective ledger | All 248 × 14 logged components checked; maximum sum error **5.62518835067749e-7** |
| M0 checkpoints | All 6 hashes, bindings and full 472-tensor state reconstructions match; 241 baseline aliases and 248 frozen-state tensors verified per checkpoint |
| Source checkpoints | All 3 hashes, source/heldout IDs, fold, config and protocol bindings checked; all 1,950 recorded source training steps remain within their source fold |
| Synthetic edge algebra | 40 finite-difference positions over two fixtures; largest discrepancy **2.2290467249774792e-9**; self, permutation, hard tie and unit-cosine checks pass |

Evidence: `A/remote_arithmetic.py:34` defines the independent AP/Jacobian derivation; the entire script, `A/independent_step_checks.jsonl:1` through its 248 records, `A/arithmetic_summary.json`, `A/remote_provenance.py`, `A/provenance_summary.json`, `A/remote_source_checkpoint_bindings.stdout`, `A/inventory_check.json`, `A/snapshot_validation.json`. Derivatives were reconstructed from the saved distance inputs and compared against the deployed objective function. They were **not saved model parameter gradients**.

## A. Ground-truth provenance and identity isolation — PASS

Identity, camera and scene come from dataset filenames and the hash-bound label manifest, not model predictions. I reconstructed all protocol records and the eligible/ineligible identity round-robin split. The source/heldout identity partitions are disjoint: source identities **103/103/104**, source records **672/683/709**, heldout identities **52/52/51**, heldout gallery records **360/349/323**. The metadata-only internal retrieval protocol retains **600 eligible queries from 60 identities**, while **432 gallery-only records** with no legal query positive remain as distractors. This is protocol verification, not a retrieval measurement.

Metric identities use the original IDs; CE uses a checked bijective source-only label map. Class zero remains legal. Current anchors are the real B64/K8 source batches. No scene mask enters training-positive or training-negative selection; same-scene different-identity negatives are retained. Only the registered retrieval mask removes same-ID AND same-scene entries.

The complete model path starts from the bound raw CLIP file and each fold's source-only Signal checkpoint, loads that state strictly, freezes Signal and its shared tail, and initializes new roles at seed 42. All three source checkpoint payloads and 650 recorded training steps per fold match this partition. The frozen field, reference tail and camera embeddings belong to the same source checkpoint; no all-fit or heldout-trained role checkpoint enters M0.

Evidence: `R/tools/build_msvr310_train_oof_protocol.py:11-104`; `R/tools/train_msvr310_signal_oof.py:31-76,89-109,324-327`; `R/tools/train_msvr310_trifusion_oof.py:27-55`; `R/modeling/trifusion/signal_preserving_v8.py:55-77,100-149`; `R/tools/train_msvr_smooth_ap.py:104-123`; `A/provenance_summary.json:3-39`; `A/remote_source_checkpoint_bindings.stdout:1`.

Limit: this audit checked label provenance and present source file metadata, not source-image content against the original archive. The install archive SHA/CRC remains a historical installation receipt (`R/evidence/msvr310_dataset_install_20260905.json:4-9`); it is not represented as a fresh archive rehash.

## B. Objective definition, normalization and arithmetic — PASS

For anchor i, the positive set contains every same-identity candidate position except i. With score `s = 1 - d²/2`, each positive p receives the ratio of its smoothed positive rank to its smoothed full rank. Both ranks start at one and sum `sigmoid((s_j-s_p)/0.01)` over eligible j; anchor i and the comparison p-to-p are excluded. AP averages over all positives, and the loss is `1 - mean(anchor AP)`. This follows the rank exclusions in Equations 3/5 and the sigmoid/mean-loss construction of Equations 6/7 in [Brown et al., Smooth-AP](https://arxiv.org/html/2007.12163v2). Temperature 0.01 is fixed. The actual embedding is normalized before Euclidean distance, making the cosine conversion valid; the synthetic unit-vector equality error was 6.38378239159465e-16.

There is no self-max score rescaling. Candidate rank denominators, positive-set size, anchor averaging, and ordinary unit embedding normalization implement the stated objective. They do not turn an engineering comparison into a benchmark score. Smooth-AP ties use sigmoid(0)=0.5. All saved self-distance derivatives are zero. Current duplicate views remain separate positive positions; historical records are unique and current-record copies are removed.

The control retains the unsquared Euclidean margin-0.3 hinge. Within each current/history max/min group, the first tied extremum receives the derivative; equality between the current and historical extrema splits the derivative equally. Independent algebra matches the deployed hard derivatives at every saved distance and in an explicit cross-group tie fixture. This convention is narrower than treating a concatenated maximum as having arbitrary or evenly distributed within-group tie gradients.

M0 steps 1–2 use the original batch-only objective at both endpoints. Step 3 activates the registered endpoint objective and queues current records after updating; step 4 first consumes history. The registered long-run source queue uses steps 1–65 as warmup, activates at 66 and first consumes history at 67. The latter result is a source sampler/T0 replay only; it is not a statement about observed Q1 scores.

Only `components.triplet_fused` is replaced; the explicit `active_fused_metric` identifies its changed meaning. The original seven CE terms and six other Triplet terms retain their definitions and weights. I additionally reconstructed the three full-branch Triplets from their saved current-distance matrices. The seven CE and three residual Triplet terms have no saved logits/residual distances; their numerical values are runtime scalars, and this audit checks their inclusion, finite values and weights, not fresh forward-derived reconstruction. Equal definitions do not imply equal values after the paired trajectories diverge.

The overfit floor is independently `0.75 × H([0.9+0.1/103, 0.1/103, ...]) = 0.5857136327437849`. First loss is **4.122129917144775** at both endpoints. Final control/candidate losses are **0.5881943702697754 / 0.5881884098052979**; excess-loss ratios are **0.0007014834585320135 / 0.0006997980052374365**, both below the registered 0.1 threshold. These are reductions of differently defined training objectives. The fixed 64 views contain **53 unique records**, and exclude every historical copy, so this overfit test gives no historical-backpropagation overfit evidence.

Evidence: `R/tools/msvr_smooth_ap.py:11-62`; `R/tools/msvr_instance_memory.py:15-37,40-80`; `R/tools/train_msvr_smooth_ap.py:122-157,246-262,365-374`; `R/modeling/trifusion/criterion.py:17-34`; `R/modeling/trifusion/signal_preserving_v8.py:696-742`; `R/tools/run_signal_preserving_v5.py:99-134,1580-1608`; `R/refine-logs/msvr310_smooth_ap_v1/TRAINING_PLAN.md:9-19,29-33`; `A/arithmetic_summary.json`; `A/provenance_summary.json:40-62`.

## C. Real artifacts, stage coverage and checkpoint binding — PASS with output-reload limitation

The original execution commit is **2e947a4325144e37fed638105ac954e7e54b5fe5**; config SHA is **974328fee25985b19aa36c84f57a557f9120993fecb98b2a902a1b8de8475302**; M0 summary SHA is **64ebb7ef11a80afc19f77a87c16a49c239b3e5022e161d1cd8ad340d2c8699e4**. Recursive bindings, source files, source checkpoints, M0 distance files, step logs and six output checkpoints exist and pass the stated hash comparisons. All 29 remote-origin files in the local intake, including the complete summary, match the remote bytes; the intake directory additionally contains its own manifest.

Original stage receipts show **T0 5.557589367 s, M0 659.747404633 s and M0_CPU 10.186520290 s**, each exit 0 and in the registered order. At the audit process snapshot the original wrapper PID **48170** remained alive, its original Q1 child **49157** existed with that wrapper as parent, and completed T0/M0/M0_CPU PIDs were absent. No process was started, stopped or restarted by the auditor except the read-only audit interpreters. No current Q1 result file was read.

Every capacity endpoint has eight recorded updates and a strict six-checkpoint reload path. Independent reconstruction combines the raw source tensors and saved alias map with the saved role tensors, and matches the final whole-state and frozen-state digests. The six strict output equality flags are backed by executed assertions in the trainer and exit-0 receipts, but their before/after output arrays were not retained. Their bitwise output equality was **not independently re-executed** in this audit.

The current repository HEAD advanced while documentation/evidence was being added. The captured commit path diff contains documentation, receipt and analysis-preparation paths; it contains no changes to the registered training code or configs. Current relevant remote hashes remain pinned. Five old local source files differ from the remote only in CRLF/LF bytes; their local hashes are not substituted for the registered remote hashes.

Evidence: `M/pipeline.json:1-91`; `M/m0_cpu.json:121-123,275-278`; `R/tools/run_msvr_smooth_ap.py:14-65`; `R/tools/train_msvr_smooth_ap.py:294-384`; `R/tools/train_msvr_instance_memory.py:94-110`; `R/tools/train_msvr310_source_style.py:162-174`; `A/inventory_check.json`; `A/arithmetic_summary.json:120-221`; `A/provenance_summary.json`.

## D. Forward/loss/backward path and gradients — WARN for reconstruction limits; inspected path is coherent

Both endpoints capture the original frozen anchor/reference/baseline fields at role entry, plus CPU/CUDA RNG. The position map and instance memory select the latest duplicate view consistently. Historical refresh re-encodes the entire original 64-record group with the **current** encoder parameters and its original RNG, preserving the original batch semantics. Expired groups are removed; historical candidates supply columns, not extra loss anchors.

The current graph uses detached refreshed historical outputs. A separate historical leaf computes the candidate partial for exactly the selected hard or Smooth-AP scalar. Each nonzero group's original 64-record encoder/fusion graph is then replayed; its coefficients produce the VJP through the normalization and encoder. The scaled history VJP is added to the current scaled gradients with weight 1, before unscale and one AdamW step. The selected domain is **189 encoder tensors**; the other **14 trainable tensors** are seven neck weights and seven classifiers, which historical fused loss does not depend on. Signal, the shared pretrained tail and fusion remain frozen/parameter-free as applicable.

All six capacity endpoints first witness nonzero history contributions in all three roles at **step 4**. Each has one single-history-group direct-graph check covering the full current objective plus historical candidate contribution over the 189 encoder tensors. Maximum logged relative L2 difference is **2.0929868300782436e-5**, below **0.005**. I recomputed its reported ratio and the associated norm/cosine/difference identities, and checked the analogous recorded arithmetic across all 248 steps. This validates the internal arithmetic of those recorded summaries; it does not independently recover their parameter vectors.

**Unavailable, therefore not PASS:** per-step historical embedding vectors and their stored RNG/field tensors were transient; historical leaf vectors and encoder parameter gradients were not saved; optimizer states/parameter deltas per update were not saved. The scalar leaf norms cannot be reconstructed from pairwise distances alone. Thus the all-step fresh-coordinate, bitwise re-encoding, gradient-addition, finite-gradient and cumulative 203/203 coverage assertions remain runtime witnesses. The direct check is six selected single-group observations, not all-step or arbitrary multi-group direct-graph verification. The present contract and M0 report already state this ceiling; I found no claim in them that requires pretending otherwise.

Evidence: `R/tools/msvr_freshness_probe.py:43-60`; `R/tools/probe_msvr_role_set_gradients.py:29-50`; `R/tools/probe_msvr_history_candidate_gradients.py:43-66`; `R/tools/train_msvr_smooth_ap.py:69-94,143-181,186-260`; `R/modeling/trifusion/signal_preserving_v8.py:283-291,356-368,494-517,606-653`; `M/m0/fold_0_smooth_ap/memory_steps.jsonl:4`; the same line in all six capacity logs; `M/m0/fold_0_smooth_ap/training.json:748-802`; `M/m0/overfit_smooth_ap/training.json:9028-9032`; `A/cost_and_runtime_summary.json:71-132`.

## E. Scope, pairing and actual cost — PASS for registered M0 scope

This is one seed, three source folds, two endpoints, one fixed objective intervention. Both endpoints use the same initial binding and all corresponding record-index/pixel-SHA sequences match. Capacity training sees **333/316/350 unique records** per respective fold; overfit uses the same **53 unique records** in 64 sampled positions. These counts are not independent test examples. M0 records **15,872 current-anchor exposures**, **3,448 historical candidate exposures**, **6,272 extra fresh-role record forwards**, **5,760 extra history-VJP record forwards**, and **384 extra direct-check record forwards**. Those last counts describe role re-encoding work, not extra raw image reads or extra optimizer updates. Activation-checkpoint backward recomputation is additional internal work, not separately enumerated in these record-forward counters.

Summed recorded fit time is **482.814781586 s**, within the **659.747404633 s** M0 stage elapsed time. Peak allocation is **11,267.1171875 MiB** and peak reserved is **12,116 MiB**. The completed M0 directory occupies **219,196,319 bytes**, including **19,783,680 bytes** of saved training distances; current Q1 storage was not inventoried. These are run receipts whose arithmetic and aggregation were checked, not a fresh performance benchmark. The sampler/age replay covers the registered 512-capacity, age-8 queue; actual short capacity M0 only reaches history age 5. No inference-parameter increase arises from the objective. Parameter domains and trainable element counts match the checkpoint tensors.

The two original five-gate groups, the fixed final epoch, and the restriction against using M0 as retrieval qualification remain visible in the contract and trainer/verifier code. No conclusion is drawn about the ongoing Q1 experiment, and this audit requests no code, contract, process or training change.

Evidence: `R/refine-logs/msvr310_smooth_ap_v1/TRAINING_PLAN.md:19-41`; `R/tools/train_msvr_smooth_ap.py:83-107,249-280,355-361`; `R/tools/verify_msvr_smooth_ap.py:263-311` (static gate definitions only); `A/arithmetic_summary.json:5-117,223-283`; `A/cost_and_runtime_summary.json:2-11`.

## F. Evidence classification and claim impacts — PASS, with explicit ceilings

| Evidence | Classification | Supported claim |
|---|---|---|
| M0 supervision and approximate AP | `real_gt` training evidence | Uses dataset identities on source records; no heldout retrieval claim |
| Saved-distance AP/hinge/Jacobian and ledger replay | Deterministic engineering arithmetic over real-label inputs | Formula and saved-number consistency |
| Handcrafted tensors, finite differences, tie/permutation checks | `simulation_only` algebra fixtures | Implementation mathematics only |
| Standalone Signal equality and strict output reload | `synthetic_proxy` / engineering output consistency | Equality to another execution; never dataset performance GT |
| Direct-graph/VJP and gradient-norm summaries | Real-input runtime engineering witnesses plus deterministic summary arithmetic | Qualified evidence of the implemented historical path at the recorded probes |
| Current Q1 retrieval and official performance | Not consumed / not audited | No supported outcome from this audit |

The claims “the full registered M0 finished,” “the saved Smooth-AP math and ledger are correct,” and “six complete checkpoint states reconstruct from source bindings” are supported. “All 248 model parameter gradients or all reload outputs were independently replayed,” “203 tensors are nonzero at every step,” “fixed-batch overfit verifies history backpropagation,” and “M0 demonstrates retrieval improvement” are unsupported statements and must not be introduced. The actual contract/report appropriately qualifies those distinctions.

## Audit attempts, artifacts and handoff

One transport attempt failed before the remote payload ran because the private helper expected `stdout.reconfigure` and the auditor's `StringIO` capture did not supply it. A standard in-memory `TextIOWrapper` corrected this audit-only issue. The failed command, error and original transport script are preserved under `attempt_01_transport_failure.txt` and `remote_readonly_attempt01.py`; helper contents/credentials were never captured.

Final snapshot validation also found 13 audit text copies normalized from CRLF to LF by the initial text intake. The original remote raw hashes had always been calculated correctly, and all CPU arithmetic had read the real remote files. The auditor preserved the normalized copies and failed snapshot inventory, then restored exact bytes from the local intake whose raw SHA already equalled the remote source SHA. Final validation is 232/232 exact. See `snapshot_validation_attempt1.json`, `attempt_02_normalized_text_snapshots/`, `snapshot_crlf_repairs.json` and `snapshot_validation.json`. No source artifact was changed.

The principal outputs are `EXPERIMENT_AUDIT.md`, `EXPERIMENT_AUDIT.json`, and the identical full reviewer text `reviewer_full_response.md`. Independent scripts, stdout/stderr, exact commands, 248 step checks, source/checkpoint evidence, cost summaries, all audited-input hashes, and trace metadata accompany them in this directory. Python-package source/version snapshots describe the current installed dependencies; they are not a timestamped attestation of every compiled kernel loaded during the completed M0.

**Disposition:** retain the M0 as **engineering PASS_WITH_LIMITS / overall WARN**, with same-family provisional semantic acceptance. The deterministic subset passed. No corrective training or running-code change is indicated by this audit; a later complete Q1 audit remains a separate task.
