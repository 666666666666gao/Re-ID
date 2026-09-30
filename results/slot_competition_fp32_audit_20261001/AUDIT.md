# FP32 slot-competition experiment audit

**Overall: WARN. Numerical/artifact integrity: PASS. Registered advancement gate: FAIL.**

Fresh native Codex reviewer: **gpt-6-astra / max**, same-family, provisional. Date: 2026-10-01, Asia/Shanghai. Scope is the six endpoints in `slot_competition_fp32_roles_20261001_r2` only. This is not a SOTA, novelty, or historical baseline audit.

Local paths are relative to `C:/Users/gb/.trifusion_github_publish_22c3bee`. Training/evaluation source and protocol references below use the authoritative remote `/data/gaob/Re-ID/Trifusion` unless identified as local. [AUDIT.json](AUDIT.json) records input hashes, actual verification scripts/results, failures and full endpoint details. Reviewer writes were limited to these two audit files.

## What the evidence establishes

Six fresh seed42 endpoints completed 300 role-stage epochs, with **20,416 formal batch lines and 48 M0 batches**. All 18 M0/train/evaluate children have recorded actual waited exit codes zero. Every endpoint selects one official fused-mAP best from its complete 50-epoch history; all metrics use that checkpoint.

All **68 intake text artifacts** match their SHA and byte counts both locally and at the remote source. All **213 bound remote source/protocol/config hashes** match. A manifest green was not treated as completion proof: raw logs, per-batch accounting, checkpoints, distance arrays, metadata and selection were checked separately.

Independent CPU ranking of **18 complete distance matrices / 72 metrics** matches official receipts within **2.9367886185127645e-6 percentage points**. Full split file enumeration and all ordered ID/camera/scene metadata match. The primary script passed 282 aggregate checks and 18,852 individual query/output positive-count checks. Another 149 checks verify the analysis report against primary evidence; check counts are not a statistical confidence measure.

The reviewer also rebuilt all six production factories on CPU. Full initial state hashes and trainable parameter counts match their recorded values and each pair. All six best checkpoints and six M0 probes strictly loaded through the production loader. This used explicit CPU device redirection only in the reviewer process; **no neural forward/backward, GPU inference, optimizer step or training was run**.

| Endpoint | M0 PID / exit | Train PID / exit | Evaluate PID / exit | Formal batches | Best epoch |
|---|---:|---:|---:|---:|---:|
| RGBNT100_competitive | 1394247 / 0 | 1396301 / 0 | 1632435 / 0 | 6559 | 1 |
| RGBNT201_competitive | 1394248 / 0 | 1396308 / 0 | 1458373 / 0 | 2649 | 2 |
| MSVR310_competitive | 1403021 / 0 | 1404310 / 0 | 1442406 / 0 | 1000 | 10 |
| RGBNT100_independent | 1446698 / 0 | 1448171 / 0 | 1621516 / 0 | 6559 | 1 |
| RGBNT201_independent | 1461508 / 0 | 1462760 / 0 | 1520298 / 0 | 2649 | 2 |
| MSVR310_independent | 1516394 / 0 | 1517992 / 0 | 1552100 / 0 | 1000 | 15 |

| Endpoint | Best epoch | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|---:|
| RGBNT100_competitive | 1 | 85.140472 | 95.160347 | 95.685130 | 96.034986 |
| RGBNT201_competitive | 2 | 72.599716 | 74.043059 | 83.133972 | 87.918663 |
| MSVR310_competitive | 10 | 52.147941 | 67.174280 | 82.571912 | 88.494080 |
| RGBNT100_independent | 1 | 85.107564 | 95.043731 | 95.685130 | 95.976675 |
| RGBNT201_independent | 2 | 72.511444 | 73.803830 | 82.655501 | 87.918663 |
| MSVR310_independent | 15 | 52.031884 | 67.512691 | 82.402706 | 87.817258 |

## A. Dataset GT provenance — PASS

Identities and environment labels come from registered dataset records, not model outputs. The protocol builder reads identity/environment metadata, verifies paths and train/evaluation disjointness, and assigns only training labels: `tools/build_official_three_dataset_protocols.py:32`, `:44`, `:49`, `:56`. The loader retains record order and uses original evaluation identities: `tools/official_three_dataset_data.py:8`; the scoring arrays are built from these records at `tools/run_correspondence_context_identity.py:207`.

The reviewer enumerated every physical file in the nine train/query/gallery split views, parsed every identity and camera/s### filename against metadata, and compared every saved ordered ID/camera/scene array to its protocol. All queries have eligible positives.

| Dataset | Train records / IDs | Query records / IDs | Gallery records / IDs | Active exclusion |
|---|---:|---:|---:|---|
| RGBNT201 | 3951 / 171 | 836 / 30 | 836 / 30 | Same ID and camera |
| RGBNT100 | 8675 / 50 | 1715 / 50 | 8575 / 50 | Same ID and camera |
| MSVR310 | 1032 / 155 | 591 / 52 | 1055 / 155 | Same ID and s### environment/time label |

Different-identity distractors in the same environment remain. MSVR310 uses s###, not v# camera/view, for exclusion: `comparators/Signal-cd1b0a6/utils/metrics.py:68`, `tools/train_msvr310_signal_oof.py:227`. Camera filtering is at `comparators/Signal-cd1b0a6/utils/metrics.py:137` and `tools/train_rgbnt100_signal_oof.py:257`.

Protocol count anchors: `logs/official_three_dataset_protocols_20260923/RGBNT201.json:78309`, `RGBNT100.json:218978`, `MSVR310.json:38538`; each environment key is at line 7. This verifies installed-data metadata provenance; original external dataset archives were not redownloaded or fully rehashed.

## B. Score normalization — PASS

AP divides by GT positive counts and Rank-k averages query successes. Values are reported as percentages, not divided by the model's own prediction maximum/minimum: `tools/train_rgbnt100_signal_oof.py:253`, `tools/train_msvr310_signal_oof.py:223`, `tools/run_correspondence_context_identity.py:222`.

Embedding L2 normalization at `tools/run_official_three_dataset_roles.py:230` and attention allocation at `modeling/trifusion/slot_competition_roles.py:9` are representation operations, not performance-score normalization. The full FP32 attention block is explicit at `modeling/trifusion/slot_competition_fp32_roles.py:10`.

Independent AP/ranking calculations reproduce all 72 values. Small CMC differences are consistent with upstream float32 aggregation at `comparators/Signal-cd1b0a6/utils/metrics.py:104` and `:166`.

## C. Result existence and exact selection — PASS

All six best files, six M0 probes and six saved distance files exist remotely and were independently hashed. Checkpoint headers bind the correct dataset, seed, protocol, baseline, FP32 architecture and normalization arm. Best epoch and all four metrics agree across checkpoint, official receipt, complete history and accepted matrix. All 300 epoch log events exactly match the training JSON histories; six evaluation logs match the official receipts. Formal batch sequences and reconstructed loss means agree.

All 18 actual exits are evidenced at these files/lines, with process.wait at `tools/queue_slot_competition_fp32_roles.py:102`:

- `logs/slot_competition_fp32_complete_intake_20261001/RGBNT100_competitive/campaign.json:53`, `logs/slot_competition_fp32_complete_intake_20261001/RGBNT100_competitive/campaign.json:100`, `logs/slot_competition_fp32_complete_intake_20261001/RGBNT100_competitive/campaign.json:147`.
- `logs/slot_competition_fp32_complete_intake_20261001/RGBNT201_competitive/campaign.json:53`, `logs/slot_competition_fp32_complete_intake_20261001/RGBNT201_competitive/campaign.json:100`, `logs/slot_competition_fp32_complete_intake_20261001/RGBNT201_competitive/campaign.json:147`.
- `logs/slot_competition_fp32_complete_intake_20261001/MSVR310_competitive/campaign.json:53`, `logs/slot_competition_fp32_complete_intake_20261001/MSVR310_competitive/campaign.json:100`, `logs/slot_competition_fp32_complete_intake_20261001/MSVR310_competitive/campaign.json:147`.
- `logs/slot_competition_fp32_complete_intake_20261001/RGBNT100_independent/campaign.json:53`, `logs/slot_competition_fp32_complete_intake_20261001/RGBNT100_independent/campaign.json:100`, `logs/slot_competition_fp32_complete_intake_20261001/RGBNT100_independent/campaign.json:147`.
- `logs/slot_competition_fp32_complete_intake_20261001/RGBNT201_independent/campaign.json:53`, `logs/slot_competition_fp32_complete_intake_20261001/RGBNT201_independent/campaign.json:100`, `logs/slot_competition_fp32_complete_intake_20261001/RGBNT201_independent/campaign.json:147`.
- `logs/slot_competition_fp32_complete_intake_20261001/MSVR310_independent/campaign.json:53`, `logs/slot_competition_fp32_complete_intake_20261001/MSVR310_independent/campaign.json:100`, `logs/slot_competition_fp32_complete_intake_20261001/MSVR310_independent/campaign.json:147`.

For each of those six endpoint folders: `training.json:651` records the selected epoch, `training.json:49` starts the full history, `official_metrics.json:6` records the reload epoch, and `official_metrics.json:16` / `:22` contain fused/diagnostic metrics. The exact checkpoint loader checks architecture, arm, metadata and state-key identity before strict loading: `tools/run_slot_competition_fp32_roles.py:52`.

The new narrative `results/SLOT_COMPETITION_FP32_COMPLETE_2026-10-01.md:3` and its SUMMARY were directly read. Pair repairs/new errors, per-query AP bytes, identity means/bootstrap, readout counts, slot aggregation and trajectory endpoints agree with evidence. SUMMARY SHA is `94179e2b2ca8810a0a6ee2d7aed5eca001c8aad7b77681fa77a4d4ea3d1c3950`.

Original neural diagnostics also have real execution evidence: `logs/slot_competition_fp32_slot_diagnosis_20261001/EXECUTION.json:1`; retained launcher wait is at `logs/slot_competition_fp32_slot_diagnosis_20261001/launcher_source.py:55`. Its three children exited zero. Six endpoint diagnostic receipts bind the correct checkpoint/protocol, all query/gallery records, unchanged state, zero first-batch instrumentation difference and metric parity. The reviewer did not rerun those neural diagnostics.

The CPU report's exit zero is recorded at `logs/slot_competition_fp32_analysis_execution_20261001.json:6`; line 10 explicitly attributes it to the executor's native-tool observation. This is **not a reviewer-observed OS exit**. Independent file/hash/arithmetic checks corroborate the produced report.

At direct read the canonical handoff ended at §41.685. No future §41.686 narrative was presumed.

## D. Called path, M0 and paired initialization — PASS

Actual commands invoke the FP32 entry (`tools/queue_slot_competition_fp32_roles.py:67`). Its main installs the FP32 factory and strict checkpoint functions (`tools/run_slot_competition_fp32_roles.py:79`), then the context entry installs its train/evaluate functions into the inherited runner (`tools/run_correspondence_context_identity.py:244`, `tools/run_correspondence_roles.py:291`).

The active evaluator strictly reloads one best, extracts complete ordered records, calls the upstream camera/MSVR metric and independent scorer, then writes all three outputs: `tools/run_correspondence_context_identity.py:175`, `:188`, `:194`, `:214`, `:228`, `:240`. Real logs and arrays support this call path. Analysis uses the imported analyze/compare/audit functions; unrelated historical mains are not treated as this panel's execution.

Real M0 uses dataset loaders, production Mamba, backpropagation and eight batches: `tools/run_correspondence_context_identity.py:81`, `:102`, `:119`; `modeling/trifusion/experts/mamba.py:29`. Each endpoint's `m0/training.json:58` reports 127/127 nonzero-gradient tensor coverage, `:61` reload difference zero, and `m0/training_steps.jsonl:1` through line 8 provides real batch scalars.

**Gradient coverage is cumulative across eight batches**, because line 122 updates a set of tensors that have ever had a nonzero gradient. It does not mean every tensor is nonzero in every batch. Frozen baseline checks and M0 strict neural reload are in the original execution path at `tools/run_correspondence_context_identity.py:147` and `:155`. The separate CPU algebra fixture correctly disclaims production Mamba testing at `logs/slot_competition_fp32_complete_intake_20261001/contract/FP32_CPU_CHECK_20261001.json:4`.

Independent CPU reconstruction gives 127 trainable tensors and scalar counts RGBNT201 **2,805,706**, RGBNT100 **2,619,850**, MSVR310 **2,781,130**. Each pair's complete initial SHA matches its original receipt (`training.json:25`, `:26`). All 12 strict loads succeed with the baseline unchanged.

## E. Scope and mechanism interpretation — WARN

This is three datasets × two arms × **one development seed**, with 50 role-stage epochs after baseline training. Official evaluation runs every epoch and selects fused mAP-best, choosing the last epoch on a tie: `tools/run_correspondence_context_identity.py:138`, `:140`. Official data has also informed prior method decisions. These results are not untouched-test, seed-stability, novelty, SOTA or three-role-necessity evidence. The current narrative acknowledges this at `results/SLOT_COMPETITION_FP32_COMPLETE_2026-10-01.md:61`.

| Competitive minus independent | ΔmAP (pp) | ΔRank-1 (pp) | Registered dataset gate |
|---|---:|---:|---|
| RGBNT201 | +0.088272 | +0.239234 | FAIL: below +0.5 mAP |
| RGBNT100 | +0.032907 | +0.116618 | PASS within this panel |
| MSVR310 | +0.116056 | −0.338409 | FAIL: below +0.5 mAP and lower Rank-1 |

Whole-panel **FAIL** is supported by `logs/slot_competition_fp32_complete_intake_20261001/contract/EXPERIMENT_PLAN.md:28`, executed report source `logs/slot_competition_fp32_analysis_source_20261001.py:69`, and narrative `results/SLOT_COMPETITION_FP32_COMPLETE_2026-10-01.md:24`. No epoch/seed/metric mixing or gate relaxation was found. Identity bootstrap at `tools/analyze_correspondence_distances.py:62` describes fixed selected models, not training reproducibility.

**Found and corrected:** the original narrative/SUMMARY said slot statistics precede all role operators. CNN convolution actually runs before sampling at `modeling/trifusion/correspondence_context_identity.py:43`–46; Transformer/Mamba mixing and explicit bridges follow at lines 52–58. Direct reread confirms narrative line 28 now states this, and line 71 preserves an explicit erratum. Current local report-tool line 137 is corrected. Original executed source remains byte-identical at `logs/slot_competition_fp32_analysis_source_20261001.py`, SHA `2fbe37c00e83672e56e3321cf6f4405dd3d9890b34cecf90ff735b9b5eb825fe`; raw SUMMARY is unchanged. No numerical rerun was necessary.

Slot similarity still does not establish physical correspondence, final-role redundancy or unique causation. Same-checkpoint global/local outputs are decompositions, not separately trained controls (`modeling/trifusion/correspondence_context_identity.py:92`; narrative line 40).

The local tree is not a self-contained exact runtime: **79/213** bound source copies match bytes, **27** differ only by CRLF/LF, and **107** comparator/config/protocol files are absent. All authoritative remote bindings pass. This is a reproduction boundary, not evidence of wrong scores. Unrelated local source differences were preserved.

## F. Evaluation classification — PASS

- Retrieval and saved-array replay: **real_gt**, from actual identity/environment labels (`tools/run_correspondence_context_identity.py:207`).
- M0: engineering/numerical prerequisite with real training data, not retrieval quality.
- Slot statistics: unlabeled activation diagnostics, closest skill taxonomy **self_supervised_proxy**; not physical-part GT (`tools/diagnose_slot_competition_fp32_slots.py:31`).
- No synthetic prediction-derived reference, simulation-only benchmark or human-evaluation claim was found.

## Practical actions and limits

1. Carry the registered FAIL gate and single-seed/official-selection qualification into the new handoff. Do not promote tiny deltas or M0 into stable gain, novelty or SOTA.
2. Preserve the CNN-specific measurement-stage correction and the original source/SUMMARY erratum.
3. Describe M0 as cumulative eight-batch gradient support; retain precise source/checkpoint/protocol bindings.
4. Keep remote-only artifact and local-mirror limitations explicit. This audit requests no new training or neural replay.

Only the authorized host and existing Python environments were used. No training, neural forward/backward, GPU inference, installs, Git operations, queues/observers, source/config/canonical edits, or model/array/image downloads were performed by the reviewer.

The first CPU witness exited **1** because the upstream CLIP constructor additionally hard-codes `Module.to("cuda")` at `comparators/Signal-cd1b0a6/modeling/meta_arch.py:75`; CUDA was unavailable and no forward ran. The completed witness explicitly maps that demonstrated device call to CPU in-process and exits **0**. Both scripts and the failure are preserved in AUDIT.json. Tooling-only alias, output-truncation, command-length and missing-rg failures were recorded rather than counted as experiment evidence.

The corrected narrative is supported within this development-panel scope. Same-family semantic review remains provisional.

