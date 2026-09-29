# First four M0 runtime artifact review

Date: 2026-09-29. Evidence snapshot: **2026-09-29 21:19:26.383950 +08:00**.

**Overall verdict: WARN. Recorded M0 contract and artifact-consistency checks pass for four endpoints. No concrete contradiction or current blocker was found in the supplied evidence.**

This is a continuation by the same native Codex reviewer that wrote REVIEW_ENTRY_658_20260929, not a fresh reviewer or an independent second review. The original requested model/reasoning attribution remains `gpt-6-astra` / `max`; actual model/effort attestation is not exposed by this runtime. `review_independence: same-family`; `acceptance_status: provisional`.

The reviewer used the supplied files as evidence, without using executor narration as proof. No SSH, GPU work, retraining or checkpoint replay was performed. The earlier entry review was not modified; its MD and JSON SHA-256 values remain `73336c4728718fd934d63f8cbf707604c177589de2d485ba9e285a9b9d2d329b` and `3862d639800eabcd074c6aea94559ec6609d642ac96b72365a5467352072bfb1`.

## Evidence actually established

All four archived training receipts have status M0_PASS, one history row at epoch 1, eight steps, no selected best epoch, and no formal checkpoint. All 32 JSONL rows have batch indices 0–7 within their respective run, finite ID/triplet/total/auxiliary scalars, auxiliary_target=none and auxiliary_id=0. The eight archived text-file hashes match the archive index exactly.

| Dataset / mode | Steps | Reported gradient tensors | Trainable parameter scalars | Triplet nonzero steps | Max loss reconstruction error |
|---|---:|---:|---:|---:|---:|
| RGBNT201 / mixed_once | 8 | 118/118 | 2,617,345 | 7 | 2.337620e-7 |
| RGBNT100 / mixed_once | 8 | 118/118 | 2,431,489 | 0 | 0 |
| MSVR310 / mixed_once | 8 | 118/118 | 2,592,769 | 8 | 2.235174e-7 |
| RGBNT201 / depth_mean | 8 | 118/118 | 2,617,345 | 7 | 2.207235e-7 |

Evidence: each `logs/cross_depth_m0_<mode>_<dataset>_658_20260929/training.json:42–59` and corresponding `training_steps.jsonl:1–8`; the four exact directory names are listed in `logs/cross_depth_m0_archive_658_20260929.json:5–47`. Mean total losses reconstructed from the logs equal each receipt exactly: 5.129286706447601, 3.8579018115997314, 5.222078323364258 and 5.129377961158752, respectively.

The 118 count is a count of trainable tensors receiving a nonzero-gradient observation across the eight updates, not 118 scalar parameters or a claim of nonzero gradient in every batch. The separate initializer counts above are scalar parameter counts. RGBNT100's zero triplet scalars do not contradict aggregate model gradient coverage: the ID objective is active. These task scalars do not establish gradient share, causal efficacy or retrieval performance.

All four receipts report `frozen_signal_unchanged=true` and `reload_max_abs_difference=0.0` (`training.json:50–55`). Their matching values and probe hashes are also present in the progress snapshot (`logs/cross_depth_progress_659_20260929.json:97–102,240–245,383–388,526–531`). These are native-runner receipts tied to the reviewed production entry, not independent tensor/gradient replay by this reviewer.

## Phase ordering and exit evidence

The sync receipt is dated 21:18:36.699557 and records “NOT_LAUNCHED”; the launch receipt at 21:18:38.471873 records “LAUNCHED_NOT_YET_VERIFIED.” Those immutable earlier states are compatible with the later progress snapshot. The launch controller PID and complete command match the progress record. Sync and launch record the same commit, `f7114ac02c1c33c88ec48ca7a17a766fc96535f2`.

| Dataset / mode | M0 PID | Recorded M0 exit | M0 job completion | Full-training start | Training PID |
|---|---:|---:|---|---|---:|
| RGBNT201 / mixed_once | 3133080 | 0 | 21:19:22.180876 | 21:19:22.215416 | 3135407 |
| RGBNT100 / mixed_once | 3133081 | 0 | 21:19:21.136998 | 21:19:21.172574 | 3135278 |
| MSVR310 / mixed_once | 3133083 | 0 | 21:19:21.835160 | 21:19:21.870439 | 3135343 |
| RGBNT201 / depth_mean | 3133082 | 0 | 21:19:20.901341 | 21:19:20.936889 | 3135277 |

All times above are +08:00 on 2026-09-29. Evidence: `cross_depth_progress_659_20260929.json:60–63,105–148,203–206,248–291,346–349,391–434,489–492,534–577`.

For every case, the receipt's completion timestamp precedes the M0 subprocess completion; the next train start is 0.034540–0.035576 seconds after that subprocess completion. Timestamp comparisons preserved fractional seconds. Each child has recorded phases [m0 COMPLETE exit0, train RUNNING], separate M0/full output directories, and the same dataset/mode/baseline/protocol/seed/epoch contract. No evaluate phase is present.

The supplied snapshot records four running endpoint chains and five pending chains, with zero completed endpoints. It supports that the queue recorded four formal-training child launches after successful M0 exits. It does **not** provide a training exit code, completed formal epoch or proof that each training process is still live at this review's later time. The snapshot's low GPU-memory/startup readings are not proof of completed GPU training updates.

## Mode, initialization, parameter and source bindings

Every receipt and both recorded subprocess commands agree on seed42, epochs50, width128, M1/M2 enabled, M3 disabled, query=context, auxiliary=none and the explicit depth mode. Initializers agree on architecture cross_depth_role_state_v1, fused width1536, context width512, frozen depth logits, learning rate0.00035 and weight decay0.0001. The dormant auxiliary coefficient is1.0 while every logged auxiliary scalar is0. No hidden auxiliary objective is evidenced.

The RGBNT201 mixed_once and depth_mean initial state hashes are identical:
`595e5d6ebab9eb4a5d5cf930b4328f22200d038a9f0c0e4ddf97756be9989a92`.
Their scalar parameter counts are also identical. See the two exact receipts:
- `logs/cross_depth_m0_mixed_once_RGBNT201_658_20260929/training.json:25–26`;
- `logs/cross_depth_m0_depth_mean_RGBNT201_658_20260929/training.json:25–26`.

This establishes a matching **recorded** initialization for that one within-dataset pair. It does not establish all three modes' production initialization parity on every dataset. The other two recorded hashes are RGBNT100 `c2fbd7a6212a3eb2cf7a888d21b94196cae7a9371057650a267e818df5694066` and MSVR310 `60972e9d789c62c18a8f3c0deabdf27fd2138f12f7a6d7d8636ae3809f8d9548`. Cross-dataset state/count equality is not the registered control.

The local manifest's whole-file SHA is `67e383e88f5151d12cbf3eae38111df42f7e1279ba7118b2a6f2cc4ca6044b14`, exactly matching progress line7. It has205 source/input hash entries, nine ordered jobs and the registered fixed contract (`cross_depth_manifest_658_20260929.json:2–63,64–269,271–284`). The progress rows preserve that job order. Its predecessor accepted-matrix hash matches both the launch command and the earlier review's audited archived matrix hash.

The four newly introduced source hashes match across sync, launch, manifest and the immutable earlier entry review. Each M0 receipt's seven source-binding fields matches the manifest. Each protocol SHA matches the dataset-specific manifest entry; each baseline SHA/path matches the recorded commands. These checks establish consistent recorded bindings.

There is one bounded byte-level caveat among the19 runtime files shared with the earlier entry review:
- 18 hashes match the manifest byte-for-byte.
- `modeling/trifusion/experts/mamba.py` retains the exact earlier-reviewed local hash `8b9cb420c42e4d70f8de7e4608637c81c505b97ca54228175462a7e92fdfcc83`, while the manifest records `c516c7ad937e5eee6a4ed1e3ec33c2afe3522b751d296bd2e4910e4f27a20ee5`.
- A read-only CRLF→LF transformation of those local bytes hashes exactly to the manifest value. Thus line-ending equivalence is demonstrated; raw-byte equality is not. No source or manifest was normalized or changed.

The remaining186 manifest entries were not independently matched to remote bytes in this artifact-only review. The sync receipt's aggregate “14 artifact checks / 23 old science unchanged” counts are recorded assertions without a per-file replay in the supplied sync JSON; they are not promoted to independent verification of all205 entries.

## A–F follow-up audit

| Check | Status | Finding |
|---|---|---|
| A. GT/data provenance | WARN | Dataset-specific protocol SHA and baseline/entry bindings agree. The archive contains no batch image/identity traces or raw dataset records, so the actual sample content and B64/K8 membership cannot be independently replayed from these texts. |
| B. Score normalization | PASS in this scope | There are no retrieval scores to normalize. All32 total losses reconstruct as ID+triplet+auxiliary within2.34e-7, with zero auxiliary. No performance percentage is derived from these losses. |
| C. File existence and numerical consistency | PASS | All13 requested runtime artifacts exist, including four receipt/step-log pairs. Eight text-file SHAs, manifest SHA, all32 scalar rows, all four means and snapshot/receipt copies agree. Probe binaries are explicitly unavailable locally. |
| D. Claimed execution | PASS for recorded M0 progression; WARN for replay | Four M0 exit codes0 are supplied by the snapshot and linked to M0_PASS texts. Training is recorded RUNNING without terminal exit or evaluation evidence. No successful metric-function execution or formal completion is invented. |
| E. Scope | WARN | Four of nine M0 runs are archived: mixed_once on all datasets and depth_mean on RGBNT201. The two other depth_mean and all three depth_recurrent endpoints are pending in this snapshot. No formal retrieval result exists. |
| F. Evaluation type | WARN / not yet evaluated | These are supervised train-only engineering M0 records, intended for the registered real_gt retrieval pipeline. They are not a retrieval evaluation, synthetic proxy performance, or evidence for persistent-role benefit. |

## Unavailable replay and claim boundary

None of the four `m0_reload_probe.pth` binaries is present in the supplied local archive. Their hash strings agree across receipts, but their actual bytes, checkpoint schema/mode/tensor contents, strict reload behavior and recomputed outputs cannot be rehashed or replayed here. Raw gradient tensors/names, optimizer state, AMP-scale history, actual model objects, compiled Mamba kernels, dataset batches and before/after frozen tensor snapshots are also unavailable. Nonzero/finite-gradient coverage, frozen-state equality and exact recorded reload0 therefore remain **receipted M0 facts under the bound reviewed runner**, rather than independently replayed tensor proofs. Finite logged losses are independently checked.

The two new conclusions supported by this follow-up are limited: the first four archived M0 receipts are internally consistent with the registered engineering gates, and the snapshot records correctly ordered launches of their full50 children. No recurrent-mode runtime behavior, persistent-state gradient measurement, all-nine initialization match, completed50-epoch endpoint, retrieval gain, harmful-flip improvement or generalization claim is established. Official-best and single-seed limitations remain ordinary interpretation limits, not permission blockers.

No retraining, GPU replay or repair is requested by this review. Preserve the supplied evidence and obtain the remaining scheduled endpoint receipts and final collector outputs before making the corresponding completion or performance claims. This report writes a new snapshot-specific conclusion and leaves the earlier source review immutable.

