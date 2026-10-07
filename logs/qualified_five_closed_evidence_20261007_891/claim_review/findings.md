# Research Findings — qualified five objective endpoints

**C1=no, C2=no, C3=scoped yes, C4=no. Broad goal ACTIVE_UNMET.** The completed record closes this training-objective comparison. It does not complete the original regional three-dataset/SOTA goal.

The five seed-42 endpoints each completed fifty epochs: 250 epochs and 9,839 aligned saved steps, with five strict-evaluation receipts and all twelve registered full-query comparisons. RGBNT100 repair_keep remains absent after its third-role Q/K auxiliary-gradient M0 rejection. The original failure is retained; there is no sixth formal score.

## What failed and what remains positive

No primary comparison passes the predeclared >=0.5 mAP / nondecreasing Rank-1 line (0/5), and no comparison in the full twelve-pair panel passes (0/12). This is a phase decision, not a null-hypothesis significance test.

- RGBNT201 MD: +0.094523 mAP/+0.358852 R1 versus semantic; +0.172721/+0.239234 versus global-only. These are small positive selected-point observations.
- RGBNT201 repair: +0.277213/+0.837321 versus semantic, with 9 repairs and 2 new errors; +0.355411/+0.717703 versus global-only, with 9 repairs and 3 new errors. It improves over MD by +0.182690 mAP but still misses the line and does not preserve every baseline ranking.
- MSVR310 MD: +0.011952/+0.169205 versus semantic; +0.012048/0 versus global-only. Repair: +0.002416/0 versus semantic and +0.002512/-0.169205 versus global-only; relative to MD it loses 0.009537 mAP and one Rank-1 success.
- RGBNT100 MD: -0.179661/-0.233236 versus semantic, with 17 repairs/21 new errors; -0.623141/-0.991254 versus global-only, with 16 repairs/33 new errors. Its selected-to-E50 mAP drop is 10.233018; this is not a ten-point improvement or an identified objective-specific causal effect.

The two strictly positive fixed-model identity-macro bootstrap intervals (RGBNT201 repair/global; MSVR310 repair/semantic) are retained. The other intervals and all ranks/query changes remain visible in the full report and exact JSON. Identity resampling of selected models does not supply training-seed variance, selection-safe inference or practical three-dataset success.

## Mechanism observations and unresolved causes

All MD steps have positive auxiliary scalar loss. Repair loss is positive on 1,657/2,649 RGBNT201 steps and all 706 MSVR310 steps. RGBNT201 repair-cell support occurs in only 189 steps; MSVR310 has support in 640. Independent protocol-aware batch counting verifies the legal-query exposure ratios 33.866553% and 48.043555%. These measured support differences motivate diagnosis but do not establish a universal cause of failure.

The rejected RGBNT100 repair M0 has legal relations and both active cells in every one of its eight batches, with all 1,024 repeated query exposures multiply supported. Third-role isolated query/key sums remain zero with unused=False. No-evaluable-relations is contradicted for those batches; numerical cause and hypothetical fifty-epoch efficacy remain unknown.

Actual repair uses an absolute positive 0.1 fused margin for legal relations whose detached current-global margin is <=0, with keep on global-correct relations. It is not the earlier relative-margin expression. Training loss activity and correction/global norm ratios are not per-role gradient evidence over full training and are not held-out component retrieval scores. The five official saved-distance payloads contain fused distances, not a new g/c/f decomposition. Independent global-only endpoints do not establish each candidate checkpoint's own-global quality.

## One bounded next hypothesis

H1: the existing corrections are active but insufficiently selective for own-global retrieval errors, allowing new errors to offset rescues. Test only the five fixed selected checkpoints' own-global g, correction c and unchanged fused f using identical complete protocols, checkpoint SHA binding, all-query AP/CMC and within-checkpoint repair/harm plus identity deltas. Check own-global weakness rather than assuming equality or degradation. No fitting, gain/threshold change, new checkpoint selection, seeds or retraining. Correction-only scores are diagnostic. This is a proposed post-selection diagnostic, **NOT_RUN here**, not a new success claim or proof that regional supervision will solve the gap.

## Future constraints and routing

Keep the sealed near-capacity/native/reconstruction/slotuniform/SIM, independent-head and detach/joint-L2 controls closed. Do not rescue these failed trials through numerical precision, losses, gains, learning rates, margins, seeds or thresholds. No full regional P1/P2/P3, formula novelty, full MDReID, general robustness, ten-point gain or SOTA claim is supported. A documented negative result must not redefine the broad goal as complete.

Route C1/C2/C4 to the recorded negative findings; retain C3 as a scoped completion fact. No source/project AGENTS/wiki edits, training, SSH, GPU query or publication were performed. All output is review-local.

## Assurance

[INTEGRITY: WARN] The completed integrity audit was inspected before finalization: warn, 0 blocking / 3 warnings. Its exact attribution remains same-family/provisional, requested gpt-6-astra/max, runtime_attestation=unavailable. This reviewer is also same-family/provisional, requested gpt-6-astra/max with runtime_attestation=UNATTESTED. No semantic acceptance is inferred from deterministic local byte/text checks. Canonical evidence_check.py remains UNRESOLVED; remote binary evidence is transcript-backed and was not independently replayed.

Full scoped results and source citations: [CLAIMS_FROM_RESULTS.md](CLAIMS_FROM_RESULTS.md); exact structured verdict/data: [CLAIMS_FROM_RESULTS.json](CLAIMS_FROM_RESULTS.json). The full input SHA manifest and forensic trace are in [.aris/traces/result-to-claim/2026-10-07_five891](.aris/traces/result-to-claim/2026-10-07_five891).
