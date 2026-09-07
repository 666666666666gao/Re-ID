# MSVR310 source-style V1 R2: complete engineering verification

Status: **PASS_ENGINEERING_ONLY**, independent CPU **PASS_COMPLETE_MSVR_STYLE_M0**, local full text **PASS_LOCAL_COMPLETE_MSVR_STYLE_M0_TEXT**. This is not an unseen-identity retrieval result or scientific promotion.

## Actual execution and full scope
R2 wrapper167448, initial source commit02cc09e; config848254aab6dc02230a7fd33be0d60fc2560c99fc77cbd320cf0ef2efd655539d. M0PID167458 ran18:20:03-18:29:11 and exited0. CPU168446 ran18:29:11-18:29:21 and exited0. Q1PID168456 started18:29:21, actual stage HEAD893be37 (only documentation/report-tool publication since02cc09e; all registered training hashes unchanged). Both wrapper and Q1 were live by /proc command checks at18:31:25. No duplicate process or reused M0 training state.

All3folds x2endpoints x8capacityupdates + freshfold0 x2endpoints x100fixedoverfit =248actual updates. No heldout/gallery image forwards, no official/dev images. Source Signal equality checked against standalone fixed B0; enabled/disabled/inactive anchor/reference field tests, all5output strict checkpoint reload equality completed for all6capacity endpoints.

| Run | Updates | Nonzero gradient tensors | Active style steps | Peak reserved MiB | Training seconds |
|---|---:|---:|---:|---:|---:|
| m0_fold_0_control | 8 | 203 | 0 | 5954 | 14.7707 |
| m0_fold_0_source_style | 8 | 203 | 2 | 6228 | 12.7259 |
| m0_fold_1_control | 8 | 203 | 0 | 6228 | 12.3791 |
| m0_fold_1_source_style | 8 | 203 | 2 | 6240 | 12.2861 |
| m0_fold_2_control | 8 | 203 | 0 | 6240 | 12.4921 |
| m0_fold_2_source_style | 8 | 203 | 5 | 6240 | 13.2754 |
| m0_overfit_control | 100 | 203 | 0 | 6132 | 134.9903 |
| m0_overfit_source_style | 100 | 203 | 100 | 6132 | 135.7809 |

Control/source-style fixed100step excess-loss ratios are0.0007021239307839527 and0.0007105026765247138, both below pre-registered0.1. The candidate uses the same fixed augmented raw batch and fixed force-active step0 statistic plan throughout all100updates. Both endpoints share exact raw pixel hashes and plans; only the style effect differs.

All8runs:203/203trainable tensors receive actual nonzero gradients, nooverflow, frozen fullstate and Signal unchanged, role states updated. Folds0/1 have99,065,869total/8,076,300trainable parameters; fold2 has99,095,053/8,102,412 because it has104 instead of103source identities. No trainable parameter is added by the style backbone.

Six compact M0 checkpoints retain every non-baseline state entry and exact B0 alias binding. Every checkpoint fullmodel SHA is independently reconstructed on CPU; allfive outputs match after strict GPU reload. Fold0checkpoint files are32,734,799bytes each. Required B0 weights and all engineering finals retained; models were not downloaded.

## Complete evidence checks
RemoteCPU summary SHA5d9d71d05fd4dcb2c2a5640ee48d6d58a11cd9ca5b375c08cf8edcdcf506f332; remoteCPU receipt SHA058f726e5c62f201ccd24fdd057e15a980b27efa9ba43fae39822bc3b4bdb931. Every248saved training step, exposure order, active plan, finite loss component, epoch mean, paired pixel SHA and state binding verified. OriginalCPU check reconstructed actual checkpoint tensors; local text verification reads no model.

29textfiles,2,059,528bytes received with complete byte/SHA match. Independent local verifier recomputed all248steps and the analytic label-smoothing floor. Maximum saved14component loss reassembly difference is4.2219957e-7: diagnostic double arithmetic, not a claim that original AMP intermediate dtypes were saved. No new loss tolerance or scientific gate was introduced.

## Preserved first-attempt failure and next phase
Original628cce0 stoppedatT0 with0model/0update/0heldout work:31Float64 coefficient least-bit differences, maximum2.220446e-16, while every appliedFloat32coefficient and all donor/activation/sample fields were identical. R2 binds actual server-generated plans; exact equality checks and the complete original experiment contract remain. See R2_RUNTIME_PLAN_BINDING.md and preserved original logs.

Q1 now runs fresh matched control/candidate in all3folds,20epochs each:1560updates,2064heldoutrecord forwards,600queries and full1032fold-localgalleryrecords per endpoint. Original scene filtering and five-output Signal exactness retained. Candidate must pass both the original five vehicle-baseline conditions and five paired-style conditions. Fixed complete terminal only; no intermediate ranking for selection.

At18:31:47 diskfree9,481,236,480bytes. First5formalepochs suggest about20seconds/epoch; full6end training+reload/evaluation expected roughly19:12-19:20+08, estimate only. Next read checks original Q1 process and first endpoint full-gallery receipt; no new model direction or official access before complete Q1. Three-dataset baseline/resource-qualifiedSOTA Goal remains active and unmet.
