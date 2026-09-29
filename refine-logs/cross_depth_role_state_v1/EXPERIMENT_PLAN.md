# Cross-depth role evidence experiment

Date: 2026-09-29. Status: implementation/preflight; no formal results.

The complete context/local panel is retained as a negative result. The query mAP effects averaged over the none/local supervision conditions are approximately +0.0210/+0.0023/+0.0214 points on RGBNT201/RGBNT100/MSVR310, respectively. Local identity supervision improves the standalone local output but does not provide a consistent fused gain. This plan changes evidence formation while retaining the existing supervised retrieval objective and regional readout.

## Claim map and decisive comparisons

One hypothesis: processing evidence at each visual depth, and carrying role evidence between these processing steps, may provide useful identity evidence beyond mixing the layers before processing them. This is a hypothesis, not a verified novel contribution or a claim of ten-point improvement.

| Condition | Processing | Purpose |
|---|---|---|
| mixed_once | Uniformly mix the three layer snapshots, then run the roles once | Direct structural control |
| depth_mean | Run the same role operators independently at each depth, then average outputs | Separate earlier evidence processing and extra computation |
| depth_recurrent | Process depths 4→8→12; add each preceding role output to that role's next input | Test persistent evidence against the three-call independent control |

The persistent state is downstream role evidence across three CLIP snapshots. It is not three separate CLIP trunks or a role state threaded through every CLIP block. CNN/Transformer/Mamba processing and messages retain the existing order. Common addresses are predicted once from the uniformly mixed Transformer-role CLS. The same query, addresses, trainable parameter set and 1536D readout are used in all three conditions. The mean and recurrent controls both call the role operators three times; their compute need not be identical because autograd graphs differ. Actual training times will be reported.

The CNN state is added to sampled anchor evidence before its output normalization, after this depth's spatial convolution. It is not a recurrent convolutional grid. Transformer and Mamba states enter their corresponding current-depth sequence processing. Frozen Signal tensors stay exact, while trainable M1 can change the adapted shared_global output.

## Fixed protocol

Three conditions × RGBNT201/RGBNT100/MSVR310 = nine complete endpoints; one preregistered development seed42. M1/M2 on, M3 off. Context affects candidate weights only. No auxiliary identity head, teacher, new loss, learning-rate change or correction-scale scan. Layer logits are frozen uniformly in all three controls; this differs from the historical context panel, so all three controls are newly trained.

The saved pure ReID baseline for each dataset remains frozen. CLIP data initialization, training sampling, AdamW, augmentation, warmup and 50-epoch schedule reuse the current context entry. Each endpoint starts from the same dataset-specific initialized tensor state. Best means the highest official fused mAP from the full 50-epoch history, with latest epoch resolving exact ties; every other reported metric comes from that one checkpoint. Independent reload checks the depth-mode binding as well as tensor keys.

The reused entry records the dormant auxiliary coefficient as1.0, while auxiliary_target=none disables that head and requires every logged auxiliary loss to be exactly0. No auxiliary objective is active; the original scalar auditor remains valid.

Official labels and the complete gallery determine evaluation. RGBNT201/RGBNT100 use the official camera filter. MSVR310 excludes same-identity/same-time-period pairs; the scene field is a time-period label. No gallery reduction, reranking, test-label learning or per-query threshold fitting.

## Evidence and interpretation

Every endpoint requires eight real production M0 batches, finite gradients and a nonzero-gradient observation for every trainable tensor across those eight updates, unchanged frozen Signal tensors and checkpoint reload agreement within1e-5 before full training. This does not require every tensor to have nonzero gradient in every batch or bitwise reload equality. The final collector requires all 50 epochs, exact best selection, checkpoint/distance/source hashes, complete metadata, all three fused/global/local CPU recomputations and full task logs. A CPU synthetic operator check is explicitly a structural check, not ReID performance or production M0.

The primary comparison is recurrent versus depth_mean on each dataset. Consistent positive fused mAP and Rank-1 differences, supported by fewer harmful flips and identity-level gains, are required before claiming reliable persistent-role benefit. A one-seed official-best comparison cannot establish unbiased generalization or training variance. If depth_mean explains the gain, the claim is evidence processing/extra computation rather than persistence. If all three remain close to global-only, do not claim the three roles became necessary. Negative endpoints stay in the table.

## Execution and cost

Finish/archive the preceding 15/15 panel first, source-review the independent entry/collector, then execute all nine M0→full50→reload/evaluate chains on free GPUs. Four GPU slots; durable queue polls every240 seconds. A failed endpoint stops launching pending work and preserves receipts; there is no retry or seed replacement. Prior single-call jobs took roughly20–110 minutes depending on dataset; three-call role jobs may take longer. Budget approximately10–18 GPU-hours, with estimates replaced by measured epoch durations. No environment rebuild or dependency change is planned.

The existing frozen runtime files and all old result receipts remain unchanged. New checkpoints use a distinct schema with an explicit depth_mode because all three modes share tensor keys and an ordinary strict tensor load alone would not distinguish their computations.
