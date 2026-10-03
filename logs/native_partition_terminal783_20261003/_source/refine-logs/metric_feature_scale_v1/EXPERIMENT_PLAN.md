# F3 proposal: isolate metric input scale with a fresh paired control

Status: PLANNED_SOURCE_REVIEW_PENDING_NOT_LAUNCHED. F2 fresh audit/claim and publication closed at section41.754, commitdd5b536b59e9d9ad0b6dd9de10a1a1eaf0a55505. This is a foundation diagnostic, not a new research module or a promised performance improvement.

## Question and evidence

F1 showed a substantial author/current package gap on RGBNT201 and RGBNT100. F2 changed both the BN/CE and margin-0.3 Triplet input from normalized to raw. Its completed six-arm result improves MSVR310 only and reduces mAP on the other two datasets. It cannot attribute that result to either input separately. The next finite diagnostic changes only the Triplet input under normalized BN/CE, following the fresh claim review's proposed bounded diagnostic. It does not fully identify BN/CE effects or interactions.

Do not tune a margin, coefficient, seed or dataset-specific recipe against these consumed official results. No roles, CNN-detail branch, adapters, SIM, AlignM, external data, text, reranking or test-time updates are added.

## Matrix

| BN/CE input | Triplet input | Evidence source |
|---|---|---|
| normalized | normalized | proposed fresh paired training control |
| normalized | raw | proposed fresh training arm |
| normalized | normalized | completed immutable F2 result, context only |
| raw | raw | completed immutable F2 result, context only |

Run one fresh control/hybrid pair on each of RGBNT201, RGBNT100 and MSVR310: six new formal runs. The old F2 results remain historical context, not a substitute for a fresh paired control or new seeds. Verify identical fresh initial states, model capacity, protocol and batch-order bytes within each fresh pair. Also bind the prior F2 initialization/batch records for reproducibility context. All source bytes actually used by both campaigns remain bound separately. Do not rewrite the original F2 seal.

If actual initial-state or batch-order matching fails, preserve the failure and resolve it before scientific interpretation. No fabricated parity or silent substitute baseline.

## Fixed execution contract

- Only server 2026 physical GPUs 0–3; at most four single-GPU processes. No server 2025 training, preemption or repeated launch on observation timeout.
- Reuse the validated tri_reid conda environment and installed datasets/public CLIP. No environment rebuild.
- Fresh public CLIP visual initialization, fresh camera and one 1536-dimensional BN/classification head; all trainable parameters covered by the optimizer.
- Current package remains fixed: seed42, B64/K8, shared geometric augmentation, AdamW, decay1e-4, nominal visual LR5e-6/other LR3.5e-4, five-epoch warmup/cosine, CE smoothing0.1, existing batch-hard Triplet margin0.3, AMP initial scale256, four workers.
- BN/CE receives its designated feature, Triplet receives its separately designated feature. The raw CLIP output and its L2-normalized version are computed from the same forward; neither reference is detached. Deployment remains one L2-normalized 1536-dimensional embedding.
- Six M0 gates with eight optimizer updates each, strict save/reload and all trainable gradient support. M0 states are discarded; every formal arm starts fresh and completes all50 epochs.
- Same earliest/latest tie handling and mAP-best rule as the sealed F2 code; all metrics follow the one chosen checkpoint. Strict full official evaluation retains every query/gallery item, MSVR gallery-only identities and correct camera/time filtering.
- Persist original batch-order/step/epoch histories, full states and selected distances. One report after all six formal runs and strict checks; no score-based early cancellation or retry.

## Interpretation fixed before execution

Primary outcome: hybrid minus its fresh normalized/normalized control in mAP and Rank-1 within each dataset. Every pair uses the fixed ≥0.5 mAP/no R1 decrease development criterion. Old F2 results remain context and cannot supply an alternative easier gate. The criterion is not statistical significance or a general claim.

Secondary outcomes: identity-macro AP, first-rank repairs/new errors, actual Triplet support, input norms, complete50-epoch degradation, execution time and memory. Fixed-model identity bootstrap does not estimate training-seed uncertainty. Official-set checkpoint and method selection remain disclosed.

The primary contrast estimates Triplet-input effects conditional on normalized BN/CE. It does not identify every pathway interaction, explain the entire F1 author/current gap, establish SOTA, validate the old roles or count as one of three novel modules. After this one bounded panel, stop input-scale variants and select the credible matched foundation for the independently read semantic/detail method; do not begin a coefficient or seed search. If no valid increment appears, close the scale-only improvement branch.

## Implementation boundary

Add a separate trainer/queue/check/report entry. Preserve original F1/F2 source and artifacts. Reuse their actual loader, initializer, optimizer, evaluator and full-state serialization. Add only the separate CE/metric feature selection and norm diagnostics required by the question. No fallback, speculative exceptions, generic compatibility layer or unrelated refactor.

Before launch: source review; exact source sync; fresh free-GPU/disk proof; registered immutable plan and source manifest. Estimated six-arm duration derives from F2 actual wall intervals at the available concurrency, with final reload/report excluded from training ETA; publish that estimate when launching. Draft creation is preparation only; it does not claim gate passage or actual launch.
