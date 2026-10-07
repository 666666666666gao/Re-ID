# Independent additive native evidence — fixed nine-arm experiment

Prepared 2026-10-03. Source review and completed F3 integrity/claim closeout precede deployment. No new run has started at preparation.

Current resource amendment §41.792: latest user instruction permits only2026 physicalGPU0/1. Queue coordinator and direct workers now rejectGPU2/3 and cap active single-GPU jobs at2. This updates the resource scope only. Old FAIL/STOP and the all-nine M0 prerequisite remain unchanged; execution-policy clarification is pending, so this amendment does not start or authorize training.

## Question and fixed foundation

Does a separate image-native reader add useful identity evidence when the original 128-token semantic reader is preserved? Compare three independently trained arms on RGBNT201, RGBNT100 and MSVR310:

1. `global_only`: shared CLIP adaptation and author heads; no registered role operators.
2. `semantic`: exactly that shared foundation plus the existing static-token CNN → Transformer → Mamba roles.
3. `native`: the same semantic arm plus an image-native reader added before the CNN output norm and downstream bridges.

Use the pinned Signal author **training package**, not its trained ReID state or SIM/GAM/LAM. F1's controlled full50 author/current comparison improved the 201/100 absolute baseline substantially. F2/F3 do not support raw-feature scale as a universal rescue, so do not continue their coefficient, margin, seed or scale searches. The new global-only contains shared adapters and must be retrained; it is not F1's plain no-adapter baseline, the old 69.6415 denominator, or a warm-started Signal extension.

All arms start from the same public CLIP visual file, resized author positional embedding, freshly initialized camera and original author heads. No ReID checkpoint, external text, masks, teacher, extra dataset, predictor, meta-learning or new loss. Train the visual backbone rather than freezing it. The pinned author optimizer handles the **whole wrapper**, including shared adapters, roles and detail; original unused normalized heads stay frozen.

## Model intervention

Preserve the existing semantic sampler at its original 128 patch tokens. The native CNN is shared across the three modalities, 3→32→64→128 with three stride2 convolutions, giving 512 positions for either image orientation. It uses its own query/key/value projection in FP32 attention, guided by existing semantic context and anchor queries. Read 16 regions, project through one biasless zero-initialized exit, and add this evidence to the preserved semantic CNN regions. No other new exit is zero-initialized. Original Transformer/Mamba bridges, static global token and 1536-dimensional readout remain unchanged.

Native adds **159,296 trainable parameters / 14 tensors**. A positive result initially supports the added reader **and capacity**, not a capacity-independent causal explanation. A matched semantic-capacity control is required before attributing a gain uniquely to native detail. This experiment is not evidence that N2/N3 or all three proposed modules work.

## Training and evaluation contract

- Fixed seed42; every arm completes50 epochs, independently from initialization; no seed replacement.
- Reuse each dataset's pinned author batch/identity sampling, augmentation, optimizer weight decay and scheduler. RGBNT201 B64/K8, RGBNT100 B128/K16, MSVR310 B64/K4; workers4. Modal image augmentation is the pinned author's behavior, not the old shared-geometry current package.
- Raw corrected features feed the original author BN/CE/Triplet heads. RGBNT201 has one1536 head; vehicles have three512 heads and sum the same author losses.
- Preserve the author learning-rate rules, including visual5e-6 and MSVR base5e-6, bias×2 and classifier×100. New MSVR role/detail weights therefore start at5e-6 and their bias groups at1e-5; classifiers use5e-4. Do not silently increase rates after seeing results. Real M0 support is an engineering gate and does not prove these rates are sufficient for retrieval.
- AMP FP16 / initial scale256. Existing attention FP32 boundary remains. No swallowed AMP skip or nonfinite values.
- Deploy one L2-normalized1536 vector. Use complete fixed query/gallery and original camera/scene filters, including MSVR gallery-only distractors. No reranking or query–gallery joint inference.
- Evaluate every epoch. Choose one highest official mAP checkpoint, latest epoch on exact ties; report all metrics from that checkpoint. Run independent strict reload and official scoring once, with the existing `<1e-5` tolerance and author/independent scorer agreement.
- Save full model state, including changed visual, fresh camera, author BN buffers and native reader. M0 states are separate from formal fresh training.

## Required preflight before formal training

For every dataset, build all three arms and check all shared tensor/buffer values, author heads and full semantic/native original state. Read the actual full author training batch separately under matched initialization; images/labels/cameras/paths must match. With the zero native exit, semantic/native full raw features, deployed embeddings and logits must be exactly equal in the initial AMP eval forward. Global-only's shared global must also match; its corrected representation need not equal the role arm.

All **nine** M0s precede every formal arm. Each M0 performs eight actual production AMP optimizer updates; all trainable tensors must receive finite nonzero gradient cumulatively. Record optimizer ownership/groups and author BN mode/counts. For native, all14 tensors must show cumulative nonzero gradients and actual nonzero parameter changes, not just one output exit. Reload the full M0 state and compare real-record deployment output. These steps are engineering evidence, not formal retrieval results.

Retain actual per-step loss/LR and all50 epochs' batch metadata. At terminal reporting, compare actual complete label/camera/path order across the three arms, rather than inferring it from a seed. Initial input tensor equality is checked; complete augmented-image bytes are not archived, so do not claim full50 transformed-image byte equality.

## Execution and storage

Only `/data/gaob/Re-ID/Trifusion` on **2026**, physical GPU0/1; maximum two single-GPU jobs and assign only currently free cards. Do not start training on2025 or preempt other projects. Queue polling240 seconds. Use existing conda/data/public weights; no package installation or dataset transfer.

F3 actually stopped before its last evaluator because a fixed10GiB per-stage guard exceeded available disk. This new campaign budgets storage once: eighteen384MiB state files,600MiB distances and256MiB text, plus2GiB reserve. This is a conservative estimate based on retained actual F3 state sizes345–347MB and the largest full distance59MB; it does not alter the failed F3 guard or receipts. Check the whole budget before construction and retain2GiB per stage. No automatic deletion. If storage is insufficient, separately inspect eligible closed artifacts under existing retirement authorization. Preserve current dependencies, best weights and primary evidence.

## Success, failure and reporting

Primary comparison: `native − semantic` on each dataset. Advancement requires ΔmAP≥0.5percentage points **and** no Rank-1 decline in all three datasets. Secondary comparisons `semantic − global_only` and `native − global_only` quantify actual role contribution. The0.5gate is a development decision, not significance or SOTA.

Finish all predeclared arms even if one intermediate score is low; do not add scales/margins/coefficients/seeds or change heads based on consumed official scores. Engineering failure stops new launches and preserves every traceback; diagnose the actual failure before a separately attributable corrected attempt.

Terminal CPU report executes once after nine full50/strict receipts, recording paired AP, first-rank repairs/new errors, identity distributions, best-to-last degradation, cost and batch metadata parity. Identity bootstrap is fixed-model descriptive uncertainty, not full-process multi-seed significance. Original benchmark has already been consumed by prior development and selects epochs here. Single42 gains cannot establish robustness, unique role necessity, three-module success or SOTA. The broad project goal remains ACTIVE_UNMET.
