# Global identity tokens inside role interaction

Registered 2026-10-01 after the complete FP32 allocation panel; no results exist for this design. Predecessor accepted matrix SHA: `e3433db691dbb1070595386c542f27782cd88b45f5f6faa8660b246e7c2fd4b8`. The predecessor's registered gate is FAIL and remains sealed.

## Hypothesis and distinction

Full-Patch memory, slot competition, cross-depth processing and added local identity supervision have not produced reliable role increments. Those comparisons are complete and are not repeated. One new intervention tests whether an identity-conditioned token participating in regional Transformer attention can make local processing useful. The prior context query changes attention weights only; it never supplies global values to role tokens. This proposal changes that boundary explicitly. It is different from old V8 SIM-to-block8-CLS feedback: here the adapted per-modal global conditions a 17-token regional Transformer, and the added token is removed before region readout.

Global/local interaction has prior art, including DSGM's global-plus-regional patch attention (`refine-logs/slot_competition_roles_v1/reference_source_check_682_20261001/SOURCE_BINDINGS.json` and `DSGM_SOURCE_MANIFEST.json`; source `modeling/fusion_part/Mask_patch.py` in the pinned author tree). This is a project hypothesis, not a claim of first global conditioning. No author text, SAM2 mask or code is copied. A direct-global control must distinguish interaction from simply adding another projection of the strong global.

## Matched three conditions

All conditions use the same 512-to-128 bias-free global projection, initialized to zero; the same trainable tensor set, full initial tensor state, 16 Patch-memory slots, 17-token Transformer, 1536D regional readout and complete training contract.

| Condition | Extra Transformer token | Extra final correction | Question |
|---|---|---|---|
| static | Projection of a fixed unit vector | None | Same-structure control |
| token | Projection of each modality's normalized adapted global | None | Does identity-conditioned role interaction help? |
| direct | Same fixed token as static | Projected per-modal global repeated across regions, through the existing shared readout | Is any gain just another global projection? |

The extra token is discarded from Transformer outputs before CNN-to-Transformer-to-Mamba regional readout. Mamba receives the 16 Transformer region outputs in the unchanged 48-token spectral/spatial order. The static projection is trainable but its input is fixed, so its effective input variability differs from token/direct; identical parameter counts alone are not identical effective capacity. Direct shares its projection between fixed-token interaction and final correction and shares the existing readout with role evidence. It is a specific bypass control, not an exact computational equivalence to token attention.

All three start with exactly equal outputs because the new projection is zero. All use new fresh training; historical 16-token endpoints are not substituted for the 17-token static control. Query context still controls full-Patch weights in all three. Attention normalization is fixed to independent, with the attention subgraph FP32 in all conditions. There is no promotion of the failed competitive arm.

The saved `joint_local` evaluator field is retained for the existing scoring path but means normalized total correction. It is global-conditioned in token mode and explicitly includes projected global in direct mode. It is not proof of pure local identity information. All metrics are labeled accordingly.

## Fixed full training

Three conditions times RGBNT201, RGBNT100 and MSVR310 = nine endpoints. Seed42; saved pure ReID initializer frozen; M1 enabled at blocks4/8/12; M2 role operators on; M3 and auxiliary ID off. Existing loader, augmentation, AdamW, learning rate, warmup, full50 schedule, fused CE(label smoothing0.1) + fused Triplet(margin0.3), camera/time filtering and full distractor galleries are unchanged. The original upstream ReID stage is additional cost.

First run the structural CPU fixture, explicitly using a linear Mamba stub and no ReID performance claim. Each real endpoint then needs eight production-Mamba M0 batches with finite loss/gradients, cumulative nonzero-gradient support for every trainable tensor, unchanged frozen upstream state and strict reload agreement within1e-5. Only then train that endpoint from fresh initialization for all50 epochs, select one official fused-mAP-best (last exact tie), strictly reload and compute full-gallery metrics and CPU ranking verification. M0 is not performance acceptance.

All nine paired endpoints are required. Failure preserves actual exits and logs and stops pending launch; no automatic retry, alternate seed, loss coefficient or gate change. Existing `tri_reid` environment is reused without installs. Actual free cards only, no preemption/reset/sudo. A durable queue uses240-second scheduling checks; status is collected at estimated milestones. Approximate budget8-12 GPU-hours plus M0/evaluation, replaced by measured times; hardware/concurrency may change wall time.

## Registered decision and scope

To advance the interaction design, token must exceed both static and direct in mAP on all three datasets, have nondecreasing Rank-1 in each comparison, and gain at least0.5 mAP over each control on RGBNT201 and MSVR310. No gate-triggered new seeds/global-only are registered here. If direct explains the gain, the supported interpretation is additional global projection; if neither improves, retain the full negative result and choose a new hypothesis from complete diagnostics.

Record formal mAP/CMC, full repairs/new errors, identity-equal AP, correction/global/fused decompositions, all50 trajectories, trainable parameters, actual train/eval time and frozen-source/checkpoint hashes. A one-seed official-best development comparison cannot establish untouched-test significance, training stability, three-role necessity, novelty or SOTA. The full three-dataset goal remains unmet.

Status: PREPARED_NOT_RUN. No new M0, formal epoch or evaluation has run. Source review and actual CPU fixture must finish before deployment.
