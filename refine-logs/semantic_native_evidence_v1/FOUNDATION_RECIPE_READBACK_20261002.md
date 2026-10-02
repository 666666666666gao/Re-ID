# Foundation recipe: pinned code readback

This is an offline source comparison, not a trained baseline result. All26 inspected source files match the active EV1 manifest243. Nothing in the active training source or sealed EV1 plan was edited.

| Variable | Signal cd1b0a6 code | Current clean / EV1 recipe |
|---|---|---|
| Heads / feature geometry | USE_A=False, USE_B=False code path: RGBNT201 DIRECT=1 has one raw concatenated1536 BN/classifier; RGBNT100/MSVR310 DIRECT=0 have three separate raw512 BN/classifiers; evaluation concatenates raw globals. | One1536 BN/classifier on L2-normalized fused embedding (global-only also normalized). |
| Loss | NO_MARGIN=True -> SoftMarginLoss on batch-hard sqrt Euclidean distances; normalize_feature defaults False. Each head contributes ID weight0.25 plus Triplet weight1; processor sums heads. | CE weight1, label_smoothing0.1 plus ReLU batch-hard Euclidean Triplet margin0.3 on normalized fused. Diagonal positives excluded. |
| Sampler / epochs | Pinned config201 B64/K8/50;100 B128/K16/30;MSVR B64/K4/50. These are code defaults, not a statement that paper30 equals paper50. | All datasets B64/K8/50, seed42; author default seed1234. |
| Optimizer / nominal LR groups | Adam; config base201=3.5e-4,100=7e-4,MSVR=5e-6; visual base nominal5e-6 except adapters; bias factor2; MSVR classifier overrides to5e-4. Weight decay1e-4. | AdamW, visual base5e-6, fresh camera and new modules3.5e-4, weight decay1e-4. |
| Schedule | 201/100 CosineLRScheduler: warm10/5, common warmup_lr_init0.1*config base, lr_min0.001*base, warmup_prefixFalse, epoch noise enabled over[0,max_epochs), seed42. Thus nominal5e-6 visual group is not its complete effective-LR trajectory. MSVR WarmupMultiStepLR warm0, steps20/40, gamma0.1. | All datasets warm5 then deterministic multiplicative cosine over50, no author epoch noise; groups keep fixed nominal ratios. |
| Triplet image augmentation | ImageDataset separately calls the random transform for each modality; flip/crop/erase are sampled per call. Resize bicubic, mean/std0.5, pad10, flip/erase0.5. | SharedGeometryTripletTransform shares one flip/crop across all3 modalities; erasing is independently called. Same nominal resize/normalization/padding probabilities. |

## Evidence limits

Offline source read only, against already sealed243-file manifest. No GPU/model/optimizer/evaluation, new plan/registration, active-source or threshold edits. Author configs enable USE_A/USE_B by default; the no-module path described here requires explicit False overrides and has not been run in this readback. Differences are candidate control variables, not proven causes of performance gaps. Code defaults and paper recipe are distinct evidence.

The author no-module path is not the current shared-adapter global-only. Nominal optimizer groups must be distinguished from effective scheduled LR. The pinned RGBNT100 YAML says30 epochs; a future uniform50 control would be an explicitly modified schedule, not an exact unmodified-code reproduction.

The source differences do not demonstrate why any dataset lost mAP, and they do not authorize changing or rescuing the six running EV1 endpoints.

## Finite follow-up

Finish EV1 full6 unchanged first. Then register a finite foundation control with matched public initialization, full50, official protocol and independent global-only. Treat a whole recipe contrast as a package contrast, not single-factor causality; choose any further separation only from full results.

Line anchors and checked source SHA values are recorded in FOUNDATION_RECIPE_READBACK_20261002.json. No paper claim or performance forecast is made.
