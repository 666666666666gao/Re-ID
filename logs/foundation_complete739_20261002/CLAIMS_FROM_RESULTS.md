# Claims from results — completed F1 foundation packages

Date: 2026-10-02  
Reviewer canonical agent ID: `/root/claim_foundation_complete739`  
Review route: fresh `gpt-6-astra`, reasoning effort `max`, as specified by the parent; no external reviewer or child agent  
`claim_supported: yes` — only for the explicitly bounded development-protocol claim below  
`confidence: medium`  
`integrity_status: warn`  
`review_independence: same-family`  
`acceptance_status: provisional`

**[INTEGRITY: WARN]** The existing fresh audit flags an omitted runtime binding for the paired-diagnostic helper, locally unavailable binary/image contents, and the single-seed, consumed-benchmark scope. This claim review preserves those warnings. The deterministic `evidence_check.py` helper remains unavailable; local existence/hash/arithmetic checks are not a substitute helper PASS or external semantic acceptance.

The task prompt supplied expected metrics and interpretations. I independently read the primary endpoint/history/step records and the active training/scoring source, but this was not a blind review. The canonical agent ID above is exposed by this task; no additional opaque runtime ID or independently queryable model identity was available. Model/effort attribution is the parent-specified routing.

The initial review was interrupted by the actual error “Selected model is at capacity,” recorded at 2026-10-02T22:28:59.955487+08:00. At that point this Markdown was a draft and no structured JSON or completed final verdict existed. The parent resumed this same agent at 22:33:44.103904+08:00 with unchanged model/effort. This completed judgment finishes the remaining artifact work from the evidence already inspected; it does not relabel the failed first call as successful. The original error and both actual requests remain in `E/.aris/traces/result-to-claim/2026-10-02_f1_complete739/`; parent-owned trace metadata is not edited by this reviewer.

## Supported claim

Under the executed fixed seed42, full50 development protocol, the modified author no-SIM/no-AlignM foundation package attained higher selected mAP and Rank-1 than the current no-module package on RGBNT201 and RGBNT100. On MSVR310, the positive numerical difference was smaller than the predeclared +0.5 mAP advancement threshold. These observations justify a controlled investigation of foundation training before assigning later gains to new roles. They do not identify the responsible component or establish final TriFusion/SOTA success.

The verdict is **yes for this narrow claim**, not partial support rounded up for a broader claim. The broader claims of all-three-dataset advancement, isolated feature-normalization causality, stable cross-seed superiority, or final-method success are unsupported.

## Primary results and the fixed gate

All metrics below are percentages; differences are percentage points. Each endpoint supplies all four retrieval metrics from its single mAP-best checkpoint. The selected epochs and metrics independently agree with the corresponding maximum-mAP history row, accepted matrix, and terminal summary.

| Dataset | Author epoch | Author mAP / R1 | Current epoch | Current mAP / R1 | Author − current mAP / R1 | Advancement |
|---|---:|---:|---:|---:|---:|---|
| RGBNT201 | 27 | 73.47275487487326 / 77.15311050415039 | 26 | 62.580551571472775 / 62.44019269943237 | +10.892203303400485 / +14.712917804718018 | PASS |
| RGBNT100 | 9 | 84.03193833734271 / 96.20991349220276 | 12 | 77.58429184381957 / 94.1690981388092 | +6.447646493523138 / +2.0408153533935547 | PASS |
| MSVR310 | 38 | 50.83882666558341 / 68.69712471961975 | 24 | 50.35673958098721 / 67.51269102096558 | +0.482087084596202 / +1.1844336986541748 | **FAIL** |

The registered rule is official query-mean mAP improvement **at least 0.5**, together with nonnegative Rank-1 change, evaluated separately for each dataset. MSVR310 fails the mAP condition by 0.017912915403798 points. Its identity-macro AP change, +0.5284354345205875, is a different estimand and must not replace official mAP; neither rounding nor changing the estimand repairs this failure. MSVR Rank-5 is unchanged and Rank-10 decreases by 0.3384113311767578 points, so even “all retrieval metrics improve” is false. No all-dataset progress pass is supported. [S: experiment plan:40–46; report source:53–58; E: six primary official_metrics.json files; terminal SUMMARY.json/pairs]

The six formal histories contain 300 consecutive epochs and 16,692 logged updates: author/current counts are 2,649/2,649 for RGBNT201, 3,129/6,559 for RGBNT100, and 706/1,000 for MSVR310. Equal epoch budgets therefore do not mean equal updates, sample exposure, or computation. Six separate eight-batch M0s are engineering probes, not six more accuracy results or seed replications. All six final-epoch mAP values are lower than their selected best values. Reporting selected bests is consistent with the contract; it is not evidence of unbiased test performance. [E: six training.json and training_steps.jsonl files; S: run_foundation_recipe.py:189–312]

## What the evidence establishes

1. **A real, completed package comparison.** All 18 recorded child commands—six M0 and six train/evaluate pairs—exit 0. The parent campaign records one terminal CPU-report invocation and exit 0 after all endpoints. I independently checked all 91 primary text hashes and all 249 sealed source hashes without mismatches. These checks establish consistency of the supplied records, not exhaustive knowledge of unrecorded activity. [E: INTAKE.json; raw/logs/foundation_recipe_20261002_v1/{manifest,campaign,accepted_matrix}.json and child campaign.json files]
2. **The active evaluation uses dataset ground truth and complete galleries.** The executed path reads original identities/cameras/scenes from the fixed protocol, extracts 1536-dimensional features, applies L2 normalization and squared Euclidean ranking, and compares the author scorer with a separate implementation. RGBNT201/100 exclude same-identity/same-camera entries; MSVR310 excludes same-identity/same-scene entries and retains gallery distractors. The protocol contains 103 MSVR gallery-only identities. This reviewer checked protocol counts and train/evaluation identity separation; the separate fresh audit additionally checked all filename-derived labels. No image bytes or distance tensors were inspected here. [S: tools/official_three_dataset_data.py:8–18; tools/run_correspondence_roles.py:60–108; tools/run_official_three_dataset_roles.py:230–238; comparators/Signal-cd1b0a6/utils/metrics.py:68,137; E: reviewer_foundation739/EXPERIMENT_AUDIT.md]
3. **The public visual and camera starts match within each dataset.** Initialization receipts agree on public CLIP, visual, camera, and the temporary current-head witnesses, while the complete author/current model-state hashes differ. Author heads differ from the retained current head; the author-side current-head witness is not evidence that the actual trained heads match. The source discards role/adapter wrappers and calls the no-module Signal path directly. [E: raw/logs/foundation_recipe_20261002_v1/initialization/{dataset}_{author,current}.json; S: tools/run_clean_clip_joint.py:37–99; tools/run_foundation_recipe.py:25–100]
4. **Foundation-package choices materially affect these selected development results.** The RGBNT201 and RGBNT100 differences are large enough to make a focused foundation-training diagnosis reasonable. A claim that new CNN/Transformer/Mamba roles caused these gains is unavailable because these endpoints contain no such role intervention.

## What the results do not establish

- **Single-component causality.** Training raw versus L2-normalized features, one fused versus three modality-specific heads, CE weighting, soft-margin versus margin0.3 triplet, Adam versus AdamW, actual LR schedules, augmentation coupling, and B/K settings differ together. For RGBNT201/100 the author scheduler includes noise; nominal visual LR alone does not describe its executed trajectory. Current's inherited cfg_yaml contains author settings that its actual branch overrides, so effective-recipe statements must follow source and logged LRs. [S: run_foundation_recipe.py:25–162,213–218; author layers/make_loss.py:37–56; layers/triplet_loss.py:113–135; solver/make_optimizer.py:4–43; solver/scheduler_factory.py:7–29]
- **Exact original-author reproduction.** RGBNT100's original 30-epoch schedule is extended to 50, and common workers4, AMP initial scale256, and the current evaluation loaders differ from the original author entry. This is the explicitly modified author foundation package.
- **Seed robustness, statistical significance over training, or untouched-test generalization.** There is one seed per condition/dataset; official labels were consumed for epoch and method development. Three datasets and six endpoints do not create independent seed replicates. The recovered helper's bootstrap resamples fixed selected models' identity-mean AP changes, not training runs, and does not account for selection.
- **A causal adapter or role result from historical references.** Historical shared-global is neither a concurrent nor a capacity-matched arm. Its mAP gains over current are +5.531117006128788, +0.9069817675123915, and +1.3656712960945185, but the latter two have negative Rank-1 changes. These references cannot establish uniform adapter advancement, rescue MSVR's F1 gate, or attribute gains to a new role mechanism. [E: SUMMARY.json/historical_references; S: report_foundation_recipe.py:59–69]
- **A diagnosed dead-triplet failure.** Direct aggregation of the current step logs finds positive triplet values on 1,882/2,649, 842/6,559, and 992/1,000 updates respectively. RGBNT100 has zero logged triplet on all 1,312 updates in epochs41–50; that is an observation, not proof of a defect or of the package-gap cause. Author logs contain combined per-head losses only, so author CE/triplet activity cannot be reconstructed. Feature norms and matched batch/augmentation witnesses needed for a mechanism comparison are absent.
- **A new research contribution, a final three-module result, SOTA, or completion of the overall goal.** F1 is foundation diagnosis on a consumed benchmark. The overall goal remains unmet.

## Integrity and missing evidence

The fresh experiment audit is `warn`, not `fail`; it does not invalidate the narrow descriptive endpoint comparison. Confidence is high in the checked arithmetic and selected-record consistency, but medium overall because the claim rests on bounded execution receipts and has limited generalization scope.

The paired helper imported by report_foundation_recipe.py was omitted from the original 249-file runtime source seal. I read the recovered helper and verified its SHA256 `833ebeb47cb5840422710fc00df1a974338cb04078249720d867f18b945f997a` against the earlier 17:52:32 pre-F1 collection and waiter hash. This narrows uncertainty but cannot establish that the helper was immutable throughout F1. The original seal remains unmodified. Paired diagnostic counts/intervals retain this warning; the primary endpoint differences above can be obtained directly from official_metrics.json without relying on that helper.

For context, the recorded identity-bootstrap intervals are [2.4076454989199263, 18.538231703864408], [3.3170855989939674, 9.403782812587892], and [-2.402104246186131, 3.6176192179890796] points for RGBNT201/100/MSVR. These are qualified fixed-model, identity-macro diagnostics. I checked their source interpretation and identity-weighted/macro arithmetic, not the remote arrays or the bootstrap distribution.

Specific gaps are: independent training seeds and a locked confirmatory selection/evaluation protocol for broader claims; controlled component experiments for causal claims; source-bound actual batch/augmentation and component/feature diagnostics for a normalization mechanism; direct inspection of the 24 remote-only checkpoint/distance assets and image-byte bindings for stronger asset assurance; and a pre-execution dependency seal including the paired helper for future runs. The missing historical runtime witness cannot be repaired by retroactively editing F1's manifest. The unresolved deterministic evidence helper is recorded, not silently replaced. [E: EVIDENCE_PRECHECK_UNAVAILABLE.json; diagnostic_source_closure/{SOURCE_CLOSURE,prior734_collection}.json and tools/analyze_correspondence_distances.py:28–75]

## Suggested next experiment

**Endorse the proposed F2 as the next bounded diagnostic, still unexecuted.** The existing logs do not identify a demonstrably better next factor. It is reasonable to test a source-visible single boundary rather than infer a component cause from F1, but the evidence does not make this boundary a proven or uniquely most likely explanation.

Run two fresh conditions, normalized versus raw input to the current training BN/classifier and margin0.3 triplet, on all three datasets for full50/seed42, exclusively on host2026. Keep current architecture, head organization, parameter/state shapes, public CLIP and fresh camera/head initialization, AdamW, actual LR schedule, B64/K8, augmentation, workers/AMP, and the formal L2-normalized 1536D inference path identical. Both controls must be newly run on that host; F1 on host2025 is context, not the normalized arm. [E: NEXT_INTERVENTION_CANDIDATE.md; S: run_foundation_recipe.py:42–53,148–162]

Before formal execution, verify full initial state and identical inference outputs on the same probe inputs, complete each eight-batch M0 from its own matched fresh start, then rebuild formal starts without carrying M0 updates. Preserve the exact current triplet formula and coefficient. Seal all actual training/scoring/report dependencies before execution, including the previously omitted helper. Confirm the protocol/image assets on the target host and record the actual hardware/software context.

During both runs, record detached feature norms, CE and triplet components, actual gradient activity and update/AMP evidence, actual LRs, and batch identity/order plus augmentation witnesses sufficient to verify matched exposure without changing the data pipeline. Seed42 alone is not the exposure witness. Preserve all 50-epoch histories and the original single mAP-best/tied-later selection, associated CMC, full galleries/filters, and 1e-5 strict-reload tolerance. Do not recover an engineering or scientific failure by changing batch size, loss, seed, threshold, or replaying to select a successful result.

Predeclare raw-minus-normalized as the comparison and retain the separate +0.5 mAP/nonnegative-R1 advancement gate for each dataset. Report all six endpoints before a decision. A pass would support an effect of this **combined training-feature boundary under the fixed current recipe**; because it changes both BN/CE input and triplet geometry, it would not isolate which loss route mediated the effect, nor explain all of F1's author-package difference. Failure to advance should close this immediate diagnostic without a margin/scale search or a claim of universal no-effect.

Subsequently, any broader training-performance claim needs preregistered independent-seed replication and a locked evaluation protocol. Any new-role claim needs concurrent no-role and appropriate capacity controls on one fixed credible foundation. These are future evidence requirements, not experiments executed or contributions established in this review.

## Evidence locations and review boundary

- **S:** `C:/Users/gb/.codex_tmp/foundation_recipe_v1_20261002/source_intake737`. The experiment plan is `refine-logs/foundation_recipe_v1/EXPERIMENT_PLAN.md`; source filenames without another directory above refer to `S/tools/`. Pinned author loss/optimizer paths refer to `S/comparators/Signal-cd1b0a6/`.
- **E:** `C:/Users/gb/.codex_tmp/foundation_recipe_complete_20261002`.
- Primary endpoints: `E/raw/trained-model/foundation_recipe_20261002_v1_full_{author,current}_{RGBNT201,RGBNT100,MSVR310}/{training.json,training_steps.jsonl,official_metrics.json}`.
- Main report: `E/raw/results/foundation_recipe_complete_20261002/{SUMMARY.json,REPORT.md}`.
- Prior audit: `E/reviewer_foundation739/{EXPERIMENT_AUDIT.json,EXPERIMENT_AUDIT.md}`; canonical reviewer `/root/audit_foundation_complete739`.
- The JSON companion stores exact endpoint values, arithmetic, activity, observed input hashes and explicit evidence references. `RECORDED_TRAINING_ACTIVITY.json` was treated as an executor aggregation, not semantic assurance; the activity statements above were derived from raw JSONL.

Only local text/JSON reading, hashing, arithmetic, and these two new reviewer artifacts were performed. No source/repository/primary-data edit, SSH, model/scorer/GPU/report execution, retraining, external review call, or child-agent delegation occurred. The parent's separate trace should preserve the actual task prompt and returned response. Routing is to preserve this bounded claim and prepare F2 as a controlled foundation diagnostic; it is not routing to paper-ready/SOTA acceptance.
