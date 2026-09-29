# Experiment integrity audit — first formal cross-depth artifacts (§41.659)

**Overall verdict: WARN. Integrity status: warn.** Fresh context, same-family review; **acceptance_status: provisional**. Native routing requested `gpt-6-astra` with separate `max` reasoning. Actual backend/model/effort attestation is unavailable; this report makes no stronger identity claim.

The three formal text bundles, all three terminal child campaigns, all 150 epoch records, 6,298 formal training steps and six eight-batch M0 text bundles are internally consistent. The recorded best epochs and reported numbers match. No demonstrated fabricated-GT path, self-normalized performance score, metric splicing or numerical contradiction was found in this scope. This is **not independent remote checkpoint or array replay**: those artifacts and the primary protocol payloads were unavailable, and SSH was prohibited.

## Scope and provenance

The reviewed claims were located independently in `EXPERIMENT_PLAN.md:9-41`, `EXPERIMENT_TRACKER.md:3-14`, and the immutable `first_formal_audit_input_659/HANDOFF_REVIEW_INPUT.md:5,7,13000-13020` under `refine-logs/cross_depth_role_state_v1/`. The claim copy SHA-256 is **97f4bd986aa151290b5deeeaf6ae39fe3d8690ae4005b1b82d1f44199706e50a**. Its selected sections, not the historical handoff as a whole, were audited.

All supplied scripts were read; every supplied JSON/JSONL was parsed, including every scalar row and complete progress histories. Pure scalar checks ran locally with Python `-B`; no trainer/evaluator, GPU inference, SSH, package install, or remote execution ran. Only this Markdown and its JSON companion were written. Prior reports, source, training and receipts were not edited.

The JSON companion preserves exact raw input hashes, supplemental dependency hashes, every local runtime hash comparison, missing runtime paths, claim impacts, and structured checks. Extra runtime files were hashed only; that inventory is not a semantic review of every historical module. Policy tracing normally asks for a separate trace directory, but this task explicitly limited writes to two files. The parent must retain the native reviewer call trace; no fictitious trace path or opaque agent ID is recorded.

## Deterministic endpoint results

All rows below use seed42 and contain the complete 50-epoch history. “Accepted” means the supplied producer collector labelled them `VERIFIED_COMPLETE`.

| Dataset / mode | mAP-best epoch | mAP | Rank-1 | Rank-5 | Rank-10 | Training + epoch-eval seconds | Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| RGBNT201 / mixed_once | 2 | 72.5888817496 | 74.1626799107 | 82.8947365284 | 88.0382776260 | 1735.132744 | 2649 |
| RGBNT201 / depth_mean | 2 | 72.5327582009 | 73.9234447479 | 82.7751219273 | 88.0382776260 | 2037.563269 | 2649 |
| MSVR310 / mixed_once | 15 | 53.0316750259 | 68.1895077229 | 82.2335004807 | 87.6480519772 | 1377.478853 | 1000 |

Evidence: each corresponding `logs/cross_depth_full_*_66[23]_20260929/official_metrics.json:3-40`; RGBNT201 `training.json:55-66,631-646`; MSVR310 `training.json:211-222,631-646`. The complete 50 rows were inspected and independently maximized by `(mAP, epoch)`. Latest-epoch exact ties match `tools/run_correspondence_context_identity.py:140-142` and `tools/collect_cross_depth_role_state.py:65-85`.

The depth_mean RGBNT201 maximum Rank-5 occurs at epoch1, and MSVR310 rank maxima occur outside its mAP-best epoch15. The reported rows correctly retain the metrics of epochs2 and15 rather than using those larger rank maxima. Every receipt hash in the accepted matrix matches the corresponding local official JSON. The first accepted MSVR310 row in `logs/cross_depth_accepted_662_20260929.json:25-131` is exactly equal as a parsed object to the later row at `logs/cross_depth_accepted_663_20260929.json:125-231` and the first-archive row. Archive text hashes also match.

## Checklist A–F

### A. Ground-truth provenance — WARN (static data flow passes)

Training labels come from protocol training records; query/gallery identities and camera/scene fields come from those records, not model outputs. See `tools/official_three_dataset_data.py:8-18`, `tools/run_correspondence_roles.py:79-108`, and `tools/run_correspondence_context_identity.py:194-224`.

The camera scorer removes only same-identity/same-camera pairs (`tools/train_rgbnt100_signal_oof.py:253-268`). MSVR removes only same-identity/same-scene pairs (`tools/train_msvr310_signal_oof.py:223-238`); the plan defines scene as time period (`EXPERIMENT_PLAN.md:29`). Both require an eligible positive per query. No top-k gallery truncation or per-query threshold appears. The inspected entry calls the upstream author scorer and asserts its path and metric parity.

However, the actual three protocol JSONs, upstream scorer file and raw distances are not local. Their SHA strings in `logs/cross_depth_manifest_658_20260929.json:254-269` establish expected bindings, not independent dataset authenticity or observed gallery contents. This prevents a full provenance PASS.

### B. Score normalization — PASS

AP is mean precision at positive ranks, and Rank-k is the proportion of queries with their first eligible positive within k. The only reporting scale is multiplication by100, in percentage points. Neither scorer divides a metric by the model's own extrema or output statistics. Feature L2 normalization in `tools/run_official_three_dataset_roles.py:230-238`, candidate softmax in `modeling/trifusion/correspondence_context_identity.py:19-32`, and layer normalization are representation operations; they are not post-hoc metric inflation.

### C. Result existence and reported numbers — WARN (local text checks pass)

All requested text inputs exist, parse, and match the numeric claims in `EXPERIMENT_TRACKER.md:11` and `HANDOFF_REVIEW_INPUT.md:13006-13016`. All 6,298 formal steps, 48 M0 steps, archive hashes, best rows, diagnostics, initialization strings and recorded elapsed times reconcile.

The remote `best_map.pth`, `m0_reload_probe.pth`, `official_distances.pt`, baseline tensors and data were not opened or independently hashed. `logs/cross_depth_first_archive_662_20260929.json:119` and `logs/cross_depth_archive_663_20260929.json:56` explicitly disclose that boundary. There is no evidence of a phantom numerical result, but remote payload existence/content cannot be accepted from a hash string alone.

### D. Dead metric code — PASS for the inspected active route

The wrapper patches the context entry at `tools/run_cross_depth_role_state.py:67-77`; the context entry patches the base runner at `tools/run_correspondence_context_identity.py:244-254`. Training calls the official fused metric at lines137-142; evaluation calls both score implementations as appropriate and writes all fused/global/local outputs at lines214-240. The collector invokes the scalar audit at `tools/collect_cross_depth_role_state.py:103-114`. Corresponding keys appear in the receipts.

The OOF scripts' `evaluate_gallery` functions have their own call sites (`tools/train_rgbnt100_signal_oof.py:389-391`, `tools/train_msvr310_signal_oof.py:344-346`); their historical 30/50-epoch OOF trainers are not the current trainer. M3 prediction and auxiliary identity branches are intentionally disabled, rather than silently counted as active evidence (`modeling/trifusion/correspondence_context_identity.py:69-90`, manifest lines272-280).

### E. Scope — WARN; current performance wording is appropriately limited

This snapshot provides **3/9 formal endpoints, two formal datasets, one training seed**, and **6/9 M0 endpoints across three datasets**. No recurrent formal endpoint or formal RGBNT100 endpoint is accepted. At 21:54:54 the matrix has three accepted rows, four UNACCEPTED/RUNNING rows and two PENDING rows (`logs/cross_depth_accepted_663_20260929.json:5-6,117-123,341-377`). A six-M0 count is not six performance endpoints.

The plan and handoff explicitly disclaim established recurrence benefit, unbiased generalization, multi-seed evidence and SOTA (`EXPERIMENT_PLAN.md:9,23,35`; `HANDOFF_REVIEW_INPUT.md:5,7,13012-13014`). Official query scores are used every epoch for selection; these are exploratory results on a consumed benchmark, not an untouched test estimate. Historical context-panel effects at plan line5 and prior-panel/SOTA history were not re-audited.

### F. Evaluation classification — PASS: real_gt route, with payload limits

Formal fused/global/local retrieval and paired diagnosis are **real_gt** by inspected code: all use registered dataset labels. M0 is supervised real-data engineering validation, not a retrieval result. No active model-generated reference is reported. The classification does not attest the unavailable dataset bytes. The synthetic structural check mentioned in the plan is explicitly not ReID performance and was not an input to this runtime review.

## Phase exits, bindings and gallery scope

The progress snapshot directly records m0/train/evaluate COMPLETE and exit0 for mixed_once RGBNT201 (lines61,158,842) and mixed_once MSVR310 (lines1940,2037,2721). For depth_mean RGBNT201, m0/train have exit0 (lines3414,3511), but evaluate is still RUNNING at lines4152-4153 in the **21:54:00** snapshot. Its official receipt completes at **21:54:12** (`official_metrics.json:38`), before collection at **21:54:54**. These times are consistent; the earlier snapshot is not a failed run.

The three additionally supplied terminal child campaign files now close the phase-exit gap: `logs/cross_depth_campaign_depth_mean_RGBNT201_659_20260929.json`, `logs/cross_depth_campaign_mixed_once_RGBNT201_659_20260929.json`, and `logs/cross_depth_campaign_mixed_once_MSVR310_659_20260929.json` each record COMPLETE at line2, m0/train/evaluate exit0 at lines53/100/147, and completed verification at lines151-153. All nine commands, phase order, output directories and verification objects match the endpoint receipts and accepted rows. For mean201, evaluate exited0 at21:54:13.594352 and the child completed at21:54:16.060450. **Phase-exit check: PASS for these three archived child campaigns.**

The collector requires all three COMPLETE/0 phases (`tools/collect_cross_depth_role_state.py:134-141`). A separate standalone collector command/exit receipt is still not supplied. The handoff's “collector exited0” (`HANDOFF_REVIEW_INPUT.md:13002`) should therefore remain attributed to the producer run record, not this reviewer's direct process observation. This does not indicate a failed collector.

Mode cannot be inferred from identical tensor keys. The new checkpoint schema saves and validates `depth_mode`, dataset, seed, protocol, baseline, variants and condition, then strictly loads keys (`tools/run_cross_depth_role_state.py:35-55`). The constructor and collector also bind mode, architecture and source hashes. All supplied binding strings agree; actual checkpoint payloads were unavailable.

The reused protocol reader fixes expected query/gallery sizes to RGBNT201 **836/836**, RGBNT100 **1715/8575**, MSVR310 **591/1055** (`tools/run_official_three_dataset_roles.py:58-68`). Evaluation extracts all records and asserts sizes (`tools/run_correspondence_context_identity.py:194-206`); collection checks complete ID/camera/scene arrays, exact matrix shapes and finite values for all three paths (`tools/collect_cross_depth_role_state.py:86-102`). All **36 stored** CPU parity differences are below1e-5; their maximum is **2.395593384107997e-6 pp**. That maximum was recomputed from the receipt values, not from the absent distance arrays.

## Scalars, M0 and frozen-state evidence

The supplied pure-stdlib scalar auditor was executed read-only on all three complete bundles. Each epoch has sequential batch indices and its logged mean equals the arithmetic mean of steps. All scalars are finite, and all auxiliary values are exactly0.

| Formal endpoint | Nonzero triplet steps | Largest `loss − (ID + triplet + auxiliary)` absolute error |
|---|---:|---:|
| RGBNT201 / mixed_once | 1139 / 2649 | 2.384185791015625e-7 |
| RGBNT201 / depth_mean | 1275 / 2649 | 2.3655593395233154e-7 |
| MSVR310 / mixed_once | 963 / 1000 | 2.3096799850463867e-7 |

Evidence: `tools/audit_correspondence_context_identity_losses.py:11-45`; each formal `training_steps.jsonl:1-2649` or `:1-1000`; accepted matrix scalar summaries at lines68-114,184-230,292-338. The original task uses ID + triplet; `auxiliary_target=none` disables the auxiliary head even though its dormant coefficient is1.0 (`tools/run_correspondence_context_identity.py:110-116`). M3 is off, so prediction_weight0.1 is also dormant.

All six M0 bundles contain epoch1, batch0..7, and matching means. They report **118/118 trainable tensors** observed with nonzero gradient at least once across eight updates, frozen Signal unchanged, and reload maximum difference0 (`training.json:43-55` in every supplied M0 directory). This is distinct from the millions-of-elements `initializer.trainable_parameters` field. Scalar logs do not contain per-tensor gradients or frozen-state snapshots; these M0 checks remain producer assertions backed by the inspected code at `tools/run_correspondence_context_identity.py:117-166`, not replayed tensor proof. The two RGBNT100 M0 triplet terms are all0; do not infer full-run triplet behavior from eight batches.

## Source, initialization, cost and mechanism interpretation

The registered source manifest has205 entries. A local read-only comparison found **71 exact raw SHA matches, 27 raw differences explained completely by CRLF→LF conversion, and 107 absent paths**. The absent set includes the upstream Signal tree/configs and the three protocol JSONs. All 27 normalized digests match the recorded runtime digests, including criterion and Mamba; their raw SHA values remain separately preserved. No byte identity is asserted for those27. All supplied active campaign scripts covered by the manifest match it exactly. This supports source consistency within the archive; it cannot establish all205 remote bytes or an unchanged package/CUDA environment. `logs/cross_depth_launch_sync_658_20260929.json:7-8` states205 checks but is itself a producer receipt.

Within each dataset, mixed_once and depth_mean M0 initial-state hashes and parameter counts agree. For RGBNT201, both formal receipts record **2,617,345** trainable elements and initial hash `595e5d6ebab9eb4a5d5cf930b4328f22200d038a9f0c0e4ddf97756be9989a92`; MSVR has2,592,769 and RGBNT100 has2,431,489. See each `training.json:25-26`. This matches copied initialized role state and frozen uniform depth logits (`modeling/trifusion/cross_depth_role_state.py:8-18,74-85`).

Mean201/mixed201 recorded training-plus-epoch-eval time is **1.1742982063× (+17.42982063%)**. The timestamps at each formal `training.json:41,646` reproduce this exactly. Timing starts after initialization and ends after training/frozen-state check; it excludes M0 and final independent evaluate. It is an observed wall-time comparison across campaign GPU slots, not a controlled FLOP or deployment-latency result. Per-epoch training timers exclude epoch evaluation (`tools/run_correspondence_context_identity.py:135-138`).

**Mechanism limit:** depth_mean averages three independent role outputs, while depth_recurrent returns only the last recurrent output (`modeling/trifusion/cross_depth_role_state.py:60-71`). The registered comparison changes recurrence and output aggregation together. A future positive difference supports that compound design comparison; it does not uniquely attribute benefit to persistence. This is an interpretation limit, not a reason to change the already registered running jobs.

## Pair diagnosis provenance and claim impact

The pair's checkpoint/distance/receipt bindings match the two accepted201 rows; its actual local SHA matches `logs/cross_depth_archive_663_20260929.json:6`. The stored metric differences match candidate-minus-control (maximum allowed float discrepancy versus upstream metrics1e-5).

From all30 stored identity records (`logs/cross_depth_pair_rgbnt201_mean_663_20260929.json:52-203`), this review independently obtains **836 queries**, weighted ΔmAP **−0.05612354871527553 pp**, macro mean **−0.04945238940888117 pp**, and **11 improving /13 worsening /6 tied identities**. The stated0 repairs and2 new errors imply **−2/836×100 = −0.23923444976076555 pp Rank-1**, consistent with the pair's CPU metric delta. Individual flips and193/212 query AP sign counts cannot be reconstructed from the aggregate identity vector alone.

The stored bootstrap interval **[−0.13702899186122802, 0.03909153709804158]** contains0 and resamples fixed-model identities, not training seeds. It was not re-executed; neither query arrays nor a suitable local NumPy environment were available, and no environment change was made. The helper does not write its own source hash or input matrix hash (`tools/analyze_correspondence_distances.py:63-75,78-88`), and its path is absent from the205-file manifest. Its reviewed current SHA is preserved below, but that is not an attestation of the exact producer version.

The handoff correctly limits this to current201 controls and same-weight output decomposition (`HANDOFF_REVIEW_INPUT.md:13012-13016`), and explicitly explains that the pair helper's generic “Different capacity” boundary does not mean different parameter counts here. The formal fused-minus-global mAP values independently subtract to **+0.2018030480**, **+0.1085534647**, **+0.7290687070** for201 mixed,201 mean,MSVR mixed; MSVR Rank-1 is **−0.1692056656 pp**. These are not independent trained ablations or proof that all roles are necessary.

## Concrete blockers and next evidence

1. Before claiming independent end-to-end integrity acceptance, replay the exact remote baseline/checkpoint/probe/three-path arrays against the frozen protocol and upstream scorer hashes. No such replay occurred here.
2. All three child terminal campaign files now document the nine zero phase exits. If retaining the exact standalone collector-exit0 claim, additionally preserve its command/exit receipt or explicitly attribute it to the producer.
3. Keep the performance scope at3/9, seed42, official-best exploratory results. All recurrent and remaining formal evidence is still outside this snapshot.
4. Bind future paired diagnostics to the producer script SHA and exact matrix SHA in a new companion receipt, preserving the existing one; replay query outcomes from the archived distance hashes before independent acceptance.
5. Retain the recurrence/aggregation confound in mechanism claims. Do not treat the two-mode result as an isolated persistence effect.

The existing qualified numerical handoff is supportable as archived exploratory evidence. Remote execution assertions, all205-source/environment immutability, bootstrap/query-level replay and strong performance/mechanism claims require the qualifiers above. No edit to training or any existing receipt is recommended by this audit.

## Exact audited input hashes

The following SHA-256 values are hashes of the raw bytes actually read. Relative paths resolve under `C:/Users/gb/.trifusion_github_publish_22c3bee/`. The companion JSON also records hash-only comparisons for the wider runtime manifest and policy inputs.


| Input path | Raw SHA-256 |
|---|---|
| `tools/run_cross_depth_role_state.py` | `937bf84151002e4b6e62a3a8875981e680ebd99a1f93c14ab937e07a06f13184` |
| `tools/collect_cross_depth_role_state.py` | `99e404637d186f297a35f337a46ef2950e2215fb4db6b7a5d719ab15e0cf8aca` |
| `tools/queue_cross_depth_role_state.py` | `4cce880fcc81240613ad1c1266f5a780e108a7d423426c02103a8884682432d2` |
| `tools/run_correspondence_context_identity.py` | `9f4cc9f11998819d4a927a32f4e47baf0c4ff9d14c4bc93b0dc35b0153ebc142` |
| `tools/run_correspondence_roles.py` | `e50865fb923297cd61cf38b33ec2bc95154f8c503b5dfe5de9823cad3f03d7ef` |
| `tools/audit_correspondence_context_identity_losses.py` | `bcc15e568c6a14efdaca1df5268c5abdec716287d3b6efb2482804add724abf1` |
| `tools/analyze_correspondence_distances.py` | `833ebeb47cb5840422710fc00df1a974338cb04078249720d867f18b945f997a` |
| `tools/train_rgbnt100_signal_oof.py` | `4462b73139e034d450d955c5b4a994aaa967548e9cc503d97bb753a14ea03b22` |
| `tools/train_msvr310_signal_oof.py` | `c25579d931df34481fef321558f0b9a1f9179d72a25d6047dbbfd3682d4ae60f` |
| `modeling/trifusion/cross_depth_role_state.py` | `8e18c7c9d397fbe3eff6f7360a5ab890ebbeecf1404628da6979ade47544b7ba` |
| `modeling/trifusion/correspondence_context_identity.py` | `eda3d1473496b33e4b69402dba7770e67d23b33b07afd41d23105b7f148d63af` |
| `modeling/trifusion/correspondence_evidence_readout.py` | `e737a977d39ccbe09a5509118e5b5333d653f7f13381e65e84af1a13e5c28dc4` |
| `modeling/trifusion/correspondence_roles.py` | `e2a51154b5b327341f35e453dd32a821b091a49a669b49a7c70e79e6e2f2591b` |
| `logs/cross_depth_manifest_658_20260929.json` | `67e383e88f5151d12cbf3eae38111df42f7e1279ba7118b2a6f2cc4ca6044b14` |
| `logs/cross_depth_accepted_662_20260929.json` | `7b59d6d7f48efd574fcf76524591f6611c7a4aacd4384f3b5393100a8e3bcf5b` |
| `logs/cross_depth_accepted_663_20260929.json` | `c6fa3a9d0f4ed32a292b6ccb28251f365a751b7182685a793119987374010908` |
| `logs/cross_depth_progress_663_20260929.json` | `1bc73da96265b17035281a0014066c2a0a9ed77233ac94b998ed3b562ef3feef` |
| `logs/cross_depth_first_archive_662_20260929.json` | `77fba1d9d137bc1580ec63a4cdd46d974c0e6cf340cec461caae811a92514dee` |
| `logs/cross_depth_archive_663_20260929.json` | `f1842992bc9ecfa89b6643f8048f67aebaf9a28fcb5b253b6b35601f8ec9582a` |
| `logs/cross_depth_pair_rgbnt201_mean_663_20260929.json` | `1d111f0b9195ec4a8e2695dbd098b04e5efb62dc99fccdfa19fc85c28fed9e47` |
| `logs/cross_depth_full_mixed_once_MSVR310_662_20260929/training.json` | `d221201b6d479e7ca2f0392c9c14b19b416d94fc9451097bacacb8253e38c745` |
| `logs/cross_depth_full_mixed_once_MSVR310_662_20260929/training_steps.jsonl` | `21a85b3d45eaf18379de1bf4e1b7ddf6ba50d85aaa99caa06917dc89d789fd00` |
| `logs/cross_depth_full_mixed_once_MSVR310_662_20260929/official_metrics.json` | `cc5a9d8b6bb39016116a361b7d1cc4a6b9f721545acda9159db882c946635389` |
| `logs/cross_depth_full_mixed_once_RGBNT201_663_20260929/training.json` | `b5e01aaed9819cbfc1c98036dca54b2d40588ba1554928960c52a8dc934673af` |
| `logs/cross_depth_full_mixed_once_RGBNT201_663_20260929/training_steps.jsonl` | `c99a424e84dd6e02c037534a0136dd8ccca026affbbf8b2718ce7d94475ed44b` |
| `logs/cross_depth_full_mixed_once_RGBNT201_663_20260929/official_metrics.json` | `7f0f1d98315d265fa3345e75b123ea153b95256ddd6c094a826e297927e74407` |
| `logs/cross_depth_full_depth_mean_RGBNT201_663_20260929/training.json` | `650d417ce1b8b941ae58fe32d9186cbb22f2236547daf3c8e71cd770ec5e869b` |
| `logs/cross_depth_full_depth_mean_RGBNT201_663_20260929/training_steps.jsonl` | `485ea7a3ab8db7e1b159eafcd70d17eba815468447bcc33c4d0ef7fd62a8f199` |
| `logs/cross_depth_full_depth_mean_RGBNT201_663_20260929/official_metrics.json` | `7ba280f26f01f9b320a60035f9093507ece042ef19aaf3d92cb5dfc312329086` |
| `logs/cross_depth_m0_depth_mean_RGBNT100_663_20260929/training.json` | `d24c301fd871e5e8bfb3d45d063818bd3efbf5808285f553e0158151699057f0` |
| `logs/cross_depth_m0_depth_mean_RGBNT100_663_20260929/training_steps.jsonl` | `c73142a46f473c6880c44f9d9f7a31927ef0cee02c0be5215e4b637938b5ecbf` |
| `logs/cross_depth_m0_depth_mean_MSVR310_663_20260929/training.json` | `ad39e61c6d8f99bb18a25275d52720d437d660356c2cc501bad73b2a1eee77d0` |
| `logs/cross_depth_m0_depth_mean_MSVR310_663_20260929/training_steps.jsonl` | `d6d9fdf46f0b683e98c98e4bf43115891ada4549b35263b4e4b220d5d37f98db` |
| `logs/cross_depth_m0_mixed_once_RGBNT201_658_20260929/training.json` | `cb054d7303bec03c5c076c4dfa7e11cf4e7247be7bb195311b12d34ca67c6af9` |
| `logs/cross_depth_m0_mixed_once_RGBNT201_658_20260929/training_steps.jsonl` | `9917289314f3156b7ab340a55bb6feb128792f0f0e009d69712e87edc868e03a` |
| `logs/cross_depth_m0_mixed_once_MSVR310_658_20260929/training.json` | `7917ac225b5f052a8e711a097e914f08c59bec96a80a9e16046072031ac3e42d` |
| `logs/cross_depth_m0_mixed_once_MSVR310_658_20260929/training_steps.jsonl` | `4b33dca6cd21c8a4ae2d1889c063f3be202db8dcf5599149eb529dffb6134e5d` |
| `logs/cross_depth_m0_depth_mean_RGBNT201_658_20260929/training.json` | `881da2600de68ea5a7792ffa94e219eccd327a9fcba21495693dc7c44acbbab6` |
| `logs/cross_depth_m0_depth_mean_RGBNT201_658_20260929/training_steps.jsonl` | `16d395e02556bd9cd93c9d0ceb635e51fb2ee08833b63132ee99d3de433e08d1` |
| `refine-logs/cross_depth_role_state_v1/EXPERIMENT_PLAN.md` | `06aadabfa79bfdb2022e9bd88c14a788649ab2050217f8317110c4ef8747cc0b` |
| `refine-logs/cross_depth_role_state_v1/EXPERIMENT_TRACKER.md` | `9c97a772a4fb153b66aba6132f3bc33bcfe0ced5ff504f233208f087527496b8` |
| `logs/cross_depth_m0_mixed_once_RGBNT100_658_20260929/training.json` | `778cee7ccad9ddd72c2521c19d5b747a89dc1c702a9116cb489c488272fb8cd0` |
| `logs/cross_depth_m0_mixed_once_RGBNT100_658_20260929/training_steps.jsonl` | `42fb3285b92298265c810c2b2a4a2e5f51df16ce3660ebc590dba6d61b55fc20` |
| `tools/queue_correspondence_roles.py` | `746fbdb8bdad04290a9ef22c0550e873c9dce9de74c2c3c3767fea9278be69bf` |
| `tools/collect_correspondence_roles.py` | `f81af437fe5ccfea4361c1ed182cb9b8841b362214d2a5220309a0fd2c78fff8` |
| `logs/cross_depth_launch_sync_658_20260929.json` | `6a101190080502f484b60651320755698bdfe2fab6aa3b5a8a3f15c88dbf7d2d` |
| `logs/cross_depth_campaign_depth_mean_RGBNT201_659_20260929.json` | `d886f325a130411f0bb5c0306b909456db8852d70f218ed496f7447b56642e2a` |
| `logs/cross_depth_campaign_mixed_once_RGBNT201_659_20260929.json` | `fda442d6aa26eeb6476b0ce9af88b72dddfba41fdf11ddec112f2f84a6571b5f` |
| `logs/cross_depth_campaign_mixed_once_MSVR310_659_20260929.json` | `9cb1fd5cd37f21bd0275559f7754fbb2866c9446429c68d2a421b8597730c94b` |
| `tools/official_three_dataset_data.py` | `c3180a88d12440709319d2025928fc699ff3b7c020b11697d468453fbfc68062` |
| `tools/official_three_dataset_model.py` | `e103ea0f4d6342b0dc129dc2dace0e8c0d0e6c956cfe94729ccb70d9588cf0f9` |
| `tools/run_official_three_dataset_roles.py` | `f30bb00a0b71429e9d11dd196cd1d543202a133c663ad66e175d2f82e28d448d` |
| `refine-logs/cross_depth_role_state_v1/first_formal_audit_input_659/HANDOFF_REVIEW_INPUT.md` | `97f4bd986aa151290b5deeeaf6ae39fe3d8690ae4005b1b82d1f44199706e50a` |

Generated at 2026-09-29T14:21:02.354Z. The JSON is the machine-readable evidence ledger; no remote execution is implied by its deterministic local checks.

