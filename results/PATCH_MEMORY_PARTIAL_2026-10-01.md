# Patch memory full50 partial intake — 2026-10-01

Scope: four of six registered endpoints accepted at 01:11 CST. The two RGBNT100 endpoints are still training; this is not a complete-panel audit or a successful algorithm gate. Original failed hardware campaign remains sealed.

| Dataset | Mode | Best epoch | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---|---:|---:|---:|---:|---:|
| RGBNT201 | local_memory | 2 | 72.5442 | 73.9234 | 83.1340 | 88.0383 |
| RGBNT201 | full_memory | 2 | 72.5335 | 73.9234 | 82.6555 | 87.9187 |
| MSVR310 | local_memory | 10 | 52.3622 | 67.8511 | — | — |
| MSVR310 | full_memory | 10 | 52.2919 | 67.8511 | — | — |
| RGBNT100 | local_memory | — | — | — | — | — |
| RGBNT100 | full_memory | — | — | — | — | — |

Every completed row ran all50 role epochs and strictly reloaded its single official-fused-mAP-best checkpoint. Full gallery order/metadata and camera/time-period filtering were checked. Same initial state and trainable parameter count within each dataset; no M3, no local auxiliary ID target, no environment/model/loss/schedule changes during recovery. This is a second training stage on dataset-trained pure CLIP ReID weights, not50 total epochs from public CLIP. Official test data participate in epoch selection; one development seed does not establish held-out or multi-seed robustness.

Full minus local: RGBNT201 mAP −0.0106882268pp, MSVR310 −0.0703240708pp; Rank-1 unchanged in both. These paired outputs do not support a gain from expanding Patch support in the two completed datasets. The registered three-dataset gate remains pending the full panel; no rescue by threshold changes or seed searches.

At those same checkpoints, joint-local mAP is39.7552→33.5180 on201 and11.9087→6.6501 onMSVR. Shared-global mAP is72.4155→72.4552 and52.3949→52.6639 respectively. These are within-checkpoint diagnostic outputs, not independently trained controls or proof of a unique failure cause.

Original verify() checked48 metrics over fused/shared-global/joint-local distance matrices. Maximum CPU recomputation error:2.7378210063489e-6 percentage points (<1e-5). New completed201-local checkpoint SHA0fb8d66e8e3ea9d6ad65d339ee123bf31a4826bcf051d6d633cb7417d8306b27; MSVR-full ed4d47204df376401411438f70e47e1a01df8756647536bfc9a034998642c92f. Preserved old201-full/MSVR-local checkpoints were separately strictly evaluated after recovery.

Evidence: logs/patch_memory_recovery_intake_20261001_0111/accepted_snapshot_0103.json, original raw training/evaluation receipts and INTAKE.json. All17 archived text members retain original bytes and verifiedSHA. Checkpoints, probes and distance arrays stay remote.210 frozen runtime sources still match the preregistered manifest.

At01:13 the remaining RGBNT100 trainers1054538/1062431 were running15/50 and16/50 onGPU3/0. GPU1/2 were released after their endpoints completed. The final observer1119184 starts reading at02:35 CST, then every240sec until the existing panel terminates; it performs no training/retry. Overall research goal remains ACTIVE / UNMET.

## All-query and slot mechanism diagnosis,01:44 CST

| Dataset | Rank-1 repairs / new errors | AP improved / worsened queries | Identity-macro ΔAP pp |
|---|---:|---:|---:|
| RGBNT201 | 1 / 1 | 235 / 195 | −0.00323 |
| MSVR310 | 5 / 5 | 236 / 305 | +0.11418 |

The official query-mean mAP deltas remain negative as above. Identity resampling is descriptive fixed-model analysis, not training-seed or held-out uncertainty. Parameter count/initial state are matched; accessible Patch support is the intended capacity difference.

Read-only complete-query/gallery inference additionally measured all three roles and modalities. All first-batch hook differences were0; all model state hashes were unchanged; all retrieval metrics matched the selected checkpoint within1e-5pp. Coverage:201 query/gallery836/836; MSVR591/1055. The hook returned the original sample output and only recomputed its attention statistics.

| Dataset | Mode | Effective Patch count | Slot attention cosine | Selected-content cosine | Distinct top1 Patches per16slots |
|---|---|---:|---:|---:|---:|
| RGBNT201 | local | 7.8569 | 0.0378 | 0.6265 | 14.0860 |
| RGBNT201 | full | 112.8362 | 0.9507 | 0.9864 | 8.4514 |
| MSVR310 | local | 5.1862 | 0.0312 | 0.6554 | 13.8825 |
| MSVR310 | full | 67.2323 | 0.9648 | 0.9857 | 3.2023 |

These are query-split means across3roles×3modalities; full gallery and per-role/per-modality tables are retained. Full support samples more similar content. Local masks mechanically limit attention overlap. Sampled-content statistics precede anchor-vector addition, role bridging and role operators; they do not establish final-role redundancy, physical-part correspondence or a unique causal explanation.

Source tools/diagnose_patch_memory_slots.py SHAaf6c92ab77dd702b5e1b4eae46eb099889952fb01b66929c41d8e9ff955e2b32. Plan SLOT_DIAGNOSTIC_PLAN_20261001.md and16 original JSON/log files plus intake receipt are archived. CPU diagnostics captured actualexit0. GPU workers produced COMPLETE reports and were absent on observation, with no retained wait parent; no GPU OS exit code is asserted. An initial source-binding error before CPU jobs is retained separately; it did not alter training or the verifier. RGBNT100 is not included until its pair is formally accepted. This is not a successful six-end algorithm gate or a SOTA claim.

## Archived trajectories and prior-art review,02:31 CST

See [four-end trajectory analysis](PATCH_MEMORY_ARCHIVED_TRAJECTORY_2026-10-01.md) and [primary-source slot review](../refine-logs/patch_memory_roles_v1/SLOT_COMPETITION_PRIOR_ART_20261001.md). No active training progress or new endpoint is included. Existing observer1119184 was verified Ss with the original02:35 schedule. Formal scope remains4/6; full gate pending.
