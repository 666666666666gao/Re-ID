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
