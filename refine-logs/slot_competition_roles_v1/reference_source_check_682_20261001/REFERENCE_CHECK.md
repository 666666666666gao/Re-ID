# Primary reference check — 2026-10-01

This is a source check during the registered six-end FP32 attention experiment. No model, loss, seed, schedule, selection rule or advancement gate was changed. Author-reported results below are external references, not local reproductions or evidence that the current experiment meets the research goal.

| Author report | RGBNT201 mAP / R1 | RGBNT100 mAP / R1 | MSVR310 mAP / R1 |
|---|---:|---:|---:|
| DEEP, CLIP | 79.6 / 84.2 | 88.5 / 97.6 | 66.0 / 82.1 |
| PMKD, DINOv2 | 84.7 / 88.9 | 91.6 / 98.0 | Not reported |
| DSGM, CLIP plus cached priors | 82.6 / 87.0 | 89.4 / 98.2 | 64.6 / 76.0 |
| RoDI, CLIP | 84.1 / 87.2 | 88.5 / 97.6 | 64.1 / 77.2 |
| RoDI, distilled DINOv3 | 85.3 / 87.9 | 89.0 / 99.1 | 71.8 / 84.8 |

All values are percentages. Each pair remains from one author table row. These sources are not an exhaustive SOTA ranking and do not establish equal resource or evaluation conditions.

## DEEP

The [author manuscript](https://aihuazheng.github.io/publications/pdf/2025/2025-DEEP_Decoupled_Semantic_Prompt_Learning_Guiding_and_Embedding_for_Multi-Spectral_Object_Re-Identification.pdf) identifies an accepted TMM 2026 version, DOI `10.1109/TMM.2026.3660160`. Its standard-resolution table supplies the row above; its higher-resolution experiment is separate. It states 60 epochs, a trainable CLIP visual encoder, frozen text encoder, and semantic inversion at inference. MSVR310 excludes same-identity, same-time-span gallery items. The inspected [public default branch](https://github.com/lsh-ahu/DEEP-ReID/tree/cc177306528dd8f8c468de2bb2eeba22978a9fb7) has ten files, consisting of metadata and images. Training/evaluation implementation and exact query/gallery manifests cannot be checked from that inventory.

## PMKD

The [publisher PDF](https://ojs.aaai.org/index.php/AAAI/article/download/38338/42300) is byte-identical to the previously verified author PDF. Tables 1/2 confirm the row above; RGBNT201 R5/R10 are 91.0/92.2. Its other vehicle benchmark is WMVeID863, not MSVR310. It specifies DINOv2, 224x224, batch32 and 50 epochs; total multistage cost remains unclear. The pinned public tree contains only README. `PMKD_PUBLISHER_SOURCE_BINDINGS.json` preserves the first failed transfer and subsequent successful native download. This strengthens source provenance, not experiment performance or equal-budget reproducibility.

## DSGM

[arXiv v1](https://arxiv.org/html/2607.29207v1) Tables I/II supply the row above. The [author record](https://arxiv.org/abs/2607.29207v1) and [official laboratory announcement](https://cmdi.dlut.edu.cn/info/1003/1530.htm) identify TCSVT acceptance; a journal version of record was not inspected. The paper uses cached GPT-4o text and SAM2 masks in training and inference, with 60 epochs for RGBNT201 and 50 for the vehicles.

Pinned public code at `6566f78636440c70c85700a45bbf2f48f18260b1` has a narrower reproducibility boundary:

- [MSVR numeric filtering](https://github.com/zw-absin/DSGM/blob/6566f78636440c70c85700a45bbf2f48f18260b1/utils/metrics.py#L60-L99) removes same identity **and** same `sceneid`, independent of `camid`. The [original benchmark](https://arxiv.org/pdf/2208.00632v2), Sections 4.2-4.3, identifies this field as acquisition time; viewpoint is separate. The [pinned original parser](https://github.com/superlollipop123/Cross-directional-Center-Network-and-MSVR310/blob/0821f289207f0bd367b9296fa4fbdeb9114e4e5a/data/datasets/msvr310.py) and [active filter](https://github.com/superlollipop123/Cross-directional-Center-Network-and-MSVR310/blob/0821f289207f0bd367b9296fa4fbdeb9114e4e5a/data/datasets/eval_reid.py) agree. See `MSVR310_PROTOCOL_SOURCE_BINDINGS.json`; this was already established in handoff Section 41.594. Query precedes gallery with no validation shuffle. Actual file lists and valid-query counts remain unverified.
- All three shipped configurations set `MODEL.DA=False`. [Model output](https://github.com/zw-absin/DSGM/blob/6566f78636440c70c85700a45bbf2f48f18260b1/modeling/make_model.py#L240-L264) puts fused visual output in `LOCAL_v`/`LOCAL`; the [default entry readout](https://github.com/zw-absin/DSGM/blob/6566f78636440c70c85700a45bbf2f48f18260b1/engine/processor.py#L276-L300) returns six global visual/text blocks excluding `LOCAL`. CLI overrides are possible. Enabling DA adds logs, but inference still returns those six blocks; training returns `LOCAL_t` last. This is public-HEAD behavior, not proof of the paper's effective launch or a claim that its reported results are erroneous.
- Static entry constructors disable reranking. External text/mask bytes, generator settings, checkpoints, and the effective paper launch are unavailable in the inspected release. No author code was executed.

## RoDI

The [CVF publisher](https://openaccess.thecvf.com/content/CVPR2026F/html/Li_Rolling_and_Denoising_Rethinking_Dynamic_Modal_Fusion_for_Multi-Modal_Object_CVPRF_2026_paper.html) confirms CVPR 2026 Findings, pp. 6560–6569. Official main PDF Table 1 (PDF page 6) confirms both encoder rows; the author PDF has exactly the historical SHA256. A supplement denoising-step row has a different mAP/R1 pair and is not substituted into the main row. The inspected [repository](https://github.com/lsh-ahu/RoDI/tree/2f38911c49d42d4ca259d440a851b8d77dddccbe) contains only README, paper and poster. Exact filtering, query/gallery manifests, total epochs, seed count, checkpoint selection and reranking remain unverified. CLIP and distilled DINOv3 remain separate resource conditions.

## Evidence and current experiment

`SOURCE_BINDINGS.json` pins primary URLs, canonical dedup identifiers, repository commits, source hashes and retrieval scope. `DSGM_SOURCE_MANIFEST.json` binds 77 locally verified source files; the two inventory JSON files bind the inspected public trees. PDFs, full paper text and author source archives stay in local scratch and are not republished. Source extraction agents were read-only; this is not independent experiment integrity approval.

The current R2 campaign remains `logs/slot_competition_fp32_roles_20261001_r2`: six fresh runs, seed42, full 50 epochs, one official fused-mAP-best checkpoint per run and strict full-gallery replay. Last accepted snapshot remains §41.681 (04:16:51, zero of six terminal results); no interim training scores were read for this check. Existing controller 1394231 and observer 1400686 were verified alive at 04:33:37. The observer remains scheduled for 06:35 CST, then 240 seconds if unfinished; the existing local wait cell 424 reads it at 06:38. Estimated completion is not a result. The research goal is active and unmet.
