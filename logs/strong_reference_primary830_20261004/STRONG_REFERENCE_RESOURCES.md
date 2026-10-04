# Original-source strong reference resources, 2026-10-04

Units are percentage points. These are paper reports, not matched local reproductions or an exhaustive SOTA ranking.

| Paper / pretraining | RGBNT201 mAP/R1/R5/R10 | RGBNT100 mAP/R1 | MSVR310 mAP/R1 |
|---|---|---|---|
| DEEP / CLIP visual + frozen text | 79.6/84.2/89.4/91.5 | 88.5/97.6 | 66.0/82.1 |
| RoDI / CLIP | 84.1/87.2/92.0/93.2 | 88.5/97.6 | 64.1/77.2 |
| RoDI / DINOv3 | 85.3/87.9/93.0/94.8 | 89.0/99.1 | 71.8/84.8 |
| PMKD / DINOv2 | 84.7/88.9/91.0/92.2 | 91.6/98.0 | Not reported |

DEEP: [author accepted PDF](https://aihuazheng.github.io/publications/pdf/2025/2025-DEEP_Decoupled_Semantic_Prompt_Learning_Guiding_and_Embedding_for_Multi-Spectral_Object_Re-Identification.pdf), TablesI/IV and IV.B. The frozen text branch and semantic prompts are resources; concrete annotated text labels are not required by this paper. Training60epochs; visual weights trainable. MSVR filters same-ID/same-time-span. [Checked repository](https://github.com/lsh-ahu/DEEP-ReID) currently contains README/license/assets, no training implementation.

RoDI: [author PDF](https://github.com/lsh-ahu/RoDI/blob/main/assets/RoDI.pdf), Table1 and4.2. Both backbones ViT-B/16; B64/K8, Adam3.5e-4 and10-epoch warmup. Total epochs remain unknown. Inference omits rolling/opinion metrics according to3.5; diffusion and pseudo-feature fusion are described, but no executable path is available. CLIP/DINO rows are not interchangeable. [Checked repository](https://github.com/lsh-ahu/RoDI) contains README,PDF,poster. CVF HTML retrieval403; this check establishes author-PDF tables, not byte identity with CVF edition.

PMKD: [official AAAI PDF](https://ojs.aaai.org/index.php/AAAI/article/download/38338/42300), Tables1/2, pages3/5. DINOv2,224x224,B32/K8,Adam4.5e-5; implementation states50epochs. The pipeline includes source training and two fresh-target distillation stages; total training cost is not established. MSVR is absent from the evaluated dataset list. [Checked repository](https://github.com/moonaricc/PMKD) contains only README.

The tree JSON files pin current public main content using immutable tree SHAs (not commit SHAs). No raw PDF or full paper text is published in this packet. Local previews of PMKD/RoDI tables were visually checked. Current model training and the330scientific-source seal are untouched.
