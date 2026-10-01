# RoDI primary-source readback

RoDI is a verified current paper reference (Li, Huang, Zheng, Tang and He; CVPR 2026 Findings, pp. 6560-6569). It is **not a verified public executable baseline** at the checked repository HEAD, and its paper scores have not been reproduced here.

## Sources and identity

- Official repository: [2f38911c49d42d4ca259d440a851b8d77dddccbe](https://github.com/lsh-ahu/RoDI/commit/2f38911c49d42d4ca259d440a851b8d77dddccbe), committed 2026-06-15. Its complete non-truncated tree contains exactly three blobs: README.md, assets/RoDI.pdf, assets/poster.png. No executable implementation, parser, evaluator, configuration or checkpoint is present. The six reachable main commits checked also contain only README/PDF/poster resources. One branch, no tags or releases were returned.
- [Official CVF paper page](https://openaccess.thecvf.com/content/CVPR2026F/html/Li_Rolling_and_Denoising_Rethinking_Dynamic_Modal_Fusion_for_Multi-Modal_Object_CVPRF_2026_paper.html); [published main PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Li_Rolling_and_Denoising_Rethinking_Dynamic_Modal_Fusion_for_Multi-Modal_Object_CVPRF_2026_paper.pdf); [supplement](https://openaccess.thecvf.com/content/CVPR2026F/supplemental/Li_Rolling_and_Denoising_CVPRF_2026_supplemental.pdf). The final Table 1 was read as text and visually inspected on physical page 6 / printed p. 6565.
- Canonical arXiv ID / DOI: **unidentified**, rather than invented. Official metadata has neither; the exact-title-phrase arXiv API returned 0 entries, and five Crossref title-query results did not match. This does not prove that no ID exists. Source-based dedup key: `cvf:CVPR2026F/Li_Rolling_and_Denoising_Rethinking_Dynamic_Modal_Fusion_for_Multi-Modal_Object_CVPRF_2026`.

## Primary Table 1 scores (percent; paper reported only)

| Variant | RGBNT201 mAP / R1 / R5 / R10 | MSVR310 mAP / R1 | RGBNT100 mAP / R1 |
|---|---|---|---|
| CLIP ViT-B/16, RoDI* | 84.1 / 87.2 / 92.0 / 93.2 | 64.1 / 77.2 | 88.5 / 97.6 |
| DINOv3 distilled ViT-B/16, RoDI† | 85.3 / 87.9 / 93.0 / 94.8 | 71.8 / 84.8 | 89.0 / 99.1 |

These are separate backbone settings. They must not be merged into one result or described as locally reproduced. The CLIP MSVR310 R1 is below UGG-ReID's reported 78.0 in the same table, so even within this table RoDI does not win every CLIP metric.

## Protocol comparison and limits

The paper states that it follows previous dataset protocols [49-51] and reports mAP/CMC. Its dataset totals (4787 / 2087 / 17250) match the sums of this project's train and gallery counts; that establishes neither exact split lists nor historical byte identity.

The bound local sources implement RGBNT201 train_171/test/test = 3951/836/836, RGBNT100 bounding_box_train/query/bounding_box_test = 8675/1715/8575, and MSVR310 bounding_box_train/query3/bounding_box_test = 1032/591/1055. Local retrieval removes only same-identity/same-camera entries for RGBNT201/RGBNT100 and same-identity/same-scene entries for MSVR310, retaining different identities. AP averages retained-positive precision; CMC uses the first positive. The local correspondence runner uses 1536-wide output, float32 L2-normalized squared Euclidean distance, mAP/Rank-1/5/10 percentages, and explicitly sets reranking=false. Exact local SHA bindings and function lines are in SOURCE_CHECK.json.

RoDI's absent implementation leaves its actual dataset parsing, filter, distance, normalization, retrieval feature before/after BN, reranking flag, checkpoint/config identifiers and exact image lists unresolved. Method-internal cosine evidence and L2 norms are not evidence of retrieval-distance normalization. The paper identifies CLIP and DINOv3 distilled encoders, but the checked repo supplies neither implementation nor model resources for either. Main §3.5 says inference omits subjective-opinion metrics and modality rolling; that does not identify the exact evaluator feature path.

## Byte bindings and execution scope

- Official main PDF SHA256: `745393be9fd1e46e1d6b2804c20c6b82a31498725ea86e169ae7f7885c393fa6`.
- Official supplement SHA256: `903279991b17b8f9650a27b06ac2bbd02b999c20440bc2ffe2280d9e3394bbf9`.
- GitHub author PDF SHA256: `b816237dbbcccda47dbf02b19c9de9816bce05b4f40723ae88ef1286ff663d86`; Git blob `dcfca845758fdb072688c5077679f10ee6f92cc3`. It is a distinct PDF from the published main file.
- Local source HEAD when bound: `34e3f93280739b6fdd24b8730b46ed0d3604a760`. The compared six files were read and SHA-verified only.

All author text and cached source evidence remain private in this directory. No project files were changed, no author model/training/evaluation was executed, and no remote server was accessed. This is source readback, not an independent scientific audit or reproduction. It supports citing RoDI's primary published results while leaving exact runtime comparability unqualified.
