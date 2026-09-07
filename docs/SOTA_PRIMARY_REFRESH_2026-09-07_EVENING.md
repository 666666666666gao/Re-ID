# Primary benchmark refresh, 2026-09-07 evening

This is a literature verification record, not a new experiment, a complete leaderboard, or a change to the registered MSVR310 Q1 gates. All scores below are published mAP / Rank-1 percentages. Internal Q1 scores must not be subtracted from these official-benchmark reports.

## RoDI: author PDF independently retrieved again

[Author repository](https://github.com/lsh-ahu/RoDI) links the [author PDF](https://github.com/lsh-ahu/RoDI/blob/main/assets/RoDI.pdf). The downloaded PDF has 13 pages, 5,636,720 bytes, SHA256 `b816237dbbcccda47dbf02b19c9de9816bce05b4f40723ae88ef1286ff663d86`. Table 1 and implementation details are on PDF page 6.

| Variant | RGBNT201 | MSVR310 | RGBNT100 | WMVeID863 |
|---|---:|---:|---:|---:|
| RoDI, CLIP ViT-B/16 | 84.1 / 87.2 | 64.1 / 77.2 | 88.5 / 97.6 | 73.9 / 81.9 |
| RoDI, distilled DINOv3 ViT-B/16 | 85.3 / 87.9 | 71.8 / 84.8 | 89.0 / 99.1 | 74.2 / 82.2 |

The paper reports B64 with eight images per identity. Encoder resources and image sizes differ between variants. The repository inventory shown this evening contains README and assets; a paper link is not evidence of a complete runnable training release. The CVF HTML request returned 403; the author PDF above was actually retrieved and parsed successfully. The initial extraction command failed only when printing Unicode through Windows GBK, after the PDF and JSON extraction had already been written; the saved extraction was then read successfully with UTF-8.

## FUSE: a newly checked primary reference

[arXiv v1](https://arxiv.org/html/2606.20044v1), submitted 2026-06-18, Tables 1-2 and Section 4.1:

| Method | RGBNT201 | MSVR310 | RGBNT100 |
|---|---:|---:|---:|
| FUSE | 81.4 / 86.1 | 50.1 / 65.7 | 88.5 / 96.9 |

The encoder starts from CLIP and receives updates at learning rate 5e-6. The paper specifies 45 epochs for RGBNT201/RGBNT100 and 50 for MSVR310, with B128/B128/B64 respectively. Its abstract's +9.1 mAP / +9.5 Rank-1 compares RGBNT201 with TOP-ReID, not with every current method. The arXiv author comment says accepted in ICML 2026; this refresh did not independently check the venue proceedings. These results do not replace the stronger resource-qualified references already recorded. No FUSE code, pretrained weights or experiment was installed or run.

## PEFT-BoA: official table checked again

[AAAI official PDF](https://ojs.aaai.org/index.php/AAAI/article/view/37537/41499), Tables 1-2 on PDF page 6: RGBNT201 82.7 / 86.1, RGBNT100 85.1 / 97.8, MSVR310 57.4 / 74.5. The model uses frozen CLIP and reports 6.62M trainable parameters. Use the actual table entries when later prose or another paper's transcription differs.

PMKD's earlier complete author-PDF verification remains in [the dedicated record](PMKD_AUTHOR_PDF_VERIFICATION_2026-09-07.md): RGBNT100 91.6 / 98.0 uses DINOv2; that verified paper does not report MSVR310. This evening its AAAI landing page was checked, but the large PDF was not downloaded again. STMI's generated text/masks and ProxyTTT's test-time adaptation remain separately qualified in [the preceding refresh](SOTA_PRIMARY_REFRESH_2026-09-06.md).
