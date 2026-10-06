# RoDI primary resource recheck, 2026-10-06

Source: [author PDF](https://github.com/lsh-ahu/RoDI/blob/2f38911c49d42d4ca259d440a851b8d77dddccbe/assets/RoDI.pdf), SHA256 `b816237dbbcccda47dbf02b19c9de9816bce05b4f40723ae88ef1286ff663d86`. Current complete tree remains README/PDF/poster only; no executable training/evaluation or weights. Primary PDF text and page7 table rows inspected.

Table2 RGBNT201: CLIP baseline76.0/77.6 → EDFL79.3/83.1 → +MSR82.7/85.9 → +LMD84.1/87.2; full delta8.1mAP/9.6R1. DINOv3 baseline81.0/83.3 →82.3/84.2 →84.0/85.2 →85.3/87.9; full delta4.3/4.6. These are source-paper results, not our reproduction.

Section4.2 lists separate CLIP ViT-B/16 and distilled DINOv3 ViT-B/16 backbones, rectangular CLIP versus224² DINO inputs, B64/K8, Adam, warmup10. Distilled labels the DINO encoder; no extra ReID teacher is described, and unavailable code prevents proving an implementation-wide absence. Total epochs, exact optimizer groups/update boundary and complete original gallery/filter lists remain unresolved.

Dataset counts4787/2087/17250 equal our historical train+gallery counts3951+836/1032+1055/8675+8575. This arithmetic is consistent; it does not prove bytewise query-as-gallery-subset or full protocol equivalence. Do not infer mismatch by adding queries again.

All external source reading was local/read-only. Original NN/timer/source/weights unchanged. Keep downloaded PDF/full extracted text private; future publication uses this curated note and hashes only. No new SOTA claim.
