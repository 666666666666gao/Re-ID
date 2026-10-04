# MSVR310 original full training pair

Original completed seed42 results on consumed official benchmarks; no smoothing, reranking, epoch reselection, new inference or optimizer updates. Independent global-only is separately trained, not the same model global trajectory. Author objectives combine classification and soft-triplet terms; their separate activity is not logged. Equal mean global loss/norm logs are aggregate facts, not proof of identical full model states or per-query global features. Norm ratios are not causal contribution estimates.

Both original best epochs are 38. Best-to-final mAP declines are 0.060976 for semantic and 0.067221 for native. These curves show modest late declines; they do not support a large late-collapse claim for this pair.
