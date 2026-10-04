# Original RGBNT100 semantic training curves

Descriptive curves from one completed seed42 experiment on consumed official benchmarks. No smoothing, reranking, epoch reselection or model execution. The independent global-only reference is a separately trained control, not this model's global trajectory. Head objectives combine author identity/soft-triplet terms; they do not show individual loss activity. Norm ratios are training-batch mean per-sample ratios, not a causal contribution measure.

The selected checkpoint remains epoch 5. Training is complete and first strict evaluation is accepted. The 1.8082-point best-to-final mAP drop accompanies lower identity objectives; the curves do not identify a unique cause.
