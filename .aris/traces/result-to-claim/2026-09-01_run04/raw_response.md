# Independent reviewer response

1. `claim_supported: no`
2. Supported: exact Signal preservation, strict reload, official-test access zero, and fused improving the exact baseline by `0.7212 mAP` on the frozen RGBNT201 development split.
3. Unsupported: fused beating every expert because CNN reaches `59.1022 mAP` versus fused `58.7321`; reaching 65 mAP; SOTA; or attributing three validated innovations. The later train-loss/dev pattern is consistent with overfitting.
4. Missing evidence: a V6-specific independent integrity audit and a passing main run.
5. Revised defensible claim: V6 is exact-Signal-preserving and slightly improves the baseline on RGBNT201 dev, but it does not establish collaborative superiority or SOTA readiness.
6. Next allowed experiment: one corrected main-only seed42 train/dev V7 with official-test access zero and no ablation or multi-seed run. The primary target is routing alignment; generalization is secondary.
7. Confidence: high. Integrity status: provisional.

Routing evidence: normalized entropy is `0.9744` with standard deviation `0.0084`; the strongest CNN expert receives only `0.228–0.245` mean weight, while weaker Transformer receives `0.325–0.367` and Mamba `0.405–0.430`. The epoch8-to-epoch60 decline is a completed-run generalization signal, not unfinished training.
