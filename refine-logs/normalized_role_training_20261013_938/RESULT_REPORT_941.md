# 角色分类入口单项完整结果

201 seed42，同一mAP-best epoch8，正式50轮/2649步。

| 指标 | mAP | R1 | R5 | R10 |
|---|---:|---:|---:|---:|
| candidate | 74.2875 | 78.8278 | 88.1579 | 91.5072 |

| 对照 | ΔmAP | ΔR1 | 修复 | 新增错误 | 身份宏APΔ | 推进 |
|---|---:|---:|---:|---:|---:|---|
| raw_semantic | -0.0874 | +0.0000 | 4 | 4 | -0.1070 | False |
| raw_global_only | -0.0092 | -0.1196 | 2 | 3 | -0.0398 | False |

First fixed seed42 development comparison on consumed official benchmarks. Primary raw semantic and independent shared-adapter global controls reuse sealed saved distances. Existing camera/gallery rules intact; no NN, new ranking rule, seed average, test centering, tuning or causal claim. Identity bootstrap in reused compare is fixed-model descriptive only, not training-seed uncertainty.
