# 预训练value增量读取：两端完整结果

| 条件 | best | mAP | R1 | R5 | R10 |
|---|---:|---:|---:|---:|---:|
| raw_semantic | 8 | 74.3749 | 78.8278 | 88.3971 | 91.7464 |
| raw_global_only | 8 | 74.2967 | 78.9474 | 88.2775 | 91.8660 |
| semantic_reference | 8 | 74.3111 | 78.9474 | 88.1579 | 91.8660 |
| uniform_reference | 8 | 74.2974 | 78.8278 | 88.1579 | 91.8660 |

| 对照→候选 | ΔmAP | ΔR1 | 修复 | 新增错误 | 身份宏APΔ | 推进 |
|---|---:|---:|---:|---:|---:|---|
| uniform_reference→semantic_reference | +0.0137 | +0.1196 | 1 | 0 | +0.0155 | False |
| raw_semantic→semantic_reference | -0.0637 | +0.1196 | 4 | 3 | -0.0439 | False |
| raw_semantic→uniform_reference | -0.0774 | +0.0000 | 4 | 4 | -0.0594 | False |
| raw_global_only→semantic_reference | +0.0145 | +0.0000 | 2 | 2 | +0.0233 | False |
| raw_global_only→uniform_reference | +0.0008 | -0.1196 | 1 | 2 | +0.0078 | False |

First fixed seed42 consumed-benchmark development comparison. Two same-parameter value reading references, semantic versus uniform; not a novelty proof. Value source, readout and parameter count differ from old semantic. Original camera/gallery rules; no NN/new ranking/test centering/tuning. Fixed-model identity bootstrap is descriptive, not training-seed uncertainty.
