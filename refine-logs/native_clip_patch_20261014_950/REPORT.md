# 原CLIP patch与辅助dense读取：固定权重

| 来源 | mean mAP/R1 | MaxSim mAP/R1 | ΔmAP | 修复/新增错误 |
|---|---:|---:|---:|---:|
| native_projected | 55.4165/59.9282 | 60.3528/58.7321 | +4.9363 | 91/101 |
| attention_only_qk | 72.8013/75.1196 | 71.9707/74.8804 | -0.8305 | 18/20 |
| attention_only_qq | 55.6361/55.8612 | 61.3513/62.3206 | +5.7152 | 105/51 |

One frozen best backbone query/gallery pass; original global path untouched. Native projected128 patches versus auxiliary actual-QK-output and QQ-output without final residual/FFN; the latter also bypasses the last external role-adapter snapshots. Larger512-width/128-token bags differ from prior128-width/16-slot evidence, so no unique bottleneck/processing cause is established. Borrowed dense-reading principle, not full ClearCLIP segmentation reproduction. Scores are local-only diagnostics, no global fusion coefficient, fitting, test centering, new formal method score or teacher assumption. Feature tensors are hashed in memory, not saved; inputs/weights/source and resulting distances are retained for replay.
