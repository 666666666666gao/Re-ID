# Absolute pretrained value: full terminal report

| Condition | Epoch | mAP | R1 | R5 | R10 |
|---|---:|---:|---:|---:|---:|
| raw_semantic | 8 | 74.3749 | 78.8278 | 88.3971 | 91.7464 |
| raw_global_only | 8 | 74.2967 | 78.9474 | 88.2775 | 91.8660 |
| uniform_reference | 8 | 74.2974 | 78.8278 | 88.1579 | 91.8660 |
| absolute_reading | 8 | 74.6337 | 79.0670 | 88.2775 | 91.7464 |

| Control | Delta mAP | Delta R1 | Repairs | New errors | Advance |
|---|---:|---:|---:|---:|---|
| uniform_reference | +0.3363 | +0.2392 | 5 | 3 | False |
| raw_global_only | +0.3371 | +0.1196 | 4 | 3 | False |
| raw_semantic | +0.2589 | +0.2392 | 5 | 3 | False |

One fixed seed42 absolute-value condition versus an already completed same-state/parameter/recipe differential control. Original bests/receipts/distances rehashed; the retired old M0 probe is not replayed or reverified. Full legal gallery/camera rules. This is benchmark-consumed development evidence, not unbiased SOTA or an original-mechanism proof.
