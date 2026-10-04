# 读取梯度边界：完整六端结果账

| 数据集 | 条件 | best轮 | mAP | R1 | R5 | R10 | best−末轮mAP |
|---|---|---:|---:|---:|---:|---:|---:|
| RGBNT201 | semantic | 18 | 72.7798 | 76.9139 | 84.9282 | 89.2344 | 2.7601 |
| RGBNT201 | native | 8 | 69.4305 | 72.1292 | 85.2871 | 90.7895 | 3.5190 |
| MSVR310 | semantic | 49 | 50.7851 | 68.8663 | 81.5567 | 85.4484 | 0.0158 |
| MSVR310 | native | 38 | 51.1388 | 69.8816 | 81.2183 | 85.9560 | 0.1684 |
| RGBNT100 | semantic | 5 | 81.9418 | 94.7522 | 95.2770 | 95.6851 | 3.7152 |
| RGBNT100 | native | 5 | 82.5124 | 95.8017 | 96.7930 | 97.0845 | 3.1849 |

| 干预：新−对应原变体 | ΔmAP | ΔR1 | 首位修复/新增错误 | 身份宏平均ΔAP | 事前推进条件 |
|---|---:|---:|---:|---:|---|
| RGBNT201 semantic | +0.8817 | +2.5120 | 72/51 | +0.9975 | 满足 |
| RGBNT201 native | -2.6968 | -2.9904 | 54/79 | -2.6073 | 未满足 |
| MSVR310 semantic | -0.1785 | -0.3384 | 5/7 | +0.0171 | 未满足 |
| MSVR310 native | +0.4632 | +1.1844 | 12/5 | +0.5933 | 未满足 |
| RGBNT100 semantic | -1.4977 | -1.3411 | 25/48 | -1.7615 | 未满足 |
| RGBNT100 native | -0.0486 | +0.2915 | 47/42 | -1.4135 | 未满足 |

| 同干预native−semantic（描述性） | ΔmAP | ΔR1 | 首位修复/新增错误 | 身份宏平均ΔAP |
|---|---:|---:|---:|---:|
| RGBNT201 | -3.3494 | -4.7847 | 50/90 | -3.5036 |
| MSVR310 | +0.3537 | +1.0152 | 14/8 | +0.2938 |
| RGBNT100 | +0.5706 | +1.0496 | 49/31 | +0.1830 |

Original once-only CPU report retained unchanged. All six endpoints and all queries included; no new checkpoint selection, model/scorer call, seed, training, gain search or acceptance rule. Fixed-model identity statistics do not establish training-seed stability. New fixed-best g/c/h/f diagnosis remains separate; scientific Goal active/unmet.
