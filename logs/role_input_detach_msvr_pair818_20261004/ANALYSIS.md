# 角色读取梯度边界：MSVR310完整配对

| 条件 | best轮 | mAP | R1 | R5 | R10 |
|---|---:|---:|---:|---:|---:|
| 原独立global-only | 38 | 50.5421 | 68.0203 | 80.5415 | 85.4484 |
| 原semantic | 49 | 50.9636 | 69.2047 | 80.7107 | 86.1252 |
| 原native | 38 | 50.6755 | 68.6971 | 81.5567 | 85.9560 |
| 读取输入detach semantic | 49 | 50.7851 | 68.8663 | 81.5567 | 85.4484 |
| 读取输入detach native | 38 | 51.1388 | 69.8816 | 81.2183 | 85.9560 |

| 配对差值 | ΔmAP | ΔR1 | ΔR5 | ΔR10 |
|---|---:|---:|---:|---:|
| 新semantic−原semantic | -0.1785 | -0.3384 | +0.8460 | -0.6768 |
| 新native−原native | +0.4632 | +1.1844 | -0.3384 | +0.0000 |
| 新native−新semantic | +0.3537 | +1.0152 | -0.3384 | +0.5076 |
| 新native−原独立global-only | +0.5966 | +1.8613 | +0.6768 | +0.5076 |

1. **区分干预配对和细节配对。** native相对原native的预登记推进条件为False；新native相对新semantic的同起点细节推进条件为False。两者不是同一个问题，0.5点门槛不是统计显著性。
2. **不外推201或MSVR到全部条件。** 201新native主指标明显下降，MSVR新semantic也未增加主指标。现有结果不能支持通用读取detach修复、已定位唯一梯度冲突或global严格保护。100仍须完成两端50轮及首次严格评价。
3. **比较合同保持。** 两端706步actual order相同，与对应原控制一致；初始化、容量、作者配方、322源码及61原依赖SHA核验，正式每端只有一份mAP-best，其余CMC跟随这份权重。自身M0依赖仍保留给最后一次原CPU报告。
4. **固定best分解仅准备。** 后续计划复用已完成原诊断的提取/计分逻辑，经实际detach入口配置模型，待全六端和原CPU报告完成后再封存九行输入并执行。训练step范数尚不能替代同模型g/c/h/f检索。

Two complete dataset pairs, RGBNT100 endpoints pending. Historical same-seed controls, not full-flow multi-seed or independent reproduction. New native minus new semantic differs from registered gradient intervention versus original native. No universal gradient-conflict repair or strict global preservation. Final once-only CPU report and fixed-best decomposition pending. Only26GPU0/1, no power/temp or parity repair. Scientific Goal active/unmet.
