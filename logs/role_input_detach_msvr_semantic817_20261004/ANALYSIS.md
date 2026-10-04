# 角色读取梯度边界：MSVR310 semantic完整结果

| 条件 | best轮 | mAP | R1 | R5 | R10 |
|---|---:|---:|---:|---:|---:|
| 原独立global-only | 38 | 50.5421 | 68.0203 | 80.5415 | 85.4484 |
| 原semantic | 49 | 50.9636 | 69.2047 | 80.7107 | 86.1252 |
| 读取输入detach semantic | 49 | 50.7851 | 68.8663 | 81.5567 | 85.4484 |

| 配对差值 | ΔmAP | ΔR1 | ΔR5 | ΔR10 |
|---|---:|---:|---:|---:|
| 新semantic−原semantic | -0.1785 | -0.3384 | +0.8460 | -0.6768 |
| 新semantic−原独立global-only | +0.2430 | +0.8460 | +1.0152 | +0.0000 |

1. **MSVR semantic未显示该干预的主指标增益。** 相对原semantic，mAP和Rank-1均下降；相对历史独立global略高不能替代干预配对。预登记推进条件未达到，不等同统计上证明无效。
2. **训练与评价链闭合。** 完整50轮、706步、唯一mAP-best及首次严格重载通过；对应初始化、容量、配方和实际批次顺序保持，322项源码和61项原控制依赖SHA未变。原九端结果和已退役M0状态未修改。
3. **不从范数推导检索贡献。** step中的修正/全局幅度是训练批次观察；尚无新模型固定best的g/c/h/f检索分解，不能据此证明global被保护或定位梯度冲突。
4. **继续原队列而不救分。** MSVR native自身M0已通过8步更新、299项有限非零梯度、三BN各8、14项细节参数参与检查、严格重载差0；随后进入fresh50。其性能和RGBNT100两端仍待完成，不修改LR、gain、seed、读取器或损失。

Historical same-seed control intervention, not full-flow multi-seed or independent reproduction. MSVR semantic mAP/R1 declined relative to original semantic, although above the historical independent global. No general role-input detach repair or protected-global claim. Current once-only final report and fixed-best g/c decomposition pending. Only26GPU0/1; no power/temp control or parity repair. Scientific Goal active/unmet.
