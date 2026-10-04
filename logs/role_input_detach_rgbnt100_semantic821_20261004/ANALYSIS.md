# RGBNT100 semantic完整50轮：读取detach四项指标下降

| 条件 | best轮 | mAP | R1 | R5 | R10 |
|---|---:|---:|---:|---:|---:|
| 原独立global-only | 7 | 84.5338 | 96.6181 | 97.3178 | 97.9592 |
| 原semantic | 5 | 83.4395 | 96.0933 | 96.6764 | 96.9679 |
| 读取输入detach semantic | 5 | 81.9418 | 94.7522 | 95.2770 | 95.6851 |

| 配对差值 | ΔmAP | ΔR1 | ΔR5 | ΔR10 |
|---|---:|---:|---:|---:|
| 新semantic−原semantic | -1.4977 | -1.3411 | -1.3994 | -1.2828 |
| 新semantic−原独立global-only | -2.5920 | -1.8659 | -2.0408 | -2.2741 |

三个semantic端均已完成。相对各自原semantic，201的mAP/R1提高0.8817/2.5120，MSVR下降0.1785/0.3384，100下降1.4977/1.3411。事前推进条件只在201满足；这些单种子配对不支持三集通用修复，也不等同统计显著性结论。
正式完成5/6、全部自身M0完成6/6。最后RGBNT100 native自身8步M0通过：299/299张量有限非零梯度、三BN各8、14项细节参数纳入检查、重载差0，随后按原队列fresh50。没有重启、调参或追加模块。
训练批次scaled c/g幅度只描述训练过程，不能代替固定best的检索贡献或证明global受保护。读入口停止梯度不改变最终融合loss通过global训练共享参数的路径。
已核对source322、历史61依赖、共同初始化、真实批次顺序和唯一正式best；原一次CPU报告与新固定best g/c/h/f分解仍待六端终态。

Same-seed intervention, not full-flow multi-seed or independent reproduction. RGBNT100 semantic all four metrics declined relative to original semantic and independent global. Detach removes only the direct role-read derivative; fused loss still trains global. No universal gradient-conflict cause. Current once-only CPU report and fixed-best decomposition pending. Only26GPU0/1; no power/temp control or queries; scientific Goal active/unmet.
