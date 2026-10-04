# 角色读取输入梯度边界：RGBNT201完整配对

| 条件 | best轮 | mAP | R1 | R5 | R10 |
|---|---:|---:|---:|---:|---:|
| 原独立global-only | 8 | 74.2967 | 78.9474 | 88.2775 | 91.8660 |
| 原semantic | 7 | 71.8981 | 74.4019 | 84.3301 | 90.1914 |
| 原native | 20 | 72.1273 | 75.1196 | 85.1675 | 89.3541 |
| 读取输入detach semantic | 18 | 72.7798 | 76.9139 | 84.9282 | 89.2344 |
| 读取输入detach native | 8 | 69.4305 | 72.1292 | 85.2871 | 90.7895 |

| 配对差值 | ΔmAP | ΔR1 | ΔR5 | ΔR10 |
|---|---:|---:|---:|---:|
| 新semantic−原semantic | +0.8817 | +2.5120 | +0.5981 | -0.9569 |
| 新native−原native | -2.6968 | -2.9904 | +0.1196 | +1.4354 |
| 新native−新semantic | -3.3494 | -4.7847 | +0.3588 | +1.5550 |
| 新native−原独立global-only | -4.8662 | -6.8182 | -2.9904 | -1.0766 |

1. **相同读取梯度干预未形成通用收益。** semantic对原控制的mAP/R1改善，但native的两项主指标均下降；不能把首项正结果写成角色读取detach已解决整个系统。该入口同时截断stages/context/shared_global向前端的读取梯度，共享适配平均写回及global直接路径仍参与训练，不等于全局冻结或已定位唯一梯度冲突。
2. **native的R5/R10正变化没有抵消主指标损失。** 同一mAP-best的完整四项均报告；不从其他epoch挑选CMC，不用较后位CMC改善覆盖mAP/Rank-1下降。两新端仍均低于匹配独立global-only。
3. **比较约束已核验。** 对应原variant的初始化、容量、配置及实际批次顺序保持；新semantic/native的2649条actual order也逐字节相同。322源码和61个原控制依赖SHA未变，每端仅一份正式best，首次严格加载/评分和实存模型及距离SHA通过。旧控制为历史同种子配对，不是新训练种子重复。
4. **后期回落仍存在。** native E8的mAP-best为69.4305，末轮为65.9114，差3.5190。完整50轮保持原预算；step范数仅是训练活动统计。固定best的g/c/h/f分解仍待六端结束，尚不能将fused损失全部归于global或细节修正。

Single seed historical-control intervention: semantic mAP/R1 improved while native mAP/R1 declined. Rank5/10 gains do not rescue the negative main metrics. No universal gradient-conflict repair or protected-global claim; same-model g/c decomposition and other datasets pending. Current six-endpoint final CPU report not invoked. Scientific Goal active/unmet.
