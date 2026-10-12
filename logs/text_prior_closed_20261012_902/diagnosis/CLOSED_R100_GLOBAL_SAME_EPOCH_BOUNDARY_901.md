# RGBNT100：同轮次 global 比较的边界

独立 global-only 的 mAP-best 在第7轮，mAP为84.5337842260。当前 pretrained 文本模型按 fused mAP-best 选择第5轮，其 own-global mAP为84.0835133277。

已经另外读取并逐字节核验封存的独立 global-only 全50轮训练记录：第5轮 global mAP为84.3692249991。因此当前 own-global 与独立 global 的差异并非全部来自 epoch 选择。

| 比较 | mAP差 |
|---|---:|
| 当前文本模型 own-global E5 − 独立 global E5 | -0.2857116714 |
| 独立 global E5 − 其自身 best E7 | -0.1645592269 |
| 当前文本模型 own-global E5 − 独立 global best E7 | -0.4502708983 |

这是分数的算术分解，不是独立因果分解。职责分离限定了损失的梯度路径，不能据此声称不同训练的全局权重、随机过程或检索轨迹逐位相同。当前同轮次差异也没有唯一定位到文本预训练、数值实现、增强或任何其他因素。需要 raw-semantic 与新模型在相同轮次、相同输入和物理权重上直接对照，才能进一步归因；没有补跑 NN 或改验收门槛。

材料：runtime/CLOSED_R100_GLOBAL_EPOCH5_COMPARISON_901.json、runtime/CLOSED_R100_GLOBAL_EPOCH_CONTROL_901。三份文本已核验 SHA；输入 seal 2dd7dedfd4a6460bd65c9429c7face2d6148afd4c6e053cc8e9c8e641ec2f1ea 不变。采集仅CPU/标准库读取历史封存文本，无当前训练进度查询、模型构造或forward。此前的 fresh partial audit 未覆盖本次新采集材料。
