# 下一主比较：MSVR310来源统计扰动

状态：DIRECTION_SELECTED_READINESS_CHECKED_NOT_TRAINING_REGISTERED，2026-09-07。
V29完整联合诊断完成后，唯一下一训练问题确定为：V27耦合来源统计扰动是否能在MSVR310原三角色上带来配对、跨身份分布的检索改善。
理由：V27在RGBNT201有全三fold/三角色正收益；V28/V29自由或有界联合头尚未稳定改善，来源已饱和，joint大部分变化表现为已有相似度缩放。优先建立另一数据集的机制证据，降低对反复开发的21个RGBNT201身份的依赖。这里不宣称提出新算法。

## 已核实输入和边界

- 三个MSVR310固定Signal epoch50 checkpoint现存且完整SHA再次匹配原B0摘要；不重训baseline。
- 原三fold source103/103/104身份，672/683/709记录；heldout完整图库360/349/323，合法query210/207/183，共600query和60query身份。95单scene干扰身份仍保留。
- 原角色完整训练元数据260batch/fold、20epoch、共780步已逐条检查；全部source记录曝光，最少camera数6/7/7，各batch都允许跨camera供体。
- 按原V27固定seed42/fold/step的独立NumPy计划，active批129/127/137；原真实source重复记录正关系曝光21836/20864/20882，必须保留并报告。
- 三fold跨scene有序正关系30232/30196/30650，总91078/349440，约26.064%；cross-camera与cross-scene是不同字段，不混称同一种覆盖。
- 固定供体均跨camera，其中跨scene13927/13888/13828次，同身份供体1613/1716/1719次。原V27未要求供体不同身份，不能暗中追加标签过滤。
- 以上是对既有训练元数据的完整检查，不是新loader像素轨迹重放；正式训练必须匹配固定曝光并留下新输入回执。

## 下一实现必须具体完成

1. 新的车辆style接口使用768×8×16 stem，原CNN/Mamba仍按8×16网格。原V27行人硬编码16×8模块保持冻结；复用统计混合公式，同模态统计、三模态共享供体/系数，anchor/reference接受同一扰动。
2. 每fold在原Signal上新初始化原V8角色，两端seed42、整个初始化状态和采样/训练预算匹配。两端均执行同样额外视觉重编码，只有是否作用统计混合不同。无V23/V24、联合Mamba、几何修正、Router、记忆或新loss。
3. 保留原B64/K8、128×256、20epoch、AdamW和七组ID/Triplet；预期每端780更新，两端1560，真实记录为准。不复用M0权重，不用旧单端终点冒充本轮匹配控制。
4. 训练前完整source-only工程门应核对Signal原路径/被冻结状态、模态耦合、anchor/reference匹配、active与inactive行为、实际全部参数梯度和固定批次拟合。沿用已验证exact_signal_forward来恢复车辆B0的精确推理执行；不能重现已修复的SIM差异。
5. 全部三fold两端固定终点完整检索，使用原scene过滤和完整图库。配对扰动收益与candidate相对Signal的既有车辆晋级条件分别登记/报告；不因其中一组更容易通过而替代另一组。门槛、CPU全量核验、源文件/初始化/输出绑定须在运行前完整确定。
6. 正式三处同步后才启动持久任务。当前仅方向和输入准备完成：新训练程序、最终合同及工程门尚未执行。下一步直接完成这些内容，不重复已完成V29诊断。

## 证据

来源元数据完整保存：evidence/msvr310_style_readiness_20260907/msvr310_style_source_readiness_20260907.json。
三个当前Signal全文件SHA：evidence/msvr310_style_readiness_20260907/msvr310_style_remote_readiness_20260907.json。
角色原实现tools/train_msvr310_trifusion_oof.py；过滤tools/train_msvr310_signal_oof.py的scene_scores；精确推理tools/msvr310_exact_signal_inference.py。
这不是零样本迁移、官方成绩或新SOTA主张。三数据集长期Goal保持。
