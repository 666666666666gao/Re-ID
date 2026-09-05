# V22 共同初始化的完整图库只读诊断

本计划在读取任何共同初始化的完整图库结果前固定。V22 的全部六端训练已终态，
原始终态 SHA b8cd7db81efc3827a91d165d47e001073785420baf8ddfc8507a2eead9c3d6a3；
本诊断不改变 V22 科学门、不赋予 D1/dev/official 资格、不选择新部署 checkpoint。

需要回答的事实问题是：在完全相同的完整图库下，当前两个终态相对共同初始化
发生了什么变化。旧 V12 的 residual/bank 约 88mAP 来自 eligible-only gallery，
不能直接与完整 3126-gallery 的终态结果作继续训练前后的比较。
源码证据：tools/build_v12_complete_path_oof_targets.py:610–617 仅收集
eligible_records 的 heldout 子集，当前 tools/train_signal_preserving_v22.py
评价全部 heldout_records。诊断只补齐这个可比性缺口，不假定结果方向。

固定范围为三个已有共同初始化模型，每个只恢复对应 V12 Signal/expert 原权重，
用当前 V22 build_model 严格加载。完整 state SHA 必须同时等于该fold两个终态
training.initial_state_sha256，全部 source/fit/heldout/架构绑定必须相等。
不构建任何新的训练优化器，不调用 loss/backward，不进行新训练，
不写 checkpoint、不读取 dev/official/test 数据、不查看中间 epoch checkpoint。

每折使用与 V22 终态完全相同的 gallery 文件名/ID/camera 清单、评价 transform、
batch128、FP32特征与距离代码、seed42、CuDNN deterministic=true/benchmark=false。
三fold gallery1000/1051/1075合计3126，47×3身份；query190/179/202合计571，
21个跨camera身份，2555条只从query分母排除而继续留作gallery distractors。
baseline/fused/CNN/Transformer/Mamba全部五类输出都保存逐query AP/Rank和指标，
不挑fold、身份或分支。三fold所有输出的初始值与两个终态的变化都报告，
全部21身份变化另作完整JSON算术复核，不添加晋级门槛或选优规则。

评价通过现有 evaluate 的 inference_mode+eval 执行；外层实际 forward hook
记录模型调用和triplet数，断言无梯度、每折完整记录数及 state 不变。
每fold baseline-only全部AP/Rank/metric必须与两个终态精确一致，
六原source权重、全部实际依赖和 V22 终态文件在前后完整SHA一致。
输出来源绑定、当前执行commit、Torch/CUDA/NumPy版本、原始逐fold数组、全部汇总，
并将初始相对终态的数据明确分类为同一复用OOF的只读描述性证据。

这次评价不是新的独立验证集、消融训练、重新训练 V12/V22，或新主版本。
结果不用于选择回退到初始化，也不重新打开已封存版本的超参数。
它只用于判断下一项尚未设计的主要干预需要针对哪一段训练/表征变化。
若未显示继续训练退化，也必须原样报告；不能预设“更多训练必然过拟合”。

当前GPU空闲；启动前实际检查可用显存>=22000MiB及源文件SHA。
预计2–3分钟，按完成窗口查询一次原进程，观察超时不能当作执行失败或重启理由。
全部模型/图像/张量操作都在远端GPU；Windows只传输、文档和JSON/NumPy指标算术。
先提交源码/本计划及SHA后执行；原始输出及日志保存、完整算术核验、
独立诊断审计后再将结果用于机制讨论。V22 Q1终态独立审计可并行且输入不改写。
