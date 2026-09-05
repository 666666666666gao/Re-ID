# MSVR310 Signal v1 tracker

更新时间：2026-09-06T05:00:34.202298+08:00。状态 **M0_R2_PASS_B0_RUNNING**。
M0 R2原wrapper63101 exit0，24次优化/48次clean source前向，195/195训练梯度张量，三折严格重载一致。
B0原wrapper63945实际04:50:55启动，固定50epoch三折source基线；执行bb01d60。
最新观察04:57:26：第一折46/50epoch，进程仍运行；完整折/检索终态尚未取得。
完整报告 results/TRIFUSION_MSVR310_SIGNAL_SOURCE_M0_2026-09-06.md。固定EXPERIMENT_PLAN.md、config及runner保持原SHA。

| Run | 范围 | 状态 |
|---|---|---|
| T0 | 2个scene排序回归+1032真实标签掩码 | PASS；R2对应排序函数AST/测试源码未变 |
| M0 R1 | 原三折source工程门，实际仅fold0的8步 | FAIL；原日志/退出/配置完整保留 |
| source梯度诊断 | 1个B64 batch，1次backward/0优化 | DONE；6个结构上无梯度参数，787968个数值 |
| M0 R2 | 三折分别重新初始化，各8步 | PASS_ENGINEERING_ONLY；24更新/0heldout |
| B0 | 三折source Signal各50epoch，固定终点完整gallery | RUNNING；预计05:13–05:16附近完成 |
| 独立审计 | 完整基线结果和工程证据 | PENDING |
| 官方/新三分支 | 不在本基线合同中 | NOT_RUN |

初次进度读取器误查event=epoch，报告epoch_rows=0；原log_tail有signal_source_epoch46。
错误及更正单独保存，不改变训练或解释成零训练。无M0权重继承、无终点/seed选择。
