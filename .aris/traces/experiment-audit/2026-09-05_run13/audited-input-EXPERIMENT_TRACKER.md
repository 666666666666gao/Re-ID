# V23 experiment tracker

更新时间：2026-09-05T23:22:40.871241+08:00；状态Q1_COMPLETE_FAIL_SEALED_INDEPENDENT_AUDIT_PENDING。
实际执行源码9f4a10b6162b9658ba103cd92466411ebb6ccd8f；原PID44684退出码0。
原三fold、两端、20epoch全部完成，120epoch行、3360次优化更新；未重启或选点。

| ID | 内容 | 状态 | 证据 |
|---|---|---|---|
| V23-T0 | 五项远端CUDA合成契约 | DONE_PASS | evidence/trifusion_v23_t0_20260905.json |
| V23-M0 | 完整54前向/116优化步 | DONE_PASS | evidence/trifusion_v23_m0_seed42_9f4a10b.json |
| V23-M0-AUDIT | 独立完整M0审计 | COMPLETE_WARN_ENGINEERING_PASS | EXPERIMENT_AUDIT_V23_M0.md/json |
| V23-Q1 | 三fold两端完整五路检索 | COMPLETE_FAIL_SEALED | evidence/trifusion_v23_q1_seed42_9f4a10b.json |
| V23-Q1-FILES | 39文件SHA和六独立回执 | DONE_VERIFIED | evidence/trifusion_v23_q1_terminal_file_verification_20260905.json |
| V23-Q1-ARRAYS | 120训练行、全部AP/Rank与bootstrap | DONE_VERIFIED | evidence/trifusion_v23_q1_array_verification_20260905.json |
| V23-Q1-LOG | 全部120训练行和终态事件 | DONE_VERIFIED | evidence/trifusion_v23_q1_log_verification_20260905.json |
| V23-Q1-AUDIT | 独立终态完整性审计 | PENDING | .aris/traces/experiment-audit/2026-09-05_run13/ |
| V23-D1 | 141-fit refit与固定30-dev | NOT_QUALIFIED_NOT_RUN | 五项固定门全部失败 |

完整图库1000/1051/1075，共3126条；合法query190/179/202，共571条、21身份。
其余2555条仅从query分母排除，仍保留图库干扰。全部为反复开发的训练内部OOF，
不是独立dev、官方测试或SOTA结果。30-dev/official访问均为0。

| 输出 | 对照 mAP | 候选 mAP | 差值 pp | 对照 R1 | 候选 R1 |
|---|---:|---:|---:|---:|---:|
| baseline_only | 77.487603116 | 77.487603116 | +0.000000000 | 79.334500876 | 79.334500876 |
| fused | 80.507515547 | 80.254811046 | -0.252704501 | 84.413309982 | 83.537653240 |
| cnn | 79.471874954 | 80.540530605 | +1.068655652 | 83.537653240 | 84.413309982 |
| transformer | 78.529832158 | 77.927403684 | -0.602428474 | 80.910683012 | 81.786339755 |
| mamba | 77.796840675 | 77.529976219 | -0.266864457 | 80.035026270 | 78.633975482 |

三折fused增益：-0.00012121161431366545, -0.9996673157096865, 0.17162975099304845 pp。
21身份、10000次seed42 bootstrap的95%下界：-1.509454322847345 pp。
原五项科学门全部FAIL；封存本次V23，不扫描width/stage/scale/模态/专家/epoch/LR/seed或改loss重跑。
不执行D1、dev、official或消融；M0工程PASS与Q1科学FAIL分别保留。
候选CNN改善不能替代融合晋级，也不证明模态适配普遍无效。
新增1777536可训练参数和额外反向计算仍是机制解释的混杂因素。
全指标和bootstrap重算最大差0.0；120epoch的14项加权loss最大舍入差3.3728824289092074e-08。
这些为执行器数值核对，独立终态审计仍待完成；M0审计不能代替Q1审计。
完整30组fold/endpoint/output指标、21身份的两端全部五路四指标及逐query变化，
见results/TRIFUSION_RGBNT201_V23_COMPLETE_Q1_2026-09-05.md及对应完整比较JSON。
原运行耗时4282.281049251556秒。
