# V20 Experiment Tracker

更新时间：2026-09-05T14:54:11.240053+08:00。状态：M0_PASS_Q1_RUNNING。

| ID | 内容 | 状态 | 证据/依赖 |
|---|---|---|---|
| V20-T0 | 三项远端CUDA数学单测 | DONE_PASS | evidence/trifusion_v20_t0_20260905.json |
| V20-M0 | 六模型配对及新损失梯度、两端容量、100步过拟合 | DONE_PASS | evidence/trifusion_v20_m0_seed42_3cea5bf.json |
| V20-M0-AUDIT | M0阶段独立完整性审计 | IN_PROGRESS | .aris/traces/experiment-audit/2026-09-05_run06 |
| V20-Q1 | 三折两端20epoch完整OOF及五路全量评价 | RUNNING | 原screen v20_cross_modal_3cea5bf |
| V20-AUDIT | 全量终态证据复算及独立审计 | TODO | 完整终态 |
| V20-D1 | 条件性141-fit refit与一次30-dev | LOCKED | Q1全部科学门通过后另行固定合同 |

source3cea5bfc17e214b1829c020527699d939efa221d；14:48:31CST启动，M0在228.975805
秒内完成，14:52:27观测Q1已进入新建第一端。只允许原进程继续完整20epoch。

两端各98,800,141总参数、7,841,292可训练参数、203训练tensor，推理新增参数0。
六模型配对初始state/五路输出/增强/绑定一致。每专家新损失单独梯度存在；
三折CNN/T/M非零encoder张量数42/54/93、42/54/92、42/54/91。
两端8步峰值reserved6062MiB，203/203完整梯度覆盖、overflow0、冻结state不变。
100步总损失1.8860781193→1.1006586552；基础ID/Triplet0.6110473871→
0.5803135037，新损失5.1001229286→2.0813806057；解析总下界1.0982433065，
超额损失比0.0030658060<=0.1。M0无heldout检索结论，不代表Q1科学通过。

M0不可变snapshot340010字节，SHA5fd4922a7a7036f6905c54397809faed18387666b1df18aa39e5429cd10876a0，
本地与远端相同；保留原RUNNING状态与空fold数组。当前无完整Q1检索结果，
dev/official/D1访问0。Q1初估60–80分钟，等待第一端完整epoch时长修正ETA。
旧V19 Q1_FAIL封存不变。无扫描、消融、新种子或本地模型执行。
