# RGBNT100 Signal source-only 三折基线 v1

更新时间：2026-09-06T10:22:31.091890+08:00。T0/M0已通过，首次完整基线在fold0首个epoch内AMP溢出停止；无检索结果。

| 阶段 | 状态 | 完整范围 |
|---|---|---|
| T0 | PASS | 8675文件/26025切片、8675query mask、2项人工camera协议fixture |
| M0 | PASS_ENGINEERING_ONLY | 3×8更新、1536源曝光、195/195梯度、48clean source重载前向 |
| M0 files/scalars | PASS | 三checkpoint内容/21文本及全部24步，0模型/图像前向 |
| B0 first attempt | ENGINEERING_STOP_AMP_OVERFLOW | 60a3d0e、33.7765秒退出1；0完整epoch，无checkpoint，成功更新数未知 |
| Source overflow capture | REPRODUCED | 8b412d0第34步；33有效更新/2176前向，0heldout；M0标量非完全一致 |
| Saved batch probe | COMPLETE_NUMERICAL_DIAGNOSIS | 192source前向/0更新；原AMP三零Gramdet、AbsBackward0 NaN，完整FP32有限 |
| Local Gram FP32 regression | FAIL_OPERATOR_FINITE_GATE | 8034451，仍1零det/各512NaN；0模型前向/0更新 |
| Stable Gram regression | PASS_SINGLE_REAL_BATCH | 6c741b8，192source前向/0更新；195finite与3072D逐位相同 |
| R2 T0 | READY_NOT_RUN | 新runner/config绑定，重新验证全量切片/协议 |
| R2 M0 | READY_NOT_RUN | 三fold分别完整首epoch，195梯度/AMP/重载门保留 |
| Full terminal verifiers | PREPARED_NOT_RUN | 原先准备完整三fold/90epoch/8675query核验，不用于本失败目录 |

R1配置/合同/回执保留；原runner源字节另行存档，R2的最小源码修订单独绑定。M0不是真实检索成绩，各阶段成本分别登记。
内部三fold协议与官方1715query/8575gallery不同；没有读取官方测试，主目标未达到。
结果页：results/TRIFUSION_RGBNT100_SIGNAL_FORMAL_ENGINEERING_STOP_2026-09-06.md。
