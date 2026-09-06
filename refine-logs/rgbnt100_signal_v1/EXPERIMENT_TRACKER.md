# RGBNT100 Signal source-only 三折基线 v1

更新时间：2026-09-06T10:22:31.091890+08:00。T0/M0已通过，首次完整基线在fold0首个epoch内AMP溢出停止；无检索结果。

| 阶段 | 状态 | 完整范围 |
|---|---|---|
| T0 | PASS | 8675文件/26025切片、8675query mask、2项人工camera协议fixture |
| M0 | PASS_ENGINEERING_ONLY | 3×8更新、1536源曝光、195/195梯度、48clean source重载前向 |
| M0 files/scalars | PASS | 三checkpoint内容/21文本及全部24步，0模型/图像前向 |
| B0 first attempt | ENGINEERING_STOP_AMP_OVERFLOW | 60a3d0e、33.7765秒退出1；0完整epoch，无checkpoint，成功更新数未知 |
| Source overflow capture | READY_NOT_RUN | 原train_source固定fold0/首异常或epoch1边界；逐步落盘，0heldout |
| Full terminal verifiers | PREPARED_NOT_RUN | 原先准备完整三fold/90epoch/8675query核验，不用于本失败目录 |

原config/plan/runner未改；M0不是真实检索成绩。原始失败现场与独立诊断的更新成本分别登记。
内部三fold协议与官方1715query/8575gallery不同；没有读取官方测试，主目标未达到。
结果页：results/TRIFUSION_RGBNT100_SIGNAL_FORMAL_ENGINEERING_STOP_2026-09-06.md。
