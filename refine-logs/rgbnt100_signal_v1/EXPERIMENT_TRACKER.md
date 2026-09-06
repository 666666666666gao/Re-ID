# RGBNT100 Signal source-only 三折基线 v1

更新时间：2026-09-06T12:35:34.237907+08:00。R2固定三折30epoch基线正在运行：首折30epoch/2479次有效更新，0 AMP下降；整体终态尚未产生。三折完整首epoch M0及全部权重/257步标量已核验通过。独立审计因服务额度限制未闭合。

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
| R2 T0 | PASS | e699eac全量切片/协议；后续相同源码/config复用真实回执 |
| R2 M0 first attempt | INCOMPLETE_STORAGE_WRITE | fold0完成81更新/5184 source前向、195梯度有限；写盘失败，strict reload未执行 |
| R2 storage recovery | COMPLETE | 两个下载zip按SHA无损迁移；所有科学产物保留，新产物固定overlay |
| R2 M0 storage retry | PASS_ENGINEERING_ONLY | 81/86/90更新，共257/16448 source前向，195梯度/48严格重载前向 |
| R2 M0 all files/scalars | PASS | 3完整权重内容、全部257步精确FP32重算；0新模型前向 |
| R2 fixed30 B0 | RUNNING | def7b9b/wrapper87066；首折30epoch/2479有效更新，整体90epoch/8675query尚未完成 |
| Independent Gram audit | INCOMPLETE_SERVICE_LIMIT | 153输入/973项重放完成，最终报告未生成；不覆盖新M0 |
| Full terminal verifiers | R2_PREPARED_NOT_RUN | R2配置/T0路径/逐步JSONL/精确FP32 loss重算；完整三fold/90epoch/8675query |

R1配置/合同/回执保留；原runner源字节另行存档，R2的最小源码修订单独绑定。M0不是真实检索成绩，各阶段成本分别登记。
内部三fold协议与官方1715query/8575gallery不同；没有读取官方测试，主目标未达到。
结果页：results/TRIFUSION_RGBNT100_SIGNAL_FORMAL_ENGINEERING_STOP_2026-09-06.md。

首次进度与终态区分详见baseline_progress_20260906_1235及baseline_progress_assessment证据；预计总65–75分钟，下一计划观察13:15。新RGBNT100三角色入口仅准备，未配置/运行，须等完整B0核验。
