# RGBNT100 Signal source-only 三折基线 v1

更新时间：2026-09-06T09:50:38.079036+08:00。T0 PASS、M0 PASS_ENGINEERING_ONLY；完整30epoch基线READY_NOT_RUN。

| 阶段 | 状态 | 完整范围 |
|---|---|---|
| T0 | PASS | 8675文件/26025切片、8675query mask、2项人工camera协议fixture |
| M0 | PASS_ENGINEERING_ONLY | 3×8更新、1536源曝光、195/195梯度、48clean source重载前向 |
| M0 files/scalars | PASS | 三checkpoint内容/21文本及全部24步，0模型/图像前向 |
| B0 | READY_NOT_RUN | 3×30epoch fresh模型；全部8675query/8675gallery，固定终点 |

原config/plan/runner不变。M0不是真实检索成绩，不能用来宣称超过基线。
内部三fold协议与官方1715query/8575gallery不同；没有读取官方测试，主目标未达到。
