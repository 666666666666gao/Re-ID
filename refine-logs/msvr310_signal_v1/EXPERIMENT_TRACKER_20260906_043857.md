# MSVR310 Signal v1 tracker

更新时间：2026-09-06T04:38:57.493093+08:00。状态M0_R1_FAILED_R2_PREPARED。正式基线与held-out检索仍0。
原M0在fold0完成8次更新后未通过全部trainable有梯度门，exit1；后两折未运行。
零更新B64诊断与源码定位6个无梯度TokenSelection参数，R2显式冻结，保留其值及前向。
详见EXPERIMENT_PLAN.md末节和evidence/trifusion_msvr310_signal_v1_m0_r2_preregistration_20260906.json。

| Run | 范围 | 状态 |
|---|---|---|
| T0 | 2个scene排序回归+1032真实标签掩码 | PASS；R2对应排序函数AST/测试源码未变 |
| M0 R1 | 原三折source工程门，实际仅fold0的8步 | FAIL；完整原日志已保存，不冒充三折完成 |
| source梯度诊断 | 1个B64 batch，1次backward/0优化 | DONE；6个无梯度参数，787968个数值 |
| M0 R2 | 三折分别重新初始化，各8步，冻结已证实不更新的选择器 | PREPARED_NOT_RUN |
| B0 | 三折source Signal各50epoch，固定终点完整gallery | NOT_RUN |
| 官方/新三分支 | 不在本基线合同中 | NOT_RUN |
