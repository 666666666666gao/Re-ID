# MSVR310 原三角色架构独立训练比较 tracker

更新时间：2026-09-06T06:24:17.950707+08:00。状态 **M0_PASS_COMPARISON_ENGINEERING_STOP**。
原M0 wrapper69455 exit0，执行1c444cd；124更新、0heldout，最后100步excess ratio0.000702022804。
三fold203/203非零梯度，冻结/AMP/五输出严格重载通过；17文件及全部124步stdlib核验通过。
Signal B0独立审计已闭合，内部53.129380561 mAP/63.0 Rank-1；新三角色完整比较还没有结果。

| Run ID | Milestone | Purpose | Status | Notes |
|---|---|---|---|---|
| MT00 | prelaunch | 源码字节绑定 | FIXED | 首次0模型调用；5项实际远端LF/Git SHA明确绑定；原记录保留 |
| MT01 | M0 | 三fold8步容量+全新fold0固定100步 | PASS_ENGINEERING_ONLY | 124更新/7936训练曝光/72role+24Signal clean记录前向；0heldout |
| MT01A | independent audit | M0工程证据复核 | PENDING | 范围含原始输入、真实采样、全步损失与调用路径；不冒充检索结果 |
| MT02 | Q1 | 20epoch三折完整五输出检索比较 | STOPPED_AT_BASELINE_FEATURE_PARITY | wrapper70422 exit1；仅fold0训练20epoch/260更新；0检索指标；后两折未启动 |
| MT03 | terminal audit | 完整文件/保存数组/独立指标审计 | NOT_RUN | 等原始完整终态，不选epoch/seed/fold |

固定config/plan/runner SHA不变；scene过滤及完整600query/1032gallery保留。
本合同不开放RGBNT201 dev/官方或消融；科学条件通过仍只是单seed内部跨数据集支持。

2026-09-06T06:41:47.899579+08:00：已封存原失败；完整360条同条件四路径只读特征诊断PREPARED_NOT_RUN。
不更改原固定计划/config/runner/门，不重训已完成fold0。独立M0审计继续以原60份冻结输入为范围。

2026-09-06T08:08:29.999652+08:00: full360 parity diagnosis completed once, exit0; B0 standalone exact, after-wrap differences isolated to SIM. Fixed64 cached-input operation diagnosis registered, NOT_RUN. Original comparison remains ENGINEERING_STOP; no ranking or repeated fold0 training.
