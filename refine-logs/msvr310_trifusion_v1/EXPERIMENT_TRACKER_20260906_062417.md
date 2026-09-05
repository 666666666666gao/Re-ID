# MSVR310 原三角色架构独立训练比较 tracker

更新时间：2026-09-06T06:24:17.950707+08:00。状态 **M0_PASS_COMPARISON_RUNNING**。
原M0 wrapper69455 exit0，执行1c444cd；124更新、0heldout，最后100步excess ratio0.000702022804。
三fold203/203非零梯度，冻结/AMP/五输出严格重载通过；17文件及全部124步stdlib核验通过。
Signal B0独立审计已闭合，内部53.129380561 mAP/63.0 Rank-1；新三角色完整比较还没有结果。

| Run ID | Milestone | Purpose | Status | Notes |
|---|---|---|---|---|
| MT00 | prelaunch | 源码字节绑定 | FIXED | 首次0模型调用；5项实际远端LF/Git SHA明确绑定；原记录保留 |
| MT01 | M0 | 三fold8步容量+全新fold0固定100步 | PASS_ENGINEERING_ONLY | 124更新/7936训练曝光/72role+24Signal clean记录前向；0heldout |
| MT01A | independent audit | M0工程证据复核 | PENDING | 范围含原始输入、真实采样、全步损失与调用路径；不冒充检索结果 |
| MT02 | Q1 | 20epoch三折完整五输出检索比较 | RUNNING | wrapper70422；约18–25分钟；无M0训练后权重继承 |
| MT03 | terminal audit | 完整文件/保存数组/独立指标审计 | NOT_RUN | 等原始完整终态，不选epoch/seed/fold |

固定config/plan/runner SHA不变；scene过滤及完整600query/1032gallery保留。
本合同不开放RGBNT201 dev/官方或消融；科学条件通过仍只是单seed内部跨数据集支持。
