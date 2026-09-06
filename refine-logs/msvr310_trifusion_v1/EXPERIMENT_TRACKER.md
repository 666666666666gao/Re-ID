# MSVR310 原三角色架构独立训练比较 tracker

更新时间：2026-09-06T08:53:29.912750+08:00。状态 **COMPLETE_COMPARISON_SUPPORT_FAIL_AUDIT_CLOSED_WITH_LIMITS**。
三折原20epoch全部完成，780更新（第0折260复用+R3新增520），600query/1032gallery全量比较。
Signal53.129380561/63.0，fused52.117390117/60.833333333；ΔmAP−1.011990444，五项科学条件全false。
三fold精确B0特征/距离、冻结/梯度/AMP、全部checkpoint/15数组及780步/3000query核验通过。
M0独立审计CLOSED_WARN；正式终态独立审计CLOSED_WITH_LIMITS。固定试验科学失败，不重训/扫描/晋级/消融。

| Run ID | Milestone | Purpose | Status | Notes |
|---|---|---|---|---|
| MT00 | prelaunch | 源码字节绑定 | FIXED | 首次0模型调用；5项实际远端LF/Git SHA绑定，原记录保留 |
| MT01 | M0 | 三fold容量+固定100步过拟合 | PASS_ENGINEERING_ONLY | 124更新，0heldout |
| MT01A | independent audit | M0工程复核 | CLOSED_WARN | run18两轮，60输入/124步独立复算；非检索证明 |
| MT02 | original comparison | 原第0折训练 | STOP_PRESERVED | 70422 exit1，原260更新及失败字节保留，未重训 |
| MT02D | SIM diagnosis/repair | 冻结后mm/bmm差异与精确修复 | PASS_ENGINEERING_ONLY | 原B0特征/距离及角色残差全360核验通过 |
| MT02R3 | full comparison | 复用第0折，只训练后两折 | COMPLETE_SUPPORT_FAIL | 75993 exit0；全部三折/五输出，5科学门全false |
| MT03 | terminal verification | 原始文件/15数组/3000query/780步 | PASS | 8.6370秒远端+0.4124秒本地；0模型/图像前向 |
| MT03A | terminal independent audit | 独立终态与修复/复用证据链 | CLOSED_WITH_LIMITS | run19两轮；102输入/3000query/780步；enginePASS/scienceFAIL |

固定config/plan/runner SHA不变；scene过滤及完整600query/1032gallery保留。
本合同不开放RGBNT201 dev/官方或消融；科学条件通过仍只是单seed内部跨数据集支持。

2026-09-06T06:41:47.899579+08:00：已封存原失败；完整360条同条件四路径只读特征诊断PREPARED_NOT_RUN。
不更改原固定计划/config/runner/门，不重训已完成fold0。独立M0审计继续以原60份冻结输入为范围。

2026-09-06T08:08:29.999652+08:00: full360 parity diagnosis completed once, exit0; B0 standalone exact, after-wrap differences isolated to SIM. Fixed64 cached-input operation diagnosis registered, NOT_RUN. Original comparison remains ENGINEERING_STOP; no ranking or repeated fold0 training.

2026-09-06T08:11:44.639211+08:00: independent M0 audit run18 CLOSED_WARN, engineering PASS, scientific/retrieval NOT_ESTABLISHED;60/60 immutable inputs and independent124-step replay pass. Two report-only field corrections by reviewer do not alter verdict. Comparison remains stopped, operation probe separate; no promotion.

2026-09-06T08:19:47.876038+08:00: operation cause measured: SIM freeze changes mm to bmm, restored flags recover exact B0. Inference-only functional view helper/full360 exact feature+distance verification registered NOT_RUN;720 role forwards/0update/ranking. Original comparison remains stopped.

2026-09-06T08:26:05.110072+08:00: full360 exact inference verification PASSED, including whole210x360 B0 distances and unchanged role/modal residuals. R3 continuation PREPARED_NOT_RUN: reuse fold0 training/features, only train folds1/2,520 new updates; all original scientific gates remain. No retrieval metrics read yet.

2026-09-06T08:36:44.580202+08:00: R3 wrapper75993 live at08:33:49, execution1ff7e2d. Fold0 reused;fold1 complete260 new updates, baseline exact;fold2 running. No complete result yet. Terminal verifiers prepared after two partial folds, NOT_RUN.

2026-09-06T08:53:29.912750+08:00: complete terminal verified; no running training. Negative fixed-study result retained without original fold0 retraining. Independent terminal audit pending.

2026-09-06T09:36:57.748608+08:00: independent terminal audit run19 CLOSED_WITH_LIMITS; numerical replay1.2726868s, report-only correction round2; fixed science failure unchanged, no runtime/retraining.
