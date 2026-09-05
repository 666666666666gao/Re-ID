# MSVR310 原三角色架构独立训练比较 tracker

更新时间：2026-09-06T06:03:04.455453+08:00。状态 **PREPARED_NOT_RUN**。
Signal B0独立审计已闭合，内部53.129380561 mAP/63.0 Rank-1；本比较尚无结果。

| Run ID | Milestone | Purpose | System | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| MT01 | M0 | 工程资格 | 原V8三角色，从车辆B0+新角色初始化 | 三折source、0heldout | 梯度/AMP/冻结/严格重载；固定100步熵下界损失比 | MUST | NOT_RUN | 24容量+100过拟合更新 |
| MT02 | Q1 | 跨数据集完整系统比较 | 固定Signal对照与同checkpoint五输出 | 600query/1032gallery/60query身份 | 全五输出、AP/Rank、身份bootstrap及五条件 | MUST | NOT_RUN | M0 PASS必要；三个20epoch固定终点 |
| MT03 | audit | 完整终态独立复核 | 原完整文件与全部标量/离散排名 | 同固定协议 | GT/算术/调用路径/来源范围 | MUST | NOT_RUN | 不在本地运行模型/张量/图像 |

不使用RGBNT201或已失败版本角色权重；不重新训练B0，不选seed/epoch/fold。
本合同不开放RGBNT201 dev/官方或消融；支持条件通过仍只是单seed内部证据。

2026-09-06T06:08:14.672010+08:00 R2：首次启动前严格SHA失败，尚未创建训练进程；5个历史CRLF/LF输入差异已核实等于Git blob且AST一致。
只修正实际远端字节绑定，原配置和诊断保留；runner/模型/优化/门/数据不变，M0与正式比较仍NOT_RUN。
