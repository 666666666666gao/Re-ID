# MSVR310 Signal v1 tracker

更新时间：2026-09-06T05:17:36.048722+08:00。状态 **BASELINE_COMPLETE_AUDIT_WARN**。
原wrapper63945已退出0；执行bb01d60，三折各50epoch/650更新，总1950更新。
完整600query/1032gallery、60query身份/155heldout身份；95个单scene身份作为干扰保留。
总mAP 53.129380561 / Rank-1 63.000000000%，仅训练内部身份隔离协议。
完整结果：results/TRIFUSION_MSVR310_SIGNAL_SOURCE_BASELINE_2026-09-06.md。固定EXPERIMENT_PLAN.md、config和runner保持原SHA。

| Run | 范围 | 状态 |
|---|---|---|
| T0 | 2个scene排序回归+1032真实标签掩码 | PASS；R2排序AST/测试源码未变 |
| M0 R1 | fold0的8步 | FAIL；原始日志/退出/配置保留 |
| source梯度诊断 | 1个B64 batch，1backward/0更新 | DONE |
| M0 R2 | 三折各8步，0heldout | PASS_ENGINEERING_ONLY |
| B0 | 三折source各50epoch，fixed epoch50完整gallery | COMPLETE_BASELINE_NOT_METHOD_QUALIFICATION |
| 执行侧核验 | 全文件/原保存特征距离/全部排序/全部训练步骤 | DONE；距离逐元素一致，原数值完整保留 |
| 独立审计 | 完整基线及工程前置 | CLOSED_WARN；engineering PASS；两轮trace17 |
| 官方/新三分支/消融 | 不在本合同 | NOT_RUN |

不按单折或终态mAP重训/调参/延长；不把内部结果当官方复现，不改变RGBNT201主目标与已有失败封存。

独立审计闭合时间：2026-09-06T05:55:13.439177+08:00。71输入全SHA一致；首轮stdlib独立算术通过，第二轮仅修正派发和二进制证据边界表述。
同家族独立上下文Type-A、后端未独立认证、远端二进制收据及单seed内部协议限制保留；新方法仍未执行。
