# V25 experiment tracker

更新时间：2026-09-06T23:38:33.458059+08:00。当前依据23:33:41真实远端观察；执行commit97468dd不变。
原训练112550存活且命令匹配，M0完整PASS，Q1已完成88/120epoch：
fold0/1两端终点完成，fold2-control8/20，最后两端由原进程继续。
完整三fold六端比较尚未完成，不提前判定Q1有效或失败。

| 阶段 | 状态 | 证据 |
|---|---|---|
| SOURCE-METADATA / T0 | COMPLETE_PASS | 全3360批、全部source身份/记录 |
| M0 | COMPLETE_PASS | 48只读batch+116优化步；203梯度；超额比0.0398826204 |
| Q1 | RUNNING | 2个配对fold完成，88/120epoch，原PID112550 |
| 完整终态CPU核验 | QUEUED_WAITING | 原wrapper114796存活，180秒间隔等待 |
| 全部身份/query报告 | PREPARED_AST_ONLY | 报告器cfa92b739b0dd2010f5905634641ea200e289542de40dd46c8334c672cec43db；尚无完整输入/输出 |
| D1 / official / ablations | NOT_QUALIFIED_NOT_RUN | 原五项科学门不变 |

完整CPU核验只在原训练terminal后单次运行，无自动重试、无新训练、无GPU前向。
报告器要求完整Q1与对应SHA核验通过，输出30个fold指标、105个身份输出行及2855个query输出行。
未读取局部终点来替代完整科学比较。

最新磁盘：数据卷free21.414532GiB，系统卷free10.364922GiB。
原24个已删权重仍不存在，12个保留模型均在且大小匹配，本次新删除0。
V25最终权重/检索数组继续保留。完整观察：evidence/trifusion_v25_disk_live_observation_20260906_233341.json。
当前三数据集baseline/SOTA总目标仍未达成；执行器核验不等于外部独立审计。
