# V21 Experiment Tracker

状态：PREREGISTERED_NOT_LAUNCHED；更新时间：2026-09-05T16:36:44.082755+08:00。

| ID | 内容 | 状态 | 证据/依赖 |
|---|---|---|---|
| V21-T0 | 三项CUDA数学/AMP/BN单测 | TODO | 已冻结源码与方案 |
| V21-M0 | 六模型配对、两端容量、SAM固定batch100步 | TODO | T0全部通过 |
| V21-Q1 | 三fold ordinary40/SAM20完整训练 | TODO | M0全部通过 |
| V21-AUDIT | 全量终态复算及独立审计 | TODO | 六端终态 |
| V21-D1 | 141-fit refit与30-dev终态 | LOCKED | 全五个Q1科学门通过后另行冻结 |

仅seed42远端GPU；B64/K8、原V8结构与七路ID/Triplet保持。
对照普通AdamW40epoch，SAM20epoch/rho0.05；前向/反传次数匹配，各3360次，
总5040优化步/6720对前向反传。数据暴露/更新次数不同，不能声称全部条件相同。
全部3126gallery/571query/21身份、三fold五路及负结果完整报告，最终checkpoint
严格重载；无中途选择或扫描。M0不通过不执行Q1，Q1失败不晋级D1/dev/official。
当前仅完成文本/AST检查，T0/M0/Q1没有执行结果。计划和固定配置执行前SHA绑定。
V20已独立审计为工程PASS/完整性WARN/科学FAIL并封存；用户整体目标未完成。
