# 跨场景 Smooth-AP 执行记录

| Run | 目的 | 状态 | 证据 |
|---|---|---|---|
| T0 | 目标/掩码/两侧导数与真实支持约束 | PREPARATION | 已有全来源支持普查复用；新公式测试待实现 |
| Review | 新合同与实现独立代码审查 | PASS_WITH_LIMITS | 同族静态审查通过；T0/M0运行仍待验证 |
| M0 | 三折两端容量与固定过拟合 | NOT_STARTED | 248更新，source-only |
| M0_CPU | 完整保存目标重算 | NOT_STARTED | 通过后才能Q1 |
| Q1 | 三折两端固定20epoch | NOT_STARTED | 1560更新，不读取官方test |
| Q1_CPU/Audit | 全量核验与独立终态审计 | NOT_STARTED | 不使用中间fold判定 |

seed42固定；无新增成绩。当前原Smooth-AP Q1_FAIL保持。
