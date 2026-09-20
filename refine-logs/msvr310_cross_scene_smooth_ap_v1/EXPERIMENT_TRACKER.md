# 跨场景 Smooth-AP 执行记录

| Run | 目的 | 状态 | 证据 |
|---|---|---|---|
| T0 | 目标/掩码/两侧导数与真实支持约束 | PASS | d35864d，00:25:36退出0；780来源batch及全部合成检查通过 |
| Review | 新合同与实现独立代码审查 | PASS_WITH_LIMITS | 同族静态审查，0阻断；不能替代实际运行核验 |
| M0 | 三折两端容量与固定过拟合 | PASS_ENGINEERING_ONLY | 00:36:52退出0，完整248步；六端203/203梯度张量、overflow0，两个原过拟合门通过 |
| M0_CPU | 完整保存目标重算 | PASS | 00:37:03退出0；248步/4945920距离元素，summary SHA c59e39cc2192bfb11b12a58f53eb8b3853455bfd41744c74aaf4742272ba1506 |
| M0_Audit | 独立完整工程核验 | WARN / CLOSED_WITH_LIMITS | 确定性工程PASS，0阻断；248步及六checkpoint全量复核，逐步反传仍为运行见证 |
| Q1 | 三折两端固定20epoch | RUNNING | 2026-09-21T03:08:17.845631+08:00原wrapper16885/Q1 18222存活；4/6端固定20epoch/260步、checkpoint/检索/receipt齐全；fold2 control自动完成第5epoch。待完整六端与CPU核验 |
| Q1_CPU/Audit | 全量核验与独立终态审计 | NOT_STARTED | 不使用中间fold判定 |

seed42固定；当前无新检索成绩。其他13项、历史VJP、候选池及固定终点不变。原Smooth-AP Q1_FAIL保持。
