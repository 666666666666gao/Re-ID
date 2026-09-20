# 跨场景 Smooth-AP 执行记录

| Run | 目的 | 状态 | 证据 |
|---|---|---|---|
| T0 | 目标/掩码/两侧导数与真实支持约束 | PASS | d35864d，00:25:36退出0；780来源batch及全部合成检查通过 |
| Review | 新合同与实现独立代码审查 | PASS_WITH_LIMITS | 同族静态审查，0阻断；不能替代实际运行核验 |
| M0 | 三折两端容量与固定过拟合 | PASS_ENGINEERING_ONLY | 00:36:52退出0，完整248步；六端203/203梯度张量、overflow0，两个原过拟合门通过 |
| M0_CPU | 完整保存目标重算 | PASS | 00:37:03退出0；248步/4945920距离元素，summary SHA c59e39cc2192bfb11b12a58f53eb8b3853455bfd41744c74aaf4742272ba1506 |
| M0_Audit | 独立完整工程核验 | WARN / CLOSED_WITH_LIMITS | 确定性工程PASS，0阻断；248步及六checkpoint全量复核，逐步反传仍为运行见证 |
| Q1 | 三折两端固定20epoch | Q1_FAIL / AUDIT_CLOSED_WITH_LIMITS | 六端1560更新、完整CPU及独立数值核验；配对2/5、Signal1/5；fused +0.567183，不晋级 |
| Q1_CPU | 完整终态重算 | PASS | 04:20:11退出0，1560步/116501504保存距离元素/2069520检索元素；反传为运行见证 |
| Q1_Audit | 独立完整终态审查 | WARN / CLOSED_WITH_LIMITS | 全量确定性复算通过；原梯度运行见证、非逐位预热、单seed及重复开发限制保留 |

seed42固定；当前已有内部Q1，尚无新官方检索成绩。其他13项、历史VJP、候选池及固定终点不变。原Smooth-AP Q1_FAIL保持。
