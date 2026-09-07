# MSVR310 current historical coordinate update V1

| 阶段 | 固定范围 | 状态 |
|---|---|---|
| T0 | 固定SHA、780batch队列、CPU数学 | PREPARED |
| M0 | 六端8步＋两端100步，共248 | NOT_RUN |
| M0 CPU | 全训练/坐标/终点核验 | NOT_RUN |
| Q1 | 六端260步，全部600query | NOT_RUN |
| Q1 CPU | 全训练、坐标、权重、排序、原门 | NOT_RUN |

control=陈旧缓存更新，fresh_memory=当前角色重编码历史坐标更新；两端匹配重编码计算。历史detach、原模型/其余loss/采样/预算保持。尚无新实验模型结果，不提前宣称有效。
