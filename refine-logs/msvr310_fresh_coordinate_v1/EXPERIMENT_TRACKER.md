# MSVR310 current historical coordinate update V1

| 阶段 | 固定范围 | 状态 |
|---|---|---|
| T0 | 固定SHA、780batch队列、CPU数学 | PASS，01:32:40退出0 |
| M0 | 六端8步＋两端100步，共248 | RUNNING，原PID192718，4/6 capacity端已记录 |
| M0 CPU | 全训练/坐标/终点核验 | NOT_RUN |
| Q1 | 六端260步，全部600query | NOT_RUN |
| Q1 CPU | 全训练、坐标、权重、排序、原门 | NOT_RUN |

control=陈旧缓存更新，fresh_memory=当前角色重编码历史坐标更新；两端匹配重编码计算。历史detach、原模型/其余loss/采样/预算保持。尚无新实验模型结果，不提前宣称有效。

## 启动与实查（2026-09-08T01:36:06.658087+08:00）

固定b4501fa/32e22d3a，原wrapper192704。最近2026-09-08T01:36:06.658087+08:00实查原wrapper192704与m0 PID192718及命令行均存在；M0已写入4/6个capacity端，完整248步与CPU尚不能提前PASS。 没有重启；自动M0→CPU→Q1→CPU，保持原门。
