# 角色 Patch 读取实验执行表

2026-09-30 10:53快照；四端真实M0通过，正式验收0/6。设备故障处理中，未产生完整检索结果。

| Run | 数据集 | 模式 | 完整预算/选点 | 状态 |
|---|---|---|---|---|
| L201 | RGBNT201 | local_memory：9 Patch | seed42 / 50轮 / fused mAP-best | M0通过；GPU0设备错误，训练2轮后中止，保留失败尝试 |
| L100 | RGBNT100 | local_memory：9 Patch | seed42 / 50轮 / fused mAP-best | M0通过；CUDA unknown error，首轮未完成，保留失败尝试 |
| LMSVR | MSVR310 | local_memory：9 Patch | seed42 / 50轮 / fused mAP-best | M0通过；22/50轮继续训练；worker暂停在后继独立评价前 |
| F201 | RGBNT201 | full_memory：128 Patch | seed42 / 50轮 / fused mAP-best | M0通过；15/50轮继续训练；worker暂停在后继独立评价前 |
| F100 | RGBNT100 | full_memory：128 Patch | seed42 / 50轮 / fused mAP-best | 未运行；队列因设备故障停止派发 |
| FMSVR | MSVR310 | full_memory：128 Patch | seed42 / 50轮 / fused mAP-best | 未运行；队列因设备故障停止派发 |

四端M0均为127/127可训练张量获得非零梯度、冻结基线未变、严格重载误差0。201两端初始完整模型SHA相同；其余两对尚缺另一端M0，不能提前写全六端匹配通过。

执行源码23f96ac；210个来源文件SHA未变。主队列controller3680049保持失败状态；两个worker3680054/3680055用SIGSTOP暂停的是后继调度，真实train child3681858/3681847未暂停。CUDA恢复并通过新进程真实kernel检查后才SIGCONT这两个worker，让其完成独立重载评价。失败两端必须另建尝试目录从相同seed/初始权重完整重跑；不复用没有优化器状态的中途best。原失败记录和两端有效完整终点分别保留，不将队列FAILED改写成全部完成。

0号GPU PCI 0000:3B:00.0无法读取；对其余三张卡分别按UUID启动新进程，CUDA初始化均失败，而两个旧CUDA上下文仍能训练。根因/Xid未知，不能断言具体硬件、应用或驱动原因。已请用户/管理员检查；用户回复“我先检查一下”。估计继续运行两端约11:10～11:20完成50轮，整机维护应避开仍在执行的训练。

证据：`logs/patch_memory_gpu_incident_20260930.json`、`logs/patch_memory_roles_start_snapshot_20260930.json`、`logs/patch_memory_incident_20260930/SNAPSHOT.json`及同目录原日志/回执快照。第一次启动因稀疏检出移除前序验收文件而未进入训练；原失败log和launch回执已封存，已将所需结果/前序证据/新计划目录加入远端sparse-checkout，防止再次被同步移除。没有修改训练源码或登记门。
