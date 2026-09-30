# 角色 Patch 读取实验执行表

2026-09-30 11:19 实查：两端训练完成 50/50；正式验收仍为 0/6。原 campaign 保持 FAILED。硬件/驱动故障尚未恢复，不能把训练检查当成正式检索终点。

| Run | 数据集 | 模式 | 固定合同 | 当前状态 |
|---|---|---|---|---|
| L201 | RGBNT201 | local_memory，9 Patch | seed42 / 50轮 / fused mAP-best | 设备故障中止，仅完成2轮；原失败保留，恢复后另建同定义完整尝试 |
| L100 | RGBNT100 | local_memory，9 Patch | 同上 | CUDA unknown error，首轮未完成；原失败保留 |
| LMSVR | MSVR310 | local_memory，9 Patch | 同上 | 50轮/1000步训练完成，best第10轮；CPU轨迹和权重字节检查通过，待严格重载评价 |
| F201 | RGBNT201 | full_memory，128 Patch | 同上 | 50轮/2649步训练完成，best第2轮；CPU轨迹和权重字节检查通过，待严格重载评价 |
| F100 | RGBNT100 | full_memory，128 Patch | 同上 | 尚未启动；原控制器已停止新派发 |
| FMSVR | MSVR310 | full_memory，128 Patch | 同上 | 尚未启动；原控制器已停止新派发 |

四端既有真实M0通过，201 local/full初始状态SHA一致。当前训练-only检查验证50轮、全部步骤、ID+Triplet算术、auxiliary=0、最高mAP选点元数据、144个保存张量有限及210份源文件SHA；未运行新神经前向、严格模型重载或完整gallery复算。因此不报告这两端为正式结果，不提前计算local/full配对差。

MSVR权重SHA：8b70501480d000400e0a7be2228182c157dd9a39a57f3c18f22ba34dd0aee8b2。
201权重SHA：c39520d660b1a31fa73ce3c12811fd4c13317d98b7a0c1a82df9d58bc3a0a444。
所有checkpoint仍保留远端；原始文本见 logs/patch_memory_training_complete_20260930/。

11:19重新使用三张可由NVML读取的GPU UUID，在新进程启动真实Mamba前向/反向检查；三次均在torch CUDA初始化时退出1，尚未进入Mamba。0号卡11:17仍不可访问。无内核日志权限，Xid及根因未知，不称算法失败，也未修改依赖、重置设备或重启服务器。

worker父进程3680054/3680055仍SIGSTOP于后续评价之前；训练子进程3681858/3681847已结束并成为待父进程回收的zombie，没有训练在运行。CUDA实际恢复并通过新进程真实kernel检查后，若父进程仍存在则SIGCONT完成原严格评价；若管理员维护已终止父进程，则以原checkpoint和冻结源代码单独运行同一评价，明确保存恢复记录。失败两端和未启动两端另建尝试目录，保持同一seed、初始化、50轮日程与预登记门槛，不覆盖失败记录。

原50轮训练、真实M0及硬件故障均独立保存。归档脚本第一次误将train.log定位到trained-model目录，未更改训练或权重；已使用实际campaign子目录日志完成原字节封存。Goal ACTIVE / UNMET。
