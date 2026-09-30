# Patch memory 面板设备恢复执行计划

2026-10-01 00:30，恢复既有六端定义；没有新的模型、损失、种子或门槛。原实验计划 EXPERIMENT_PLAN.md 和210份运行源字节保持，原故障campaign仍为FAILED。

00:23实测四张RTX3090均通过新的真实Mamba前向/反向，原controller和两个暂停worker均不存在。MSVR310 local与RGBNT201 full已完成50轮的权重SHA与原封存值完全一致。

恢复目录：logs/patch_memory_roles_recovery_20261001。

1. MSVR310 local、RGBNT201 full：复用原M0及原50轮权重，独立新进程执行原严格evaluate入口。仅写尚不存在的official_metrics.json和official_distances.pt，保存新进程真实退出码、评价stdout、原training/checkpoint SHA；不伪造旧训练进程退出码，也不改原campaign状态。
2. RGBNT201 local、RGBNT100 local：保留原中止尝试，另建新目录，从原初始化重新执行真实8批M0、完整50轮及严格评价。没有optimizer状态，不用中途best冒充同日程续训。
3. RGBNT100 full、MSVR310 full：原来未启动，按原合同首次执行真实M0、50轮和严格评价。
4. 先同时运行两端评价与两端新训练；评价释放GPU后立即按240秒队列周期启动剩余两端。复用现有队列，不抢占其他用户任务，不自动重试。
5. 每端仍调用原collect_patch_memory_roles.verify，核对原schema/源SHA/最高mAP权重及完整gallery和真实过滤，门槛没有放宽。恢复collector只增加对原完整训练路径的明确引用；六端都验收后再比较local/full，当前不据两端中途或正式分数调整策略。

恢复脚本：RECOVER_PANEL_20261001.py。它只是执行和归档层；原训练模型/入口/环境不变。复制原manifest.json到恢复目录，并单独记录recovery_manifest.json绑定原campaign、manifest、脚本和四卡真实kernel回执SHA。

环境复用：/data/gaob/Re-ID/conda-envs/tri_reid/bin/python。没有环境重建、pip安装或模型权重修改。训练、checkpoint、距离数组仍只保留远端。

启动命令（远端仓库目录）：

    /data/gaob/Re-ID/conda-envs/tri_reid/bin/python -B refine-logs/patch_memory_roles_v1/RECOVER_PANEL_20261001.py --original /data/gaob/Re-ID/Trifusion/logs/patch_memory_roles_20260930 --campaign /data/gaob/Re-ID/Trifusion/logs/patch_memory_roles_recovery_20261001

Goal已由继续指令恢复ACTIVE，SOTA目标仍UNMET。先完成既定六端再决定新结构，不扩大种子搜索，不重写已封存失败。
