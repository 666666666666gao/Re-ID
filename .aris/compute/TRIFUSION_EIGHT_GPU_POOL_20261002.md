# TriFusion 两台服务器八卡资源池

用户在2026-10-02明确允许2025、2026一起启动实验。实际是两台SSH服务器，每台四张24GB RTX3090，共八张，设备UUID不重叠。沿用已经通过真实CUDA/Mamba、数据和生产入口M0的环境；环境规格未变，本次只核查容量，不重新安装。

| 端口 | 项目 | 环境Python | 数据 |
|---|---|---|---|
| 2025 | /data2/gb/Re-ID/Trifusion | /data2/gb/Re-ID/conda-envs/tri_reid/bin/python | /data2/gb/Re-ID/dataset |
| 2026 | /data/gaob/Re-ID/Trifusion | /data/gaob/Re-ID/conda-envs/tri_reid/bin/python | /data/gaob/Re-ID/dataset |

实测容量见logs/foundation_progress738_20261002/RESOURCE_POOL.json。20:24的2026四卡空闲，无compute进程，磁盘约48.91GiB；20:25的2025剩余约1.032TiB，仅F1的RGBNT100两项仍运行。

后续成对端按已登记计划分配到两台服务器，每台最多四个单卡任务；不因增加卡数改变单模型batch、训练轮数或评价协议。登记时核对每台实际磁盘与运行占用。已有F1由唯一控制器233821完成，不迁移、拆开父队列或重建已有训练；F1六端报告与科学决策完成后，后续新批使用两台资源。当前没有新的2026训练启动。

观察按实际预计完成里程碑或180—300秒，不因观察超时重启。日志、模型及距离留各自服务器；代码与文本证据统一发布。此文件是资源/授权记录，不是环境重新验收或科学机制通过。
