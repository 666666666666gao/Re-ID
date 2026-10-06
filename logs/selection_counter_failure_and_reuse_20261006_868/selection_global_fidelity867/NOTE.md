# RGBNT201基础参照的离线全轨迹核对

新26 global-only与历史F1作者来源配方，2649步所有记录字段（epoch、batch、loss、head_losses、lr）及完整50轮除计时外的字段精确相同。共同视觉/camera/heads/model初始状态记录相同；协议仅dataset_root从/data2/gb映射到/data/gaob，其余包含有序records、query_rows、label_map、counts、filter等全部字段相同。

两端同为第27轮mAP-best，73.4728/77.1531/85.8852/89.9522；正式距离的原回执SHA相同，当前26二进制此前实际核验过，旧25仅使用已接收回执而没有重新访问。checkpoint文件SHA不同，未重载旧binary，不能唯一归因于schema/provenance差别。

原masked退出前53步loss的均值4.445760803402595与修后已观察的第1轮均值精确相同。这只核对聚合量，修后逐步记录尚未接收，不能当成梯度/增强字节或完整训练等价性。

这是同seed的基础配方和修订保真证据，不是新增模块增益、多种子、泛化或SOTA。所有分析仅消费本地已有文本，无SSH/NN/新的检索评价，不修改活动实验。公开归档待当前NN/报告闭合后执行。
