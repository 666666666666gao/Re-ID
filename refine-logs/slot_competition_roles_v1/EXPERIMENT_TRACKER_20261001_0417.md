# 槽位竞争执行状态

2026-10-01 04:16:51实测。Goal ACTIVE / UNMET。

R1：competitive RGBNT201八批M0失败exit1，缺少Mamba内容Q/K两个参数非零梯度。原FAILED与两个未派发候选保留，三个已通过M0的independent控制继续完整50轮，不自动重试。

R2：提交a885302abd9b5be9eb4d0bf012ecaa52b5eb6558，controller1394231于04:11:53实际启动，目录logs/slot_competition_fp32_roles_20261001_r2。两个模式的完整注意力子图均FP32，其余合同不变。213个运行源SHA通过，seed42每端fresh8批M0/full50，最高官方fused mAP-best同一权重严格重载。

| 新R2端 | 当前任务 |
|---|---|
| competitive RGBNT100 | GPU2；M0实际exit0、127/127梯度、重载差0；正式1/50完成 |
| competitive RGBNT201 | GPU3；M0实际exit0、127/127梯度、重载差0；正式6/50完成 |
| competitive MSVR310 | GPU0；M0实际exit0、127/127梯度、重载差0；进入正式训练、完整epoch尚未保存 |
| 三个independent R2 | 排队，释放卡后按100/201/MSVR启动，未有M0或正式成绩 |

GPU1仍运行原R1 RGBNT100正常控制；四卡占用10GB左右且任务存在，GPU0当次5%短时利用率不表示停止或故障。原R1代码、旧210源和正常训练未改。

持久observer1400686，首查2026-10-01 06:35 CST，未完240秒。source SHA bd1ca17580997c52385baf4d85ba661aae3eb048222086f601f237efca009ca6。预计完整面板约06:30–07:00，仅估计；不重复提前查询或重建观察器。尚无新六端正式结果，不将M0/中途best称性能成功。
