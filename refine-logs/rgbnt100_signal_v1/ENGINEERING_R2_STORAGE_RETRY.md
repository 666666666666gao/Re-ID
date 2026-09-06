# R2 完整首epoch M0 的存储恢复合同

登记时间 2026-09-06T11:36:28.223741+08:00，当前 READY_NOT_RUN。

e699eac 上的首次R2 T0完整通过，回执SHA为 e54c826d1cfba2eca6526d4e4bdecb6758a19ff3b0ada4a498adfb3f11883ba5。
M0 fold0完成81有效更新、5184 source训练记录前向，195梯度有限、AMPscale256不变。
11:08:23.174872 wrapper退出1，失败发生在torch.save；数据盘仅剩1785856字节。
101712000字节不完整checkpoint和完整日志保留，不加载或复用；它没有通过strict reload，也不算完整M0或正式训练。

已登记并完成两个已安装数据集下载压缩包的无损迁移。分别完整SHA校验前后相同才移除冗余原副本，
MSVR310.zip与RGBNT100.zip现在位于 /root/trifusion-storage/downloads。
实验权重/数组/日志、已安装数据集均未删除或修改；公共挂载写入0。
原数据盘可用2077556736字节，overlay可用17817534464字节。
原迁移回执的started_at字段是在copy之后赋值，是校验结束附近的时间；不是实际开始时刻。elapsed_seconds=11.049577126为完整操作实测时长。

新M0输出固定为 /root/trifusion-storage/artifacts/rgbnt100_signal_source_oof_v1_r2_storage_retry_seed42_20260906。
执行前确认该位置在owned overlay、至少8GiB可用；这是针对已经发生的写盘失败的明确容量检查。
不使用自动fallback，不改变R2模型、配置、固定Gram下限、AMP、优化器、采样、seed和各项工程门。
复用上述真实R2 T0回执；代码/config/SHA均相同，所以不重复全量T0。
三个fold均从通用CLIP重新初始化，各完整一个source epoch，不复用任何M0训练权重。
实际步数按全部steps.jsonl/training.json登记，不以len(loader)推断；此前81更新另记工程成本。

M0后依次执行新R2完整文件/权重内容核验和本地全部步骤标量/源索引重算。
文件核验仅远端读取保存权重，0模型/图像前向；本地只运行stdlib和JSON。
核对3份权重、每个实际步骤、195梯度、原selector不变、3072D严格重载以及source/heldout边界。
预计3–5分钟，约4分钟首次观察；不提前评价模型有效性。

完整终态核验器已在正式基线重启前适配R2配置、真实T0路径和逐步JSONL。
全部浮点损失按真实_signal_training_loss逐项FP32运算重组，并已在保存的81步验证完全相同。
正式30epoch仍需M0及核验真实通过后单独登记启动；此合同没有正式训练、未知身份检索、消融或官方测试。
