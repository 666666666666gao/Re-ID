# §41.961 接近Signal的车辆端：有限读出参数适配

用户原Goal允许结果接近时保持模块不变调整参数。实际资格：RGBNT100旧R2 mAP高+.6619，R1持平（1715条仅需净多1命中），R5略低1命中；MSVR旧R2 mAP低.5627，R1低2.0305（591条少12命中），R5/R10已更高。当前74.6337的公共CLIP端仍明显不接近，不参与本比较。

保持已完成50轮、固定seed42、唯一mAP-best、Signal-SIM/CNN/Transformer/Mamba/R2训练定义及所有权重不变。201原已通过，保持原ratio1及83.4284同best。车辆只调整最终既有residual bank相对baseline的幅度：预先固定ratio .25/.5/.75，不扩网格。没有新层、头、优化器更新、重训、测试均值或逐query gate。原ratio1结果保留。

源融合将bank匹配到baseline范数。由此在非零等能量条件下，改为Normalize([g,ratio*r])时，平方归一化欧氏距离为((1−ratio²)Dg+2ratio²Dold)/(1+ratio²)。CPU阶段只使用已保存完整距离进行固定有限比较，所有候选都报告；以四项严格超过Signal优先，再按mAP/R1选择一个参数。没有通过四项时，只记为诊断/部分进展，不宣布完整目标达成。

任何缓存候选尚不是新正式成绩。随后原模型每个选中端一次完整query/gallery编码：严格载入原role最佳与作者Signal，原model state不变、grad空、更新0；验证原ratio1和baseline回放、实际非零等能量，并用同次特征做实际ratio读出及作者camera/scene评价。距缓存的误差和原失败必须记录，不换checkpoint、改容差或挑重复编码。参数只是推理读出适配，不混写成新50轮/因果贡献或无偏SOTA。

原全Signal初始化是已有强路线资源；模块-removed Signal比较与该强参照保持资源/初始化说明，不把新74分public-CLIP与旧83分差值算作模块收益。正式目标仍三个数据集全面超过Signal，同一网络模块仅允许数据集参数适配。代码/交接及时发布，保持最优与唯一记录；仅2026GPU0/1、warm环境，无T/P或新seed。
