# 末层角色bias的实际参数路径与封存值核查

本项在读取两份实际best参数之前登记，只分析原RGBNT201/seed42/gradient、random两份E8及§933保存区域／读出。训练STOP；不会创建模型、图像forward、M0、loss/autograd、优化器、改变后的检索排名或训练。

问题：§934只证明输出特征层面的共同平移任务对称性，未证明真实可达参数方向。当前最终Mamba `output_norms[2]` 的LayerNorm bias在传播之后，后续只有结构化线性读出。保持其他参数、global和gain不变，令该128维bias增加δβ，则每个样本均增加同一个correction：

\[
\Delta c_{m,r}=W^M_{m,r}\delta\beta,\qquad
\Delta h=\gamma A\delta\beta,
\]

其中 \(A\in\mathbb R^{1536\times128}\) 是3模态×4组×128输出对应的Mamba输入列。该末层参数不改变CNN/Transformer或上游Mamba传播，因此不同于CNN/T bias的整路径干预。对§934同batch原任务，可构成有条件的真实参数任务平坦子空间；不将任意1536维平移都称为参数可达。Adam参数weight decay及浮点实现仍在该结论范围外。

固定核查：checkpoint中的末层bias、readout和gain；与§933读出子集逐值一致；双精度还原保存区域的correction误差≤1e-5；固定第一坐标单位bias改变后的线性平移误差≤1e-12。数值秩使用 `max(shape)*eps64*smax`，报告阈值及全部奇异值，不以目标rank调阈值。

报告两端、Q/G各自的bias范数、映射数值秩、实际已学bias的直接读出范数、与描述性common correction范数的比值、单独bias分量能量／原correction能量，以及固定e0方向对现存L2向量的解析响应。后两种比值不是信息、噪声或可加的解释比例；query/gallery均值只作描述，不用于校准或评分。

无新增mAP/R1、无变体对照或调参；也不根据结果自动删除bias、中心化、改BN/head/loss、重启训练或扩大车辆/seed队列。即使路径确实存在，也要区分“能够变化”与“当前已学值在实际误差中重要”。若本bias很小，不能继续把它当共同能量的主要来源。

仅26已安装conda环境，CPU、CUDA_VISIBLE_DEVICES为空、两CPU线程，无GPU/功率/温度查询、安装、源码覆盖或权重删除。原best/输入/§933区域保持SHA；保存小型JSON与调用记录。所有原失败保留，不因观察超时或局部失败重跑。
