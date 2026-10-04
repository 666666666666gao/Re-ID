# 当前冻结版本的读出与槽位职责（只读源码核对）

此记录针对 role_input_detach_v1 的冻结322文件清单，15份实际源码字节与缓存清单一致。未导入 torch/模型，未读取权重、运行训练或访问服务器。不是性能实验，也不修改任何已登记实验。

1. 当前 semantic/native 的继承链到 ContextIdentityTriFusion 时明确启用 structured_readout=True。最终读出没有把全部模态、槽位平均成单个384维输入。每角色 B×3×16×128 先按4×4槽位索引分为四组，每组平均四个槽位，得到 B×3×4×128；三角色在通道维拼接成 B×3×4×384，再用12个128×384权重块各自投影并拼成1536维。
2. 虽然保存的 readout.weight 形状是1536×384，当前作用是12个不同输入块上的分组线性映射。不能仅凭这个保存形状断言最终全部修正被限制在一个全局rank128或rank384子空间。等效分块映射在维度上可以达到1536秩；这不证明实际训练权重的秩或判别性。
3. 四组的“区域”目前是学习槽位的索引分组，不能直接当作四个真实空间/语义部件。当前实际 FP32SlotCompetitionRoles.sample_context 覆盖旧采样：读取全部128个patch，采用每槽位对patch的独立softmax，不使用传入positions或local_support。offset参数冻结，返回的positions是固定参考点；旧grid_sample路径不是这个版本的实际角色读取。
4. native 细节分支读取全部512个CNN位置，具有独立key/value。细节query由anchor_queries加context_queries中的CNN角色项生成，并扩展到三模态。context_queries reshape中的3是角色索引，不能解释成三个模态各自的query。当前细节reader没有显式局部位置mask或额外位置编码；卷积网格及CLIP路径仍具有空间来源，不能泛化为整网没有空间信息。
5. 新细节在CNN出口LayerNorm之前相加，CNN证据还经桥接影响Transformer/Mamba。仅删除最终CNN读出分量，不等于删除CNN/细节路径。Transformer使用每模态一个固定上下文token加16个槽位；Mamba使用48个空间/模态token及共享参数的正逆扫描。
6. input-detach切断角色输入对共享编码器的直接反向路径；h=g+gain*c中的g路径仍训练，且梯度受融合损失影响。不能将其描述为严格保护global不变。

以上是当前代码的结构性质。它们不能唯一解释负增益、证明对应失效、槽位塌缩或指定下一项干预。等待全部六端及固定best诊断完成后，再据完整证据决定唯一下一实验。
