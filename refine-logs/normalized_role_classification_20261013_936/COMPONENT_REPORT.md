# 只改角色分类入口：实际CPU合成head检查

**新最小入口已实现，第一次实际CPU检查通过；尚无真实初始化、M0、50轮或新检索成绩。**训练仍停止，训练范围确认问题未收到人类回复。本结果仅收窄下一项值得检验的训练入口，未证明模块有效或三数据集超过Signal。

## 已执行什么

实际在2026服务器现有conda中用CPU执行一次`check_normalized_role_classification_cpu.py`，2026-10-13T21:19:25.089064+08:00→2026-10-13T21:19:27.355814+08:00，EXIT0。CUDA_VISIBLE_DEVICES为空，Torch 2.5.1+cu121、FP32、2线程。只实例化合成evidence stub和真实原/新增head wrapper；没有图像、CLIP、Mamba、真实权重、优化器或检索评价。

4行×1536维、2身份，固定classifier权重与shift=[0,2,0,...]。四个case共8次合成evidence forward、8个训练BN/classifier head调用、8次CE向量—雅可比积；另以原作者TripletLoss前向核对raw困难距离。不存在真实模型训练、优化器更新、权重写入/删除或指标重选。“没有全模型NN”不等于没有合成head前向和autograd。

## 检查结果

| 分类入口 | 固定shift | 角色CE | raw Triplet | 共同平移方向CE梯度最大绝对值 |
|---|---:|---:|---:|---:|
| 原raw分类 | 0 | 0.6226119399 | 0.2736792564 | 2.980232239e-08 |
| 原raw分类 | 2 | 0.6226119399 | 0.2736792564 | 8.940696716e-08 |
| 角色L2分类 | 0 | 0.5797339082 | 0.2736792564 | 0.1568222791 |
| 角色L2分类 | 2 | 0.5347117782 | 0.2736792564 | 0.2538129687 |

原raw分类两次logits最大差为1.78813934326e-07，候选为0.248234391212。预设判据全部满足；这只是一个有限数值见证，不证明消除了所有盲区或在真实训练中获得有效身份补充。

两wrapper对应case的global logits、raw特征、部署L2向量、原作者困难正负距离及Triplet逐值相同；state键/形状、参数量相同。合成总参数21505、可训练16897只属于fixture，**不是实际TriFusion模型参数量**。实际新增wrapper不注册参数。

角色CE只对correction/gain有非零有限梯度，不达shared/global BN权重/classifier权重；global CE只达shared和global head，不达correction/gain。持久BN计数为1，running mean/variance符合global一次更新的预设浮点容差；角色使用原`detached_head`克隆buffer，eval不更新buffer。参数逐值未改变，单次外部forward只调用一次合成evidence。

## 为什么不与旧实验混写

新入口为**角色L2(h)→现有BN→现有停止梯度classifier值，角色Triplet仍raw h**。global作者任务、角色读取与推理L2(g+gain*c)保持。BN仍在L2后，不能称直接余弦分类、完全同部署几何或NormFace复现。F2/F3是无角色current基础配方；joint-L2改的是角色Triplet；独立角色头已经三集失败。本项不是把上述旧尝试当新方向。

训练BN/raw Triplet共同偏移与L2检索差异在旧2026-10-05计划中已写明，§934是复核，不声称首次发现。§935当前bias贡献极小的结论仍有效。本候选没有删除bias、以测试均值中心化、调gain、margin、层数、seed或读取概率。

## 失败与资格边界

第一次实际CPU调用成功，不存在组件重跑。此前三个本地launcher检查失败及一个只读哈希检查失败，均发生在SSH之前：JSON字段、Windows/远端字节seal范围和Git的EOL状态语义处理不正确。原tool退出和后续修订均记录于EXECUTION_PROVENANCE.json；它们不是训练或算法失败。远端474项封存来源检查前后相同，本地366项可用来源按独立Windows字节pin保持；未覆盖远端的108项不被假称为本地文件。

新增生产入口仅支持201 semantic/direct，driver尚未实际prepare/M0或生产运行，根源源码检查不等于独立源码审计。若恢复训练，须先核实初始化/optimizer所有权/真实BN与8步M0，再运行一项固定seed42 fresh50。原raw semantic74.3749/78.8278、global-only74.2967/78.9474与完整Signal80.3029/85.1675的记录保持，本次没有替换或超过这些成绩。

用户目标仍为三个数据集全面超过Signal、形成针对问题的有效原创机制。此次改变只是有证据指向的训练接口对照，不作为论文创新；既不因toy通过宣布成功，也不因仍无正式分数取消总体目标。训练启动确认待回复；不自动部署车辆、多seed或新模块队列。没有本次生成的无用权重可清理，所有原best/必需依赖保留。
