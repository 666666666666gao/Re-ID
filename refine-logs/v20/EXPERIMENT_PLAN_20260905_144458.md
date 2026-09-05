# V20 每专家跨模态身份监督主实验方案（2026-09-05，执行前冻结）

## 问题、主假设与解释边界

V19三折两端完整训练未通过Q1，随后全六端只读诊断及独立审计显示：
所有source身份的七路分类准确率、五路source检索均100%，而heldout fused
mAP约80%。这支持来源拟合与身份泛化有明显落差，并不证明唯一的失败原因。

主假设：保持原V8 CNN局部、Transformer全局、Mamba跨模态空间角色不变，
在每个专家内部加入真实身份的跨模态监督，可以改善未参与该fold完整训练
路径的身份检索。最低证据是对本次实际匹配对照通过下面全部Q1科学条件。

跨模态cosine较低只构成待检验方向。原concat距离只比较对应模态；对每个
模态独立作正交旋转可改变跨模态cosine而不改变原检索距离。因此V20是一个
泛化正则假设，不是数学上必需的修复，也不能预先声称它解决V19的唯一原因。
不宣称监督对比学习本身新颖，不把当前OOF结果等同独立dev或官方结果。

## 原模型与唯一干预

- 两端均从对应V12 fold的94-source Signal与V8 expert最终checkpoint严格
  恢复；source六权重与V12汇总SHA固定。不能加载V19的私有尾部训练终态。
- 完整3072D Signal direct+SIM及camera SIE冻结。V8在CLIP索引8后取得
  anchor，保留共享冻结索引9/10/11尾部，交替原CNN/T/M角色模块。
- 每专家每模态512D。专家仍减去冻结reference，保持原角色head和BN。
  fused为3072D Signal + 4608D等能量残差银行；单专家为3072+1536D。
- 不新增推理参数、私有尾部、投影、Router、HFER、mask、文本或外部backbone。
  原baseline-only、fused、CNN、Transformer、Mamba五路输出必须全部保留。
- 实际训练两端为identity_concat（新损失权重0）与cross_modal_identity
  （新损失权重0.25）。两端同一模型/参数量/初始化/训练预算/基础损失；
  两端均计算并记录新损失，只有反向总损失中的系数不同。
- 此比较能排除“新增模型容量”的解释；仍只能支持这一固定完整训练目标，
  不能将结果外推为唯一机制或跨数据集效果。主实验达标前不做消融和扫描。

## 新损失的精确定义

对专家e的单位化模态残差z[e,i,m]，真实身份为y[i]，固定温度tau=0.07。
对所有三个专家、六个有向不同模态对(m,n)，定义：

    p[i,j] = exp(cos(z[e,i,m], z[e,j,n]) / tau)
             / sum_k exp(cos(z[e,i,m], z[e,k,n]) / tau)
    L[e,m,n] = mean_i mean_{j: y[j] == y[i]} (-log p[i,j])
    L_cross = mean_{e,m!=n} L[e,m,n]
    L_total = L_V8 + lambda * L_cross

同一triplet的不同模态是有效正例；同身份其他实例也是正例。B64/K8保证每个
anchor在目标模态有8个同身份正例和56个不同身份负例。既有cross-camera
sampler继续提供真实跨camera正例；不把同身份的其他实例误当负例。
不同专家之间没有对齐项。新损失无可训练参数，cosine/log-softmax显式FP32。

数学来源参考作者SupContrast多视图监督对比目标：
https://github.com/HobbitLong/SupContrast/blob/master/losses.py
本地实现独立编写，限定到上述六个跨模态方向，不复制其通用fallback逻辑。
UPCL/MixStyle比较与数据/代码许可范围见
docs/GENERALIZATION_MODULE_SOURCE_NOTES_2026-09-05.md；不移植其模块或混合数据。

## 固定训练合同

配置：configs/RGBNT201/TriFusion-signal-preserving-v20-cross-modal-identity-rtx3090.yml。
启动时验证本方案/配置的外部SHA，记录实际HEAD、runner、模块和依赖源码SHA。
V19 runner仅复用source绑定、criterion、seed及冻结state函数，不实例化V19模型。

| 项目 | 固定值 |
|---|---|
| seed /设备 | 42；远端单卡RTX3090 |
| 实际batch | B64/K8，无梯度累积，workers4 |
| 数据 | 固定141-fit registry共3126triplet；每fold94-source/47-heldout |
| sampler/增强 | 现有CrossCameraIdentitySampler、SharedGeometryTripletTransform |
| 两端预算 | 每端完整20epoch，只评价最终checkpoint，无中途检索或best选择 |
| optimizer | AdamW；全部原可训练角色模块/分类头LR0.00035，weight_decay0.0001 |
| scheduler | 原5epoch线性warmup、其后cosine |
| AMP | FP16，GradScaler初始256，原gradient checkpointing |
| CuDNN | deterministic=true，benchmark=false，两端相同 |
| eval | batch128，原256×128 resize/normalize，无reranking |
| 新损失系数 | 对照0，实验端0.25；温度0.07，不扫描 |

基础损失完整沿用V8七路ID与batch-hard triplet：fused ID0.25/Triplet1；
三branch各ID1/12/Triplet0.25，三residual各ID1/12/Triplet0.25。
margin0.3，label smoothing0.1。身份权重和0.75。两端同样记录基础损失、
新损失与总损失，不能只报告总损失下降而隐去身份目标恶化。

## T0 / M0：训练内工程门

T0仅在远端CUDA执行三项数学测试：8个同身份目标的解析熵；
身份重标号、batch/模态排列不变性；三个专家和全部模态的有限非零梯度。
本地只做Python AST、文本、JSON及算术核验，不执行模型训练/推理。

M0顺序固定：
1. 三折两端各取前8个真实source batch。全部初始state、五路输出SHA、
   增强receipt、绑定和正例数量两端精确相同。所有batch每anchor有8正例，
   有跨camera同身份正例，完整Signal前缀精确，eval状态不变。
2. 每模型最后一个batch单独反传新损失，检查每个专家encoder有有限非零
   梯度，baseline无梯度，模型state不变。该probe使用FP16前向/FP32新损失。
3. fold0两端分别8个不同batch真实训练步；全部实际训练tensor必须在8步中
   有有限非零梯度，overflow0，冻结参数/Signal buffer SHA不变；
   各端峰值reserved显存<24576MiB。真实参数量/tensor数由builder报告。
4. 从source重新构建实验端，固定同一真实batch完整100步，base LR不warmup。
   ID平滑熵下界为0.75H，K8均匀正例的对比交叉熵下界为log(8)，总下界
   F = 0.75H + 0.25log(8)。有限温度可使实际最优值略高于该下界。
   固定要求 (loss_final-F)/(loss_initial-F) <=0.1，并保持全部梯度覆盖、
   overflow0与冻结状态不变。记录全部100步三个损失分量和解析下界。

M0不通过即保存M0_FAIL并停止Q1。实现错误可按具体报错修复；真实负结果
不能通过换温度/权重/LR、换batch/RNG或放宽门槛改成通过。M0权重不进入Q1；
所有正式端点从其V12 source checkpoint独立重建。

## Q1：完整身份隔离OOF主比较

M0通过后执行三折×两端×20epoch，共3360优化步（每端580/560/540步）。
每端保存完整非baseline参数与BN buffer，绑定冻结Signal、配置与方案SHA；
必须从新建模型strict reload，并验证完整state SHA等于最后训练state。
完整sample-order SHA、前8增强batch与初始state须配对相同；两端baseline
检索输出对象完全相同。任何源权重文件不能被改写。

每fold全部47-heldout身份保留为gallery，包括单camera身份干扰项。总gallery
3126记录，合法query571条来自21跨camera身份；2555条只排除于query分母，
不能排除于gallery。评价原五路全部mAP、Rank1/5/10以及所有query AP/rank，
报告完整三个fold、全部21身份，不删除负收益身份或择优分支，不混跨fold距离。
aggregate按全部571query加权。所有固定科学门必须同时满足：

- 实验端相对本次实际对照fused aggregate mAP增益>=1.0个百分点；
- 三个fold的fused mAP增益均>=0；
- CNN、Transformer、Mamba各自aggregate mAP增益均>=0；
- paired AP差以身份聚类bootstrap10000次、seed42、显式float64，
  95%区间下界>0；
- 实验端fused aggregate mAP严格高于同checkpoint baseline及三个expert。

继续复用已消费的train-internal complete-path OOF开发资格；不能称新独立验证。
必须六端全部完成再作科学结论，中间结果只说明进度。任一科学门失败即封存，
不执行D1/dev/official，不扫描温度、权重、epoch、RNG或恢复sealed版本。
若全部通过，另行固定141-fit refit与一次30-dev终态合同；本runner不执行D1。

## 成本、顺序与目标状态

| 阶段 | 预算与预估 | 进入条件 |
|---|---|---|
| T0 | 3项远端CUDA单测，通常<1分钟 | 已冻结并发布源代码/方案 |
| M0 | 6模型配对probe+16容量步+100固定batch步，约3–8分钟 | T0全通过 |
| Q1 | 六端20epoch及终态评价，暂估60–80分钟 | M0全通过 |
| 全量终态复算/独立审计 | 所有数组、mask、配对、权重来源和科学门 | 完整终态 |
| D1/dev | 条件性后续合同 | Q1全部门通过 |

启动前GPU至少22000MiB空闲；当前远端约8.5GB空闲磁盘，六个V8非baseline
checkpoint预计约0.2GB，足以保留，不清理旧证据。稳定epoch时长取得后修正
ETA，进度采样间隔180–300秒或按完整端点里程碑估计，不反复短轮询。

可部署30-dev最好仍V8 Phase-B58.4050mAP/59.3939R1，exact Signal58.0109/
57.4545；65开发门和官方SOTA目标未达。V19保持Q1_FAIL；RGBNT100/MSVR310
只完成安装与标签协议核对，无训练/检索成绩。用户总体目标继续active。
