# RGBNT100 原完整三角色独立训练比较合同 v1

登记时间：2026-09-06T13:56:31.328098+08:00。状态 **REGISTERED_NOT_RUN**。
Signal R2完整B0通过全部权重/数组/8675排名/7794更新执行侧核验；三角色尚未执行。
这是第三数据集的完整主模型比较，不是结构消融、B0重训或已失败数据集版本重跑。

## 主张、对照与边界

唯一待测主张：从对应source训练的完整Signal出发，原CNN/Transformer/Mamba残差银行
能否在RGBNT100新的身份与完整干扰图库上稳定改善检索，而不只帮助少数身份。
最小证据是下列固定五项条件全部通过。本比较不能排除额外参数/计算的解释；
容量、单/双角色、融合、多seed与效率消融按用户要求留在主结果成功之后。
不声称零样本域迁移、官方复现、SOTA或新frontier primitive。

对照为三折固定epoch30 Signal R2，内部89.52417504200771 mAP / 96.82997118155619 R1，
fold mAP82.73298803596518/90.29154465089529/96.81599006034025。
不重训或换选checkpoint；每fold只加载自身B0，角色和七个分类头seed42新初始化。
B0 summary：/root/trifusion-storage/artifacts/rgbnt100_signal_source_oof_v1_r2_storage_retry_seed42_20260906/baseline/summary.json，
SHA 549476408e82f085e7987fe087f6784773a54cd8f7eaf82e4e01faefd85711d5；三checkpoint及retrieval_arrays.pt的整个文件SHA由summary绑定，入口逐一核对。
执行侧完整核验见evidence/trifusion_rgbnt100_signal_v1_r2_baseline_executor_closure_20260906.json。
独立审计服务目前不可用，没有本B0/新三角色独立verdict，执行者不得冒写。

## 固定输入、原模型与损失

新配置：configs/RGBNT100/TriFusion-source-oof-v1.json。
基线配置configs/RGBNT100/Signal-source-oof-v1-r2.json，SHA 9d5ecf5f350c2f1eea50650bf0abc9580a4d2e86947e20a83b7ba94c67faf997。
协议protocols/rgbnt100_train_oof_v1.json，SHA 42bd612ecc8720db7f6684214e1f60d1cb4bab6b2fa8db3df343a9d2c52e4abf。
原Signal commit cd1b0a672d1fe642e7608731cb4899a19dda7d51、21源文件、
10项目基线源绑定与整个CLIP权重由configure原样验证。
角色入口、四核验器和实际远端依赖另绑定；五个已有本地CRLF/远端LF差异记录实际SHA/相同AST，
不改源码或运行时归一化。

沿原signal_preserving_collaborative_v8_expert_formation，semantic768/feature512/adapter128，
输入高128×宽256、grid8×16、block8后分叉，共享冻结tail9/10/11参数但三路径分别执行。
全部Signal及camera SIE冻结；仅原角色模块、可训练BN项与七分类头更新。
baseline3072D、完整branch4608D、三角色残差银行4608D、fused7680D。
完整branch是Signal+该角色残差，不是残差单独检索；等能量拼接让一半相似度来自残差，没有不下降保证。
没有Router/HFER、V23模态MLP、V24原型、投影、蒸馏或新损失。

沿原ExpertFormationV8Criterion及weighted_training_loss：七组ID/Triplet，
fused ID0.25/Triplet1；每完整branch及residual ID各1/12、Triplet各0.25，
margin0.3、label smoothing0.1，FP32求和分组不改。
AdamW LR0.00035/wd0.0001、AMP初始scale256、seed42，正式20epoch/5warmup，
原learning_rate_multiplier不改；M0恒定基础LR。
source33/33/34类对应静态预测trainable6248460/6248460/6274572，
仅源码计数预测，须M0核对实际参数与203可训练张量，不能当作实测FLOPs。

## 数据与唯一终点评价

复用RGBNT100 R2 montage loader：768×128拼图切RGB/NIR/TIR各256×128，
camera1–8映射0–7、view=-1，三模态同步flip/crop、独立erase。
source33/33/34身份、5550/5725/6075记录；heldout17/17/16身份、3125/2950/2600记录。
合计8675内部query/gallery、50身份，每条记录恰作一fold heldout。
完整图库不按容易query或身份裁剪；只删同身份且同camera，保留其他同camera身份。
不用MSVR310 scene过滤/re.txt。B64/K8、workers4，以每步真实source索引计采样成本。
B0跨camera正对1502928/1745856=86.0854503464%是已观测完整覆盖，不是因果结论。

每fold固定epoch20后strict reload，一次clean前向全部gallery，query取同一数组相应位置。
首先断言baseline全部features及distance逐元素等于原B0，
再保存五输出全部features/平方欧氏distance/NumPy默认argsort/无损gzip全排名、
逐query AP和首正例rank、作者eval_func交叉核对。合计43375 query-output/50身份。
不跨fold计算距离，不读官方1715/8575或RGBNT201 dev，不做reranking、缺失模态填补、
test-time adaptation或中间heldout选择。
同PyTorch2.5.1冻结SIM导致mm/bmm dispatch变化的既有实测问题，
用未改动msvr310_exact_signal_inference的functional投影视图恢复B0数值，
不变更注册参数或冻结状态；RGBNT100必须真实验证，不借MSVR310 PASS。

## 执行顺序与工程门

1. 复用同源R2 T0全8675文件/26025切片/全部camera mask真实回执，不重复无变化测试。
   本地只做文本/JSON/AST/SHA和全量标量；模型、张量、图像均远端。
2. 三fold fresh角色M0，各8条clean source独立Signal/前缀核对及各8步B64/K8容量，共24更新。
   每fold保存/严格重载后8source五输出逐元素相等；合计clean角色72、独立Signal24记录前向，
   heldout/dev/official0。203可训练张量须有限非零梯度，0 AMP下降，
   Signal及全部冻结state不变、可训练state实际更新，peak reserved<24GiB。
3. 再从fold0 fresh初始构造，一个固定增强B64/K8 batch恰好100步过拟合；
   原七头label-smoothing解析熵下界，最后一步excess ratio<=0.1。
   不选中间最小值、不延长或重新抽batch。M0共124更新/7936source训练记录曝光。
   每步loss/source索引/AMP/是否更新/全部gradient finite先落盘再判断工程停止门。
4. M0 summary PASS_ENGINEERING_ONLY且远端完整权重、本地124步全部算术核验通过，
   才执行三fold×20epoch；fresh模型初始SHA同对应M0初始化，不加载M0训练权重。
   工程失败封存现场，不改变门/LR/宽度/seed/步数或扫描重跑。
5. 完整终态保存3权重/15数组/43375排名及全部真实逐步日志；
   远端CPU内容/源绑定核验、本地全AP/CMC/bootstrap/FP32 loss/采样重放。
   工程PASS不是科学PASS；独立审计按实际可用性保留未闭合限制，不阻止授权工程执行。

## 固定五项科学支持条件

全部满足才写内部SUPPORT_PASS：

- fused查询加权mAP相对Signal至少+1.0个百分点；
- 每fold fused mAP增益非负；
- 每完整CNN/Transformer/Mamba的加权mAP均不低于Signal；
- 50身份聚类bootstrap下界严格>0：seed42/10000次/2.5%分位/linear插值，
  重抽整个身份、保留每身份全部query权重；
- fused mAP严格高于baseline与三完整branch。

保留各身份与每query AP变化、Rank-1修复/新增错误。失败封存，不调本配置重跑。
正结果仅支持本内部协议单seed，不自动解锁RGBNT201官方或主结果后消融。

## 预算、产物与里程碑

B0前置7794主干更新另计，本比较不匹配与Signal的总训练预算。
M0估计3–8分钟；正式约5190更新、实际以完整sampler日志为准，
按MSVR310实测约1.3秒/步粗估100–140分钟，以本M0更新ETA，不减少epoch/身份/图库。
新产物仅/root/trifusion-storage/artifacts下独立私有目录。
此前旧数据盘实际ENOSPC，因此启动前检查新卷>=8GiB可用、GPU无其它训练且free>=22000MiB；
不删除/覆盖B0或原失败产物。m0/和comparison/各有wrapper/launch/log/exit/terminal。
长任务180–300秒以上按估计里程碑观察，终点前数分钟检查，不逐epoch重复轮询。

| 阶段 | 问题 | 状态 |
|---|---|---|
| B0 | 同源基线是否完整可信 | COMPLETE，文件/7794步/8675query执行侧PASS |
| M0 | 本数据上原三分支工程/优化能力 | REGISTERED_NOT_RUN，固定124步及全部核验PASS后继续 |
| 完整比较 | 残差银行是否跨身份稳定改善 | NOT_RUN，三fold×20epoch唯一终点和原五门 |
| 后续主结果/消融 | 总体主张及容量解释 | 主目标成立后另登记，当前不执行 |

RGBNT201保留dev58.4050/59.3939，65与官方目标未达；
V23/V24/MSVR310负结果、单seed、同家族审计和远端二进制限制保持。
