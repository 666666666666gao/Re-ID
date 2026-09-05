# MSVR310 source-only Signal 三折基线合同 v1

登记时间：2026-09-06T04:17:46.871857+08:00。状态 PREPARED_PENDING_REMOTE_T0_M0。
这是新数据集的同源基线建设，不是TriFusion新机制或V24晋级；RGBNT201整体目标仍未达到。
本计划不包含正式官方评估、结构消融、多种子、车辆候选或已有失败版本重跑。

## 唯一主张及证据边界

要建立可复用、完整路径身份隔离的MSVR310 Signal固定终点基线。
每折模型及分类头只见source身份，之后在该fold完整held-out gallery评估一次；
完整权重/配置/原始逐query指标可核查后，才称本项目具有车辆内部基线。
不以任何内部mAP数值自动认定新方法成功，也不与公开官方53.2/72.4直接相减作复现差。
不为这个基础建设阶段设置或扫描新的mAP晋级阈值。

## 已固定数据

使用protocols/msvr310_train_oof_v1.json，SHA
4ff4c60bca3d019929add5788212c526387d93d535a2c52aa7b1c3acfd387cb4。
source103/103/104身份、672/683/709记录；每折40个跨scene source身份。
held-out gallery360/349/323记录，query210/207/183；合计600query与1032完整gallery。
95个单scene身份的432记录继续作为干扰；同身份同scene才从排序排除。
camera0–7用于SIE；scene保持真实值，不用camera替代scene过滤，不跨折计算特征距离。
只有真实官方训练目录bounding_box_train参与本合同。没有官方测试图片读取。

## 模型与优化

从固定通用CLIP文件重新初始化每折Signal，不继承RGBNT201、V24或另一个fold的学习权重。
Signal commit cd1b0a672d1fe642e7608731cb4899a19dda7d51，
既有整体diff SHA b889caca9c4a92689b13eb7e20bd3224067f3e5ed2a3db6825201870ca422741；
16份实际源码/配置文件SHA单独绑定于configs/MSVR310/Signal-source-oof-v1.json。
其中make_model_clipreid.py为已存在的本地权重路径补丁，不冒充Git原blob；本次没有更改Signal源码。

沿用原MSVR配置：高128×宽256，8×16 Patch，DIRECT=0的模态分类头，
USE_A/USE_B均启用，Gram0.2、Patch0.01、各头ID0.25与Triplet1；
CLIP视觉主干按原FROZEN=False训练，camera SIE启用、view SIE关闭。
DIRECT=0没有删除检索中的直接特征：评估仍拼接三模态全局特征与SIM，总3072D。

Adam基础LR5e-6，保留原参数组规则；车辆classifier是基础LR的100倍，
其余bias/base分组以实际make_optimizer输出为准并完整记录。
调度必须使用原train.py的MSVR分支WarmupMultiStepLR：
epoch20/40起分别衰减，gamma0.1，无warmup，保持原step(epoch)时序。
不能复用当前RGBNT201辅助训练函数中的固定cosine调度。

seed42，B64/K8，workers4，AMP初始scale256；不扫描seed、LR、epoch或batch。
相对原公开K4配置，K8为项目现行固定约束；增强改为项目已有三模态同步几何、
模态内独立擦除，尺寸128×256。明确这两项差异，不称完全照抄作者全部训练条件。
原RandomIdentitySampler保留，无新增跨scene采样规则；保存每次真正抽取的record indices，
以实际optimizer步数计账，不能用sampler估计长度代替真实batch数。

## T0 / M0 / 基线执行顺序

1. T0：仅远端运行两项scene排序协议回归；人工可解的合成距离fixture，
   明确标为测试，不冒充真实数据指标。另在真实协议上核对全部source/held-out和query masks。
2. M0：三个fold分别从新初始化开始，每折真实B64/K8恰好8次优化更新。
   记录所有trainable/gradient张量、参数组、每步原目标组成、输入indices、AMP与峰值显存。
   每折只在8个source记录上做训练后与严格重载后的3072D特征比较（合计48条clean source前向）。
   held-out/dev/official前向均0；M0权重单独保存，正式基线不用M0权重。
   工程PASS要求三折均8步、无overflow/非有限量、没有缺失梯度的trainable tensor、
   模型状态确有更新、strict reload后源特征逐元素一致、真实输入为128×256。
   M0不是固定100步拟合门或新机制有效性检验，不能沿用V24的CE下界/科学门解释它。
3. 固定三折基线：仅在同配置/同脚本M0回执PASS后执行，每折重新初始化，训练完整50epoch。
   只保存第50epoch作为基线终点，严格重载；此后一次性提取对应fold全部gallery。
   query与gallery为同一清单中的确定图像，query从同一次clean gallery特征中按固定位置读取。
   三折合计1032held-out图像前向；不逐epoch查看held-out，不选best。
4. 按原作者eval_func_msrv交叉核对全量mAP/Rank1/5/10，与逐query原始AP及首个正例排名一致；
   原作者函数会写re.txt，运行目录固定为本次新建fold artifact目录，避免覆盖项目中的其他文件。
   特征、距离、checkpoint留在远端，JSON逐query数值及文件SHA可归档本地；
   完整终态后做独立审计，再登记后续配对方法合同。

训练入口tools/train_msvr310_signal_oof.py，原损失计算复用现有
tools/build_v12_complete_path_oof_targets.py::_signal_training_loss；车辆优化循环独立且固定。
没有新loss、风格混合、原型记忆、Router、三分支训练或可学习融合。

## 判定、成本与停止规则

M0任一条件失败即停止相应正式三折启动，先定位有证据的实现问题，不添加fallback或宽松通过分支。
正式基线完整50epoch/三折、权重与source标签绑定、原协议算术核验和完整原始结果是完成条件；
低mAP也如实封存，不延长训练、不调参、不选其它epoch，也不据此判断TriFusion机制失效。
基线不是新主结果，不解除RGBNT201现有晋级限制。

初步估算M0约3–8分钟，150个fold-epoch约45–120分钟；尚非实測MSVR耗时。
按实际M0和首轮耗时更新预计完成时间，长任务180–300秒以上的阶段检查，
优先在预计结束前几分钟核对，不逐步频繁轮询。当前训练和检索次数仍0。

## 当前执行状态

代码仅完成本地AST解析，没有本地import模型、张量、图片或执行检索。
远端T0/M0及正式基线均NOT_RUN；需要先发布固定源码/合同并记录真实启动回执。
