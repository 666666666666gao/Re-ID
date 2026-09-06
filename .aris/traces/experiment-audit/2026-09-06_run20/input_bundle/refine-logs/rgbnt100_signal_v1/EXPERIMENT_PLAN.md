# RGBNT100 source-only Signal 三折基线合同 v1

状态：PREPARED_NOT_RUN。登记于2026-09-06。
这是第三个数据集的独立同源基线建设；不属于新TriFusion方法、结构消融或V23/V24/MSVR原三角色负结果重跑。
RGBNT201 dev65和官方85.3/87.9主目标仍未达，不以本基线的内部数值自动晋级。

## 固定数据、完整检索与用途

真实官方训练目录为RGBNT100/rgbir/bounding_box_train，8675张768×128三模态拼图，50个身份501/503/.../599。
全量文件名、大小、SHA已从服务器读取；这次清单检查没有图像解码、模型或排序。
数据清单：evidence/rgbnt100_signal_train_file_inventory_20260906.json，
SHA50d5bb1ab2f24946ea30aa3f98d070e7e416017bc2930de08745d7b7c8b535e0。
协议：protocols/rgbnt100_train_oof_v1.json，
SHA42bd612ecc8720db7f6684214e1f60d1cb4bab6b2fa8db3df343a9d2c52e4abf。

按跨camera资格组内排序身份round-robin三折；实际50个身份全部跨camera，单camera身份0。
source身份33/33/34、记录5550/5725/6075；heldout身份17/17/16，query=gallery分别3125/2950/2600。
全部8675条记录在自己的heldout fold中评估一次；没有挑选容易query或去掉困难负例。
只过滤同身份AND同camera；同camera不同身份负例保留。camera减1映射0–7用于SIE。
view保持作者loader的-1，view SIE关闭；不得把MSVR310 scene过滤直接套过来。
不同fold距离单独计算，最终池化逐query AP/Rank；不跨fold比较特征。
官方query1715/gallery8575不参与source/heldout划分、训练、验证或checkpoint选择。

## 模型与作者配置

Signal固定commit cd1b0a672d1fe642e7608731cb4899a19dda7d51；
已有diff SHA b889caca9c4a92689b13eb7e20bd3224067f3e5ed2a3db6825201870ca422741。
包括既有本地CLIP路径补丁，完整实际源码SHA绑定于本配置，不改作者源文件。
CLIP ViT-B-16文件SHA5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f。
每fold独立seed42从通用CLIP重建，source分类头重建；无RGBNT201/MSVR310/M0/其他fold已学权重。

作者RGBNT100配置为高128×宽256、8×16 patch、DIRECT0、USE_A/B=True、TOPK112，
camera SIE=True、view SIE=False、FROZEN=False；检索仍为完整3072D direct+SIM。
Gram/Patch权重均0.1、各身份目标ID0.25/Triplet1，沿用原四组ID/Triplet及Gram/Patch调用。
与MSVR310的Gram0.2/Patch0.01不同，不继承车辆前一数据集权重。
原TokenSelection Q/K产生离散mask、W_v未调用，六个参数本来不被Adam更新；
显式冻结该原模块参数并核对其状态不变，沿用已查明的工程处理，不改前向选择机制。

Adam原参数组：配置BASE_LR0.0007；非adapter的CLIP base参数按作者make_optimizer固定5e-6，
其他bias按原因子，任务参数按原基础值；RGBNT100没有MSVR classifier×100规则。
记录每个参数组initial_lr及调度后的实际epoch LR，不能把所有参数都描述成0.0007。
完整固定30epochs；使用作者create_scheduler/CosineLRScheduler，
warmup5、lr_min=0.001×BASE_LR=7e-7、warmup起点7e-5；
保留作者seed42、epoch1–29噪声、t_initial30、cycle_limit1，epoch30落在lr_min。
保持processor.py中每epoch开始前scheduler.step(epoch)时序，不另写近似cosine。

项目现行约束为真实B64/K8、workers4、AMP初始scale256，无梯度累积，seed仅42。
作者原配置B128/K16/workers12；本项目保持每批8身份但每身份8张，明确这是复现条件差异。
增强采用已有共享flip/crop、各模态独立擦除；不声称作者原增强几何本来同步。
清洁输入用BILINEAR resize、[0.5]*3归一化，沿用作者test默认插值。
拼图先按RGB[0:256]、NIR[256:512]、TIR[512:768]切成三张256×128图，不使用raw目录替代压缩后的拼图。
原RandomIdentitySampler保留，完整保存每步实际记录indices；用实际完整batch数记账，不用估计len(loader)代替。

## 一次性T0、M0、完整基线

1. T0 tools/verify_rgbnt100_signal_protocol.py：在远端再次核对全部8675文件SHA，
   逐张对比新loader与作者read_image的全部26025模态切片像素；17350次拼图解码，
   每条真实query正例计数/完整gallery/source隔离验证。
   两项人工可解camera排序fixture要求同camera负例保留、所有合法正例用于AP；
   交叉核对作者eval_func。fixture明确是合成协议测试，不是真实数据检索结果。
   不做模型前向/优化/真实检索，无官方测试图像访问。
2. M0：通过同配置/runner绑定的完整T0后，三fold各自fresh初始化、真实B64/K8恰好8步，
   共24优化更新/1536源记录暴露。保存全部逐步目标、AMP、源indices、参数组与峰值显存；
   无缺失梯度的可训练张量、无非有限量/overflow、原六个selector参数状态不变、模型状态确有改变。
   每fold8条clean source在保存前后严格重载对比3072D特征，共48条源模型前向；heldout模型前向0。
   这是新数据集基线容量/调用链检查，不是新方法100步拟合门或科学收益证明。
   M0权重保存为独立工程证据，正式训练不用它。
3. 完整基线：同配置/runner的M0 PASS后，各fold重新seed42初始化，一次训练完整30epochs。
   不逐epoch做heldout，不存best候选，第30epoch是唯一最终checkpoint，严格重载后提取全部heldout gallery。
   query复用该次gallery提取的位置，三fold共8675条heldout模型前向；无重复query解码/前向。
   L2归一化后CPU squared Euclidean、NumPy原argsort、无rerank、无Oracle或测试时调参。
   全部8675条AP/首正例名次、mAP/Rank1/5/10、完整gallery排列均保存，
   full_rankings.json.gz仅压缩完整整数排列，不截取Top-k，不抽样。
   特征/距离/checkpoint保留服务器，本地只接收JSON、文本、压缩JSON及文件SHA。
4. 完整终态后核对全量原始数据、训练标量与作者eval_func，独立审计后才登记候选配对训练。
   低基线分数也完整报告；没有按内部mAP自动通过或失败的机制门，不扫描epoch/LR/seed以提高它。
   不启动官方评估、原三角色消融或已有科学失败版本重跑。

## 成本、停止及观察

T0全量CPU图片/协议检查预计1–3分钟，M0约2–5分钟；这是事前估计。
正式90 fold-epoch约7000–8000步量级，以原sampler实际输出为准，预计60–150分钟；
根据M0与首折实际epoch时间只更新观察时刻，不改变30epoch终点。
任何工程门实际失败则保留失败日志和可用逐步证据，先定位具体问题；不放宽门、不加fallback、不盲目重启。
新训练进程前核对RTX3090至少22000MiB可用。长任务按预计终态前几分钟观察，间隔180–300秒或更长，
不以观察超时当成训练终止，不按临时heldout指标提前停止。

当前代码只完成本地文本/AST检查；本地模型/张量/图片运行0。T0、M0、完整基线均NOT_RUN。

