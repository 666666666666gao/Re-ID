# V5：原作者完整批次、单进程两卡模型分段

目的：解除RGBNT100作者B128的真实单卡OOM，执行已登记的global_only/semantic/native三数据集九端。不是新增科学模块，不改配方、precision、seed、batch、数值门或评价协议。V1 OOM、V2/V3 checkpoint梯度失败、V4 CPU-save及§780一次边界诊断失败继续封存；不再尝试CPU-save或新增诊断臂。

## 固定计算与放置

每个worker只看两张2026物理卡，(0,1)或(2,3)；最多两worker/四物理卡，先核对两张卡都空闲，不抢占。一个进程内的cuda:1放视觉前6 block、conv/CLS/position/camera/ln_pre、第4层adapter；cuda:0放后6 block、第8/12层adapter、ln_post/CLIP投影、全部角色/native读取器/readout及完整作者BN/分类/Triplet。只一份参数，不复制主干或跨网络累积梯度。参数移动在optimizer创建前完成、对象身份保持，FP32存储、名字/状态/原分组保持。

两处持久pre-hook只搬原图到前端、第7个block的输入到后端。原作者ViT block仍直接执行；第4层snapshot先按原公式stack，再用保留autograd的GPU间to搬到后端，其余snapshot仍按原公式。最终depth/role/modal stack、RGB/NIR/TIR顺序、三角色均值写回、CNN→Transformer→Mamba、原生独立QKV/159296参数/14张量/唯一零出口不改。没有checkpoint、CPU saved-tensor钩子、compression、新stream或fallback。train/eval同一放置，全部BN与Triplet一次处理完整B，不用DDP或梯度累积近似原batch语义。

## 逐卡预算：来源明确但仍须M0实测

依据§780实际1323份pack元数据，三个模态各441项；每模态前6 block结束在本地序号205，后段从206开始（block7的LN）。前段单模态保存逻辑值1194906976B，后段1255505760B；不扣共享storage/view/AMP权重缓存，以全部逻辑值乘4估计B32→B128，前端14338883712B、后端15066069120B。其形状来自RGBNT201 B32实测，车辆长宽交换而token总数相同；这是保存值的保守逻辑计数，不能单独当作完整峰值上界，且不证明新放置后的kernel实现/暂存相同。

前端参数为6×7087872+691200+3×100672=43520448，加真实camera_num×768；后端不含角色/head部分为43526016，实际模型构造时按named_parameters生成每卡真实参数数目/字节及FP32 gradient+Adam两状态12字节/参数账。每卡另预留约1GiB用于参数/梯度/Adam/AMP cast；后端额外估计2GiB原captured/内stack/最终stack暂存（B128最终stages684785664B）、4GiB角色/原生分支保存值、1GiB workspace。前端额外1GiB workspace与约128MiB跨段snapshot/输入暂存。估计前端约15.5GiB、后端约22.0GiB，后端余量有限；4GiB角色与workspace是源码尺度估算而非已测上界，不据此宣称容量通过。两张24GiB卡也不合成一块显存。

实际M0记录两卡max_memory_allocated/max_memory_reserved，含optimizer状态和严格fresh重载；任何OOM/skip/门失败均封存，不减B128、改precision或挑seed救分。CPU约115GiB可用，仅用于已有数据、CPU状态保存及报告，不 offload saved tensor/optimizer。新权重目录固定/home/gaob/trifusion-native-evidence-v5，原固定整批18×384MiB+600MiB+256MiB预算加2GiB磁盘保留；/data和/home运行时均保留2GiB。没有旧权重搬运或symlink。

## 必须先完成的工程阶段

1. 源码fresh复核，包括所有参数唯一所有权、capture与跨卡autograd、完整BN/Triplet、两设备GradScaler/Adam、state保存/重载、队列同时归属两卡。通过后才部署。
2. 每数据集global_only/semantic/native真实初始化，共9份；公开CLIP、新camera/head/适配器，作者配方seed42。不加载ReID/Signal训练权重，不用SIM/AlignM/教师/新loss/外部资源。作者batch201 B64/K8、100 B128/K16、MSVR B64/K4。
3. 每数据集完整作者batch的初始semantic/native配对eval，全部原state、raw/L21536/global/作者logits/BN匹配。零出口不等于训练保底。
4. 各数据集/variant的原单卡与新两卡production AMP FP16、scaler256、真实作者batch首32前向/反向比较，零optimizer更新；使用同batch/state/CPU及两CUDA RNG。原forward1e-5、paramgrad1e-4 allclose不变，记录全部pre-assert梯度/输入；原两CUDA RNG/完整state/buffers/BN及临时capture hook清理精确门不变，持久两个transfer pre-hooks明确保留。失败即停，不自动更改方案。这个witness不代替完整B128容量M0。
5. 全部9端完整作者batch八次真实optimizer更新M0，原全参数活动、BN8、14项native参数累计非零有限梯度与参数实际改变、完整state严格重载1e-5继续要求。所有9M0通过以后才允许任何50轮正式训练。没有按官方分数修改合同。

## 正式阶段与报告

九端全部完整50轮，单一fused mAP-best覆盖保存best_map.pth，全部CMC来自它；严格全状态fresh重载和作者完整合法query/gallery及camera/MSVR时间过滤。主要native−semantic≥0.5 mAP且R1不下降项目推进线，同时semantic/global及native/global增量、首位修复/新增错误、身份宏AP分布、完整曲线、真实训练+评价时间、两卡峰值内存与参数数目记录。只一次CPU终态报告；任何端失败不拼出成功矩阵。native额外159296参数，不能直接证明信息来源的容量独占因果、三模块科学有效、完整流程多seed或SOTA。

工程probe待其严格验收和全部报告依赖闭合后退役；正式端只留一个mAP-best，公开/作者权重和活动依赖保留。观察首次按估计180–300秒，队列240秒；不能因观察超时重启。2025仅同步文本，不执行其conda/CUDA/模型。
