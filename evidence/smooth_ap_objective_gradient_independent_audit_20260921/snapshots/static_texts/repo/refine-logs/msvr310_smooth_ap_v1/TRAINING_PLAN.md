# MSVR310 Smooth-AP v1 完整配对训练合同

2026-09-08登记。原role-set完整Q1与来源正例补充审计均关闭；科学FAIL不改变。本合同检验新的单一目标，不重新运行旧科学实验，也不声称标准Smooth-AP为原创。

## 唯一干预及借鉴

两端为control、smooth_ap，均从原source-only Signal及seed42角色初始化独立建立原V8三角色。保留fresh历史坐标、完整历史候选参数导数、固定拼接，64当前anchor，历史不作额外anchor。其余13项损失及权重不变。

control用原batch+history最远正例/最近负例0.3未平方欧氏hinge，保留原current/history极值tie导数约定。smooth_ap用Brown等人ECCV2020 Smooth-AP公式，在实际单位fused表示上以s=1-d²/2计算余弦等价分数，对全部真实正例位置计算平滑AP并取1-mean(AP)。温度固定0.01，fused度量项权重沿用1；不扫描或来源自适应缩放。原文 https://arxiv.org/html/2007.12163v2 §3–4与§5.3，独立实现公式，不复制作者代码。

正例是全部相同真实身份位置，排除anchor自身位置；比较某正例的排名时也排除该正例与自身的比较。保留同记录不同当前视图，历史同记录唯一化并排除当前记录副本。负例是所有其他身份，包括同scene。scene只作分层记录，不新增训练scene过滤、身份配额或重采样。历史叶子偏导与当前主目标使用同一定义，历史group按原64记录和RNG重放，经VJP加入189个encoder参数，一次optimizer更新。

此干预同时改变关系覆盖、软权重和目标尺度，不能单独归因于正例数；来源存在非最远正例反序不保证新目标能泛化。两端均计算hard/AP及四空间距离用于记录和核验，不使用role-set提议，不新增Router、联合头、统计扰动或AP推理重排序。

日志为复用原加权入口保留components.triplet_fused键，但每步active_fused_metric及summary.fused_metric_storage_key明确实际语义。原7组监督中，候选fused度量项变为AP，其余6项仍Triplet；不能把候选AP称为降低了同定义Triplet。

## 固定数据、优化与终点

递归绑定原role-set/history-gradient配置的source-only初始化、协议、采样和优化。仅seed42，B64/K8，每端20epoch/260更新，三折两端1560更新。原学习率/AdamW/衰减/增强均不改；前65步两端原批内目标，step66启用新目标并入队，首次历史候选由实际日志确定（旧合同为step67）。容量512、最大年龄8；当前重复视图允许，计数为训练曝光而非独立样本。

三折source路径及身份完全隔离于heldout；完整gallery保留无合法正例身份的干扰记录。MSVR310使用same-ID AND same-scene评价过滤，600合法query/60身份、三折gallery共1032记录。每端全Signal/fused/CNN/Transformer/Mamba结果、全部query AP/排名、全部身份/折统计均保存；不读取官方test调参。两端完整采样和增强像素SHA须一致；不承诺独立CUDA数值轨迹位级相同。

## 工程及科学顺序

持久wrapper固定T0→M0→M0_CPU→Q1→Q1_CPU，阶段非零退出即封存，不自动修改或重启。M0和CPU通过后才能进入heldout；M0完成后独立审计可与固定Q1运行并行，不能据局部fold改变目标。Q1结束另做独立完整审计。

T0：新合同递归输入哈希、当前注册核心/补充审计绑定、完整780来源batch队列/mask/年龄重放，原memory/历史链式法则合成检查，以及新Smooth-AP标量/有限差分/自位置/tie/重排检查。control标量与距离导数须和原hard实现一致。0图像/模型/更新。

M0：三折两端各8更新容量检查（2步预热），再fold0两端各100更新固定batch过拟合，共248更新，无heldout。保留203/203累计非零覆盖、AMP有限、冻结Signal/参数hash、固定步数、checkpoint严格重载全部输出逐位一致；过拟合保留原label-smoothing损失下界校正后末/首比≤0.1。过拟合固定当前记录会排除其历史副本，不能用此声称历史反传过拟合成立。

各容量端首个实际单历史group，四个重编码输出逐位一致；当前总目标导数+历史VJP对比完整图总目标encoder参数导数，相对L2≤0.005。真实新历史目标须在每角色上有非零参数贡献，报告实际见证而非只看距离导数；不宣称每步203张量全部非零或全程参数梯度独立重建。64anchor之外无额外loss anchor。

CPU逐步重放全部保存四距离矩阵和真实metadata；独立Float64 NumPy rank sums核对每anchor smoothed AP、正例计数、hard/AP标量和14项加权账本。标量/逐anchor AP允许原登记FP32对Float64绝对误差2e-6；余项沿用原有限值/队列/checkpoint/完整检索/资格核验。不会用CPU复算冒充图像前向或全程参数梯度重演。

两组原五项全部保留：配对fused≥+1pp、各fold≥0、三个角色增益≥0、身份bootstrap下界>0、候选fused严格超过Signal及各角色；候选对同协议Signal组同样要求fused≥+1pp、每fold≥0、三个角色不低于Signal、身份下界>0、fused严格最高。两组全通过才能晋级。seed42/10000身份重采样不是多训练种子。失败保持，不调温度/倍率/终点救回，不以工程门/内部结果替代官方或三数据集Goal。

## 资源与保留

复用tri_reid实际Python3.10.14/PyTorch2.5.1+cu121/RTX3090环境，无安装或环境重建。23:34实查GPU1MiB/24576MiB，输出卷9954095104B空闲，主卷2594549760B。输出/root/trifusion-storage/artifacts/msvr310_smooth_ap_v1_seed42_<commit>，启动重新核验GPU<500MiB、空闲≥4GiB，最大预期新增3GiB。每端只保留必要M0/固定epoch20精简角色checkpoint，不按epoch累积大权重。

维持四空间距离记录的最坏预算约1.07GB，加必要12checkpoint/检索数组/日志，3GiB覆盖；无新增推理参数，但记录实际历史重编码、VJP重算、峰值显存与总/fit时间。Smooth-AP使更多历史group上游非零的可能性不等于免费训练。预计M0约10–25分钟、完整Q1约3–6小时，以实际耗时修正；screen真正脱离连接、保留原PID和退出码，按180–300秒或预计结束里程碑观察，不因观察超时重启。
