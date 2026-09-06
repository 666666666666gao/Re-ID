# V27：冻结 Patch stem 上的来源环境统计混合

登记于 2026-09-07；状态 IMPLEMENTED_REGISTERED_NOT_RUN。M0/Q1尚未运行。
本轮只有一个新干预，完整三数据集/SOTA目标仍未达到。

## 证据与区别

V26的责任目标只有极少融合难关系；随后完整来源普查、支持分解表明，
纯残差银行在原增强来源上也接近饱和（mAP99.999877、R1=100）。
因此不把V26门控改成纯银行、不放大其权重，也不直接加全融合XBM。
这些发现并未证明来源环境变化是唯一原因。本轮将环境变化作为独立可证伪假设。

V24已有共同几何、弱erase0.5、强erase0.6及每模态独立亮度0.8–1.2，
还同时包含原型监督；本轮不是重跑这些增强或原型。
本轮读取实际Signal VisionTransformer：conv1输出B×768×16×8，
随后才加入CLS、camera-SIE、位置编码并进入ln_pre与12个Transformer块。
扰动只插在conv1输出；不在已高度语义化的block8输出上机械做IN。

思想来源：Zhou等的[MixStyle，ICLR2021](https://arxiv.org/html/2104.02008)、
[作者仓库](https://github.com/KaiyangZhou/mixstyle-release)。
作者建议低层CNN统计混合并报告ReID域泛化；这不证明冻结CLIP stem同样有效。
独立实现论文统计公式，没有复制作者代码。作者仓库标MIT。
跨摄像头供体规则、三模态共用计划、原Signal与扰动角色双路径是项目适配，
不是MixStyle原实验复现，亦不宣称已证明的新颖性。

## 唯一干预

两端均用原V8、原采样、原图像增强、原14项ID/Triplet和相同训练预算。
control与source_style均先运行完整原冻结Signal，保留3072D输出。
训练时两端另执行每模态一次冻结视觉编码，得到角色anchor和reference。
control返回未扰动stem；candidate按下面固定计划返回统计混合stem。
两端都计算混合算术，区别只在是否采用扰动。额外视觉执行次数相同。
新的reference也来自同一次扰动路径，避免把风格变化本身当作角色残差。
所有冻结层与三角色参数仍共享原对象；没有新增可训练参数。

对样本i、模态m的stem特征X，按每通道128个空间位置计算
mu=mean(X)，sigma=sqrt(var_population(X)+1e-6)；
mu、sigma停止梯度。供体j满足camera(j)!=camera(i)。
mu_mix=lambda_i*mu_i+(1-lambda_i)*mu_j，sigma_mix同理；
X_mix=(X-mu_i)/sigma_i*sigma_mix+mu_mix，FP32算术后转回原dtype。
只混合同模态供体统计，不混RGB与红外统计、不复制供体空间内容、不改身份标签。

每batch独立Bernoulli p=0.5；每行lambda~Beta(0.1,0.1)。
三个模态共用供体行索引和lambda，空间增强原本同步。
供体可重复、可同身份；它们不是新增独立身份或新图像。
每行在所有异摄像头行上均匀选择，通过独立U(0,1)随机分数argmin实现。
原采样保证每batch至少两个摄像头；没有异摄像头供体直接合同失败，不用替代规则。
计划使用独立NumPy PCG64 SeedSequence([42,fold,zero_based_step])，
不消费原图像增强、模型或优化器随机流。逐步保存完整供体和lambda。
p/alpha/epsilon/插入位置固定，不从本轮或已消费官方指标选择。

推理时只有原V8路径，所有增强关闭且不执行额外视觉pass。
M0须验证三折两端原始推理全输出一致，真实训练原Signal prefix也完全相等。
混合通道统计可能同时影响颜色/身份线索；是否保留足够身份信息由M0和完整Q1检验，
不把“统计属于风格”当作本项目已证实的分解。camera-SIE仍用query原camera标签。

## 初始化、采样和预算

六端复用合法V12 source Signal及角色初始化，不重训baseline。
每折94个source身份，14跨相机/80单相机，记录2126/2075/2051。
使用原CrossCameraIdentitySampler，B64/K8，workers4，seed42，
单视图SharedGeometryTripletTransform；原真实跨相机正对占比8.070790816%。
全部1680批次/端采样与已有V26固定原采样元数据一致，直接复用其control合同，
V26责任损失不继承，V25采样不继承。输入和供体计划在两端逐步配对。
20epochs，580/560/540 updates每端，六端3360updates/215040样本曝光/120epoch。
AdamW LR0.00035/wd0.0001，warmup5/cosine20，AMP初始scale256，
原七组ID/Triplet、margin0.3、label smoothing0.1及全部权重不变。
总参数98800141，可训练7841292、203tensors；新推理参数/计算0。
训练增加冻结视觉计算，必须报告实际时间和显存，不能称训练免费。

## T0与M0

T0远端CPU只用合成4×3×5×7数组：独立NumPy FP64验证统计公式值及
停止统计梯度后的导数，最大误差<2e-6，输入不修改、供体确为异摄像头。
无模型/真实图像/权重读取。

M0三折×两端×8个原source batch，48次原模型只读前向。
每批再只读运行backbone训练接口（原Signal+角色重编码），强制激活扰动计划；
candidate的anchor/reference均须改变且有限，control完全一致，
两端训练Signal prefix完全相等，所有冻结state保持，推理全五输出相等。
这些额外48次backbone前向是工程成本，不计为优化器更新。

然后fresh fold0 control8步、candidate8步容量；fresh candidate固定第一批100步。
固定过拟合使用同一供体/lambda且强制active，避免未激活增强时冒充新机制检查；
不选更容易的样本、供体、步数或中间最低loss。
共116updates。逐步日志立即刷新，保留真实已完成更新。
203tensors非零有限梯度、冻结state不变、AMP overflow0、reserved<24576MiB。
原ID下界0.75*H94(0.1)=0.57838292104621；
固定(L100-floor)/(L1-floor)<=0.1。
两端容量供体计划完全一致，candidate至少一批active，过拟合100步全部active。
M0若失败就封存并停止Q1，不调p/alpha/位置或更改门槛救回；
工程不通过不表述为held-out检索失败。

## 完整Q1与原科学门

同一持久原进程在M0通过后执行三fold×两端全部六个epoch20终点。
每端只保存一个final（不含重复冻结baseline），严格重载再评估。
完整gallery1000/1051/1075共3126，eligiblequery190/179/202共571，21身份。
同身份同camera过滤；无合法正例记录继续充当图库负例。
30-dev/官方测试访问0，重排序/测试时更新0，无中途检索选择终点。
原五项门全部保持：
1. aggregate fused mAP相对本轮control至少+1pp；
2. 三折fused增益均非负；
3. 三个完整角色aggregate mAP增益均非负；
4. 21身份bootstrap10000次seed42的95%下界>0；
5. candidate fused严格超过自身Signal与三完整角色。
六端结束才作Q1判定；负fold不中断。
全3360训练行、120epoch、30路检索数组/完整排序均保留并完整复算；
报告所有21身份/571query AP与R1修复/新增，不挑样本或补写未读图像成因。

单seed42、重复使用OOF开发身份仍是限制，bootstrap不消除选择偏差；
已有非bitwise训练证据保留，匹配控制不能被称完全确定性实验。
不启用Router/HFER、原型、V23MLP、V26责任、XBM、SmoothAP、PCGrad，
不做旧失败版本救回、提前消融/多seed或官方调参。
通过后的D1/refit仍需另行登记。

## 环境、磁盘、溯源

复用远端tri_reid/PyTorch2.5.1cu121/RTX3090，依赖环境无修改。
本地仅文本/AST/JSON，不运行本地Torch，不下载图像、模型或特征文件。
2026-09-07 03:00数据卷free20007673856B（18.63GiB），系统卷11129257984B（10.36GiB）。
已清24个冗余resume共24.900344GiB，12个保留模型本轮完整SHA复查通过。
新实验只保存六个final和全量检索数组，预计新增<4GiB；
不存逐epoch权重或重复恢复权重。磁盘卷分别统计。
M0预计4–10分钟，完整Q1约1.5小时，按实际速度修正；
实验轮询180–300秒或预估里程碑，不以观察超时为重启理由。
外部独立审计额度不可用；执行器验证不称独立审计。
科学配置、源码SHA、完整计划在运行前提交并三方同步。
