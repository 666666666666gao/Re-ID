# 冻结文本包必要性对照：两轮方法评审后的完整候选方案

## Problem Anchor

- Bottom-line problem：在RGBNT201、RGBNT100、MSVR310三个数据集形成超过同协议强baseline、明确预训练资源且可经当前文献核验的SOTA方法；完整流程稳定性和机制必要性仍须证明，不能将本次小实验替代整个目标。
- Must-solve bottleneck：当前角色已经参与训练且有独立判别能力，新增修正仍未稳定改善global的未见身份排序；更大的读取容量、纯视觉重建、独立fused头、MD/repair和当前同模态几何辅助均已有闭合结果。下一项必须检验不同的身份证据学习依据，而非重命名上述控制或继续搜索gain/温度/margin/seed。
- Non-goals：不把训练配方、初始化/显存/数值修复计作创新；不宣称固定文本变换创造了输入图像没有的新信息或已经识别真实部件；不使用测试身份标签、在线gallery适配或人工Oracle。
- Constraints：只2026物理GPU0/1，现有conda、三数据集及公开CLIP；不查25/GPU2/3/温度功率，不安装。seed42、每个合格端fresh50、单一fused mAP-best与同权重CMC、完整合法query/gallery及camera/scene过滤；官方集已消费，属于开发比较。闭合旧控制不重训、不调原门槛。
- Success condition：机制对照必须同时优于匹配原角色控制与独立global，并分清新增容量与预训练先验；任何小试成功都不自动证明SOTA、多种子或三个创新点。最终完整目标保持ACTIVE_UNMET。

## 本轮修订与没有改变的科学问题

本轮只落实R1的C1—C4，不增添模块、loss或正式实验条件。R1为REVISE、5.90，pilot为REVISE_BEFORE_NN；不是NN授权检查通过，更不是论文READY。原始请求、原始proposal、评审原文及实际失败文件保留。方法评审行政上限为两轮，本轮后使用同一评审进行R2；达到轮数不等于科学或执行通过，也不为凑9分添加机制。

H2a已完成三端150轮/6,484更新，主ΔmAP为+0.0116/+0.0046/−0.5631、0/3推进；同模态normalized K-K辅助没有证明raw Q-K身份读取产生额外价值。近容量、视觉重建、独立fused头和MD/repair也已有闭合结果。当前试验检验一个此前角色主线未使用的资源：冻结CLIP文本预训练包能否改善区域query上下文，相比同结构冻结随机包产生净收益。

T(g)是g的确定性重编码，不增加图像中原来没有的像素或独立身份信息。它可能用预训练先验改变从现有patch读取的内容，也可能只是重复global或改变优化条件。本轮不承诺互补、不承诺论文新颖性、更不将六端pilot替代三数据集强baseline/SOTA/完整流程稳定性的完整目标。

## 最小方法：只增加一个上下文适配器

复用raw职责分离semantic模型。原视觉/global/head/三角色/readout/gain、作者配方和1536维部署接口不改。每图g:B×1536，拆成三模态g_m:B×512。新增共享ψ为Linear512→128、GELU、Linear128→2048，生成4×512 pseudo-word增量；两个Linear含bias，总329,856可训练参数。ψ输入为sg(normalize(g_m))，部署不输入身份ID。新增W为无bias Linear512→512，总262,144参数；唯一新增可训练集合ψ+W共592,000。

模板固定为“A photo of a X X X X person.”或vehicle，只由任务类型选名词；不学习style prompt、不使用ID索引bank。实际BPE检查两者均为12token，pseudo-word位置5、6、7、8，EOT位置11。其余token固定，四个X固定embedding加ψ输出。每模态得到冻结T输出t_m:B×512，t=normalize(mean_m(normalize(t_m)))，augmented_context=normalize(original_context.detach()+W(t))。文本不复制为value、不过度扩维，所有区域value仍来自原128视觉patch，后续CNN→Transformer→Mamba及读出不改。

### C1：明确位于视觉detach之后的可学习入口

现有DetachedSemanticTriFusion.role_evidence会将incoming context.detach()。因此禁止先在forward_features构造augmented_context，再调用继承的detach实现；那会永久切断ψ/W。

未来最小实现为一个新semantic子类覆盖role_evidence：先stages=stages.detach()、g=shared_global.detach()、context=context.detach()，然后在此函数内计算ψ→冻结T→W→augmented_context，最后直接调用self.roles(stages, augmented_context, g)。不得再调用会detach context的super().role_evidence。仅视觉数据停止梯度，新增上下文到ψ/W的autograd保留；冻结T意味着其参数requires_grad=False，不意味着T的forward进入no_grad。全局loss、角色loss以及global/作者head梯度归属继续沿用已封存raw合同。

这是已读原源码约束未来接入的真实风险，尚无新代码运行失败，不追认成旧实验原因。原role类无需复制或重写，原forward_features无需另做编码，保持修改最小。

### C2：使用显式普通CLIP文本路径

现有Signal Transformer.forward对nn.Sequential传多位置参数；ResidualAttentionBlock.forward默认有视觉prompt/adapter分支。不能把这个包装器当普通encode_text使用，也不添加fallback或修改作者源。

构造独立冻结文本组件，精确复用已核验作者ResidualAttentionBlock(d_model=512,n_head=8,pattern=None)的普通参数结构和forward_ori；12层逐层显式调用forward_ori，执行标准pre-LN MHA残差及QuickGELU MLP残差。因果mask为上三角−inf，主对角及下三角0；按prefix长度给每层设置正确mask。序列以L×N×512进入MHA，经过12层后ln_final，取EOT11，再乘512×512 text_projection。两臂完全相同FP32运算路径，T保持eval且无参数更新。

公开archive SHA5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f。pretrained臂从精确text tensor名严格映射，不加载visual/logit/metadata、不允许遗漏/宽松load。组件持久保存12层全部tensor、77×512 position、ln_final、projection和12×512固定template embedding，保持严格重载；不持久注册49408词表，不做运行期checkpoint依赖注入。encoder本体形状清单为148state tensors、38,131,200元素；另加固定12×512 template buffer（1tensor/6,144元素）。持久组件合计149state tensors、38,137,344元素/152,549,376FP32字节。这是静态估计，source构造后要实际核对，不能直接当作真实checkpoint或显存证据。

### 冻结随机包与完全一致的新增可训练初始化

random臂不是声称“复现官方所有初始化细节”，而是预登记的CLIP尺度固定随机包。用私有CPU RNG seed42，固定以下抽样顺序：临时49408×512 word table N(0,0.02)后提取同一template12位置；77×512 positional N(0,0.01)；按block0..11依次生成MHA in_proj、out_proj、MLP c_fc、c_proj，std分别为512^−0.5、512^−0.5×24^−0.5、1024^−0.5、512^−0.5×24^−0.5；各bias置0、各LN weight1/bias0；最后text_projection N(0,512^−0.5)。临时词表随后释放，不作为持久组件。所有文本参数及template固定，不进入optimizer。

ψ/W使用另一个私有CPU RNG seed42和PyTorch Linear默认初始化，顺序为ψ第一Linear、ψ第二Linear、W；新组件构造/载入时fork_rng不改变原模型及训练的全局RNG。pretrained和random臂必须共享这份ψ/W初始state，而非依靠“都设seed”推定一致。全部共同visual/camera/head/role/readout/gain初值逐tensor核对，优化器覆盖新增集合恰好一次，冻结T集合零次；batch/增强随机流与封存原训练合同一致。固定随机包结果只对这个包/seed有效，不估计随机初始化总体，也不单独隔离可读词义或某个text层。

### C3：预登记首步开通及之后的任务梯度

原context_queries记C，初始C=0。exact init时ψ/W的目标梯度应为0，因为读取对augmented_context的导数包含C；C本身可有非零目标梯度。不能把这已知首步零梯度误判为永久断开，也不能靠weight decay导致ψ/W变化谎称角色loss监督有效。

保持C原零初始化、ψ/W正常非零，不新增零W出口。真实8步M0记录：第1次反向C有限非零目标梯度及第1次实际C更新；第2—8次反向每个新增ψ/W tensor至少一次获得有限非零目标梯度，且累计实际参数变化非零。这里task_gradient直接取该次角色任务反向在optimizer.step前的grad，Adam/weight_decay尚未加入，不能以总参数变化代替。首步已知0单独记EXPECTED_INITIAL_CLOSED，不删原记录；如果在固定8步内没有建立全部新增参数的任务路径，则M0失败且该端没有正式成绩，不延长探针或事后改LR/门槛。

新增ψ/W及C的grad/state change与冻结T全state不变、global梯度隔离同时记录。MSVR普通参数仍base5e−6及原分组，不因M0或检索弱继续搜索学习率。活动只证明可学习接入，单次或累计非零不证明学到了足够判别性；完整50轮仍记录实际更新、输出幅度和最终检索。

## 前缀资格、真实资源与执行边界

12token只删EOT之后的不可见padding，因果数学论证不等于数值等价。事前固定工程对照：两种冻结包、person/vehicle模板、相同固定非零pseudo-word输入及相同固定上游向量，12 vs full77的EOT输出allclose(atol=1e−5,rtol=1e−5)，对pseudo-word输入VJP allclose(atol=1e−4,rtol=1e−4)。仅比较四个有效pseudo-word输入梯度；两个路径共享完全相同参数、前12position和模板，full77按作者padding构造，不执行训练更新。测试实际失败原样保留，不通过修改阈值、挑选重试或自动fallback取得通过。输出/VJP目前均未运行。

所有真实NN资格仍须详细实验合同和source复核后执行，方法review不代替它们。检查GPU0/1峰值显存、完整B128/K16执行、optimizer一次覆盖、冻结T无梯度/无更新、raw/fused/logits及BN初始行为、8步活动、严格state reload/单份best全部指标。原visual分段first6GPU1/last6+headGPU0，text在GPU0，不探25/GPU2/3/温度功率、不安装软件。

持久encoder本体FP32估计152,524,800B，另有template24,576B，组件合计152,549,376B，新trainable2,368,000B。每新best估计约511MB，六best+一活动M0 probe+2GiB余量约5.7GB，但实际空闲、峰值和文件大小未资格。先核对依赖、实存SHA和现有OWN无用缓存，只在已闭合且无消费者后退役；保留三H2a best、六sealed控制、强Signal/V8/V27赢家及作者/公开初始化。无证据不盲删、无偷偷省略冻结state的重载兼容。若资源不能达到固定合同，记录工程缺失，不转写为算法无效。

## 训练、部署和机制判定

训练仍L_g(g)+L_r(sg(g)+gain*c)，角色使用停止head参数梯度的当前作者头值并克隆BN buffer。不加text contrastive、local ID、独立新头、teacher、matcher、meta、margin或新排序loss。部署输出1536维Normalize(g+gain*c)，一次单样本编码可离线建库；T及模板推理也保留，因此额外时间/显存/冻结参数均如实报告，不称训练期免费蒸馏。

新增pretrained/random两臂×3数据集共六端fresh50/300epochs，复用封存三个raw semantic和三个独立global，不重跑闭合队列。主要机制配对pretrained−random固定+0.5mAP且R1不下降；同一单份mAP-best报告201R1/R5/R10及各数据集所有合法query/gallery。还要各自对匹配semantic、独立global报告绝对净增量，不能用prior差值遮盖完整模型不如基线。该项目线不是统计显著性，单seed官方已消费开发pilot不等于新SOTA或稳定性。

既有近容量/visual reconstruction不是本text必要性结果。正结果最多支持此固定包/接口/recipe/seed的条件优势，不能推出语言语义独占作用、新图像信息、真实部件或论文新颖性；负且活动合格结果也不否定所有文本方案。若只有局部独立mAP提高而fused失败，则本目标仍不满足，不调gain/temperature救分。

事前诊断保持t模态/身份关系、ψ/W真实更新、Q/K范数及logit_std/熵、g/c/f分解、全query首位修复/新增错误和全部AP差分。观察不用于筛query/seed/epoch或后改方案。只有合格完整主配对出现净收益后，才另行规划完整流程稳定性、近邻及入口必要性；P2/P3不自动启动。两GPU历史H2a三端3h47仅成本参考；六端≥7.6h再加T成本，16—30GPUh是规划区间，真实M0及完整epoch后再估计里程碑，长队列按180—300秒或预计结束前观察，不因低利用率重启。

## 最接近的工作与本pilot地位

PromptSG（CVPR2024）已经有image-conditioned pseudo-word、文本引导空间交互及推理提示使用，与本入口非常接近。它必须列为直接近邻，而非只比较ID-indexed CLIP-ReID。当前PromptSG公开repo为空，不能宣称完整实现复现；当前阅读范围是publisher主页面与搜索索引论文摘录，直接PDF曾403，未假称读完全篇。
https://openaccess.thecvf.com/content/CVPR2024/html/Yang_A_Pedestrian_is_Worth_One_Prompt_Towards_Language_Guidance_Person_CVPR_2024_paper.html
https://github.com/YzXian16/PromptSG

CLIP-ReID原实现按训练ID索引四context，stage1学文本prompt、stage2文本原型分类；本ψ是实例g输入，不是其复现。DEEP用冻结文本空间、实例inversion及光谱style prompts做fusion/semantic embedding，推理也调用文本，60epochs；其公开tree10文件/0Python，不能取作可运行移植代码。本pilot没有复现其三个模块，区别并不足以建立原创性。
https://github.com/Syliz517/CLIP-ReID
https://aihuazheng.github.io/publications/pdf/2025/2025-DEEP_Decoupled_Semantic_Prompt_Learning_Guiding_and_Embedding_for_Multi-Spectral_Object_Re-Identification.pdf
https://github.com/lsh-ahu/DEEP-ReID

贡献质量仍有真实缺口：这是对当前三光谱角色路线冻结文本包必要性的诊断，不是新架构发明。本轮不靠再堆模块解决这一缺口。完整目标保持ACTIVE_UNMET；目前source实现、新组件构造、NN forward、M0及formal全部0。下一步只请求同一R1 reviewer复核修订是否解决C1—C4并判断有限pilot科学可取性，论文verdict与pilot准入要分开。

## 实际R2收束及普通执行义务

同一方法reviewer实际R2为REVISE、6.40/10，pilot为WORTHWHILE_PENDING_ORDINARY_GATES；C1—C4在设计层面解决，不需要第三轮方法改写。source/详细合同/资源/prefix/M0均未通过。requested Astra/max、actual model/effort UNATTESTED，同家族/provisional，CALIBRATION:none；分数不是科学效用或执行授权。

实际构造必须使用d_model/n_head名称。block默认初始化会消耗随机draw，所以先在保护外部RNG的构造域内建立模块，再使用独立固定CPU generator按已登记顺序填充random包，不能让constructor抢先消耗该generator。ψ/W另独立fork和相同state。以上是普通实现义务，不添加方法模块或改变控制问题。

148/38,131,200是原encoder本体；新固定template buffer另计的静态算术勘误已单独记录，未改写R1/R2或原估计JSON，尚无组件实际构造。两个方法轮已封存；论文新颖性、全流程稳定性及SOTA仍缺证据，完整目标ACTIVE_UNMET。
