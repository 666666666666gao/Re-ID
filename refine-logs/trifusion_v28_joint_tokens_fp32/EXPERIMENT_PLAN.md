# V28 R2：仅修复新联合模块的FP32执行

登记日期2026-09-07，IMPLEMENTED_REGISTERED_NOT_RUN。
原V28 c9a38e6 M0_FAIL（218/219，dt_proj.weight恒零）保持封存。
三项诊断总3次重建更新及两次旧跨进程回放失败完整保留。
同前向配对证明AMP目标梯度0/2048，FP32 fast/unfused均2048/2048；
这是当前代码中的真实数值问题，不是为了假想边界添加逻辑。

## 唯一工程差异

模型架构、token顺序、hidden128、双向共享Mamba、零输出投影、九槽位归一化、
原独立角色、七组ID/Triplet、所有权重与epoch/seed设置，都按原
[完整V28计划](../trifusion_v28_joint_tokens/EXPERIMENT_PLAN.md)保持。
该原计划SHA853fb5b532f16623c246472c3c50cab52c5227954093b29d8752d52757bd730d未修改。

仅新增joint_tokens_v28_fp32.FP32JointResidualTokens：
在这个413056参数的joint模块forward内关闭autocast，将输入转FP32后
调用同一个原JointResidualTokens.forward。两次Mamba仍走原fast kernel。
原Signal和C/T/M角色执行精度不变，不将整个网络切成FP32。
不改loss scale、学习率、模块宽度、初始化值或融合权重。
新的类严格加载原joint初始化state，不把诊断更新后的参数带入训练。
原V28源码/runner/config保持，R2使用另一个明确文件与执行commit。

仍为control98800141总/7841292训练/203张量；
candidate99213197总/8254348训练/219张量，新增413056。
精度改变会增加候选时间和显存；实际成本需如实报告，无低成本推理保证。

## T0与真实fixture回归

先保留原合成代数/梯度检查，再运行check_v28_fp32_fixture.py：
只读取已捕获的输入/joint state/真实upstream fixture，
旧AMP实现必须重现0/2048，新FP32类必须恢复2048/2048且全部梯度有限。
没有图像读取、optimizer更新或检索评价。
该fixture只用于一次性工程测试，模块随后丢弃；
不向任何fold模型传递参数、梯度或特征，实际训练仍从合法各fold V12独立构造。

## 完整M0与Q1合同

M0 fresh三折×两端×8批预检，另48次原V8对照及48次backbone训练接口；
初始配对输出逐bit相等，原独立角色/logit不变，旧V8末端归一化差<1e-5。
fresh fold0 control8步/candidate8步容量，再fresh candidate固定第一批100步。
全部梯度存在有限、按阶段219/203张量非零覆盖、overflow0、冻结state不变；
容量reserved<24576MiB，原ID下界0.57838292104621，
(L100-floor)/(L1-floor)<=0.1。不能放行仍零的dt权重或修改门槛。
这是R2独立116步，不替代原R1的116步及诊断的3步成本。

M0通过后同一持久进程执行完整三折两端各20epoch，3360更新/120epoch。
两端均从合法V12初始化，B64/K8/workers4/seed42、AdamW3.5e-4、wd1e-4、
warmup5/cosine20、原七头/14loss/margin0.3/label smoothing0.1。
两端都采用固定V27 stem统计混合p0.5/alpha0.1，共享三模态供体和lambda，
原采样与1680批次/端元数据逐条匹配。无V25采样或V26责任目标。

全部六个固定epoch20终点保存一次、不含重复冻结baseline；严格重载再评价。
原三折3126完整图库、571query/21身份、五输出与camera过滤不变；
无合法正例query从分母排除但仍保留在图库，禁止dev/official/reranking/TTA。
无中途检索选择或按第一折改变后续运行。
五科学门保持：fused配对增益>=1mAP；三个fold增益均>=0；
三个原独立角色aggregate增益均>=0；身份bootstrap10k/seed42下界>0；
fused严格高于Signal和三个原独立分支。缺一即Q1_FAIL。

旧V28完整终态核验脚本绑定R1，不能直接用于R2或伪造R1检索结果。
R2 M0/终态核验将明确绑定R2实际commit/config/plan与precision_fixture记录。
新precision诊断不能代替这些检查。仅固定epoch终点评价，不读取新官方指标调设置。

## 预算与边界

预计M06–10分钟，Q1约80–120分钟；使用实际容量速度更新估计，
实际训练轮询180–300秒。数据盘目前约14.5GiB，计划新六final和数组低于1GiB，
诊断fixture约116MB保留作回归证据；不删除受保护旧终点或初始化。

候选新增计算/精度成本与控制不匹配，后续成功后才登记容量/时间等预算消融。
标准FP32执行修复不作为算法创新；最终结构效果仍需完整检索证明。
单seed42和反复使用OOF身份限制继续保留，外部独立审稿仍不可用。
整个多数据集/SOTA目标尚未达到。
