# 匹配语义参考下的跨光谱证据变化：组件及有限验证设计

2026-10-13，§942。状态COMPONENT_PASS_NOT_INTEGRATED。原§938一端50轮及首次strict已经结束并封存；本次没有新完整模型、初始化、真实M0或训练。

## 问题及项目化依据

§941仅改变角色CE入口的L2归一化没有净增益，不能再用尺度修复解释全部瓶颈。§929–935又说明分支活动、公共能量或实际读出差异都不等于有效互补。旧Signal+V8在201较强，但与近期系统的初始权重、预训练尾层接续、训练阶段及输出维度不同，不能跨行作单因素解释：完整Signal前缀3072维，V8残差bank4608维，整体7680维；无SIM纯V8则6144维。

借LMF的模态独立因子乘积运算，结合本项目V8的匹配语义参考原则。并不是首次乘法：V17早已对角色peer做Hadamard乘积。新的有限问题是：在同一实际区域读取后，显式保留跨光谱内容的联合变化，是否比同参数加性变化或普通绝对联合内容更有净检索用途？参考条件可微，不是冻结教师。

## 实际组件

Processed是原CNN/T/M及桥接产生的三套区域；Reference从同一CLIP多层语义、同一角色query/Q/K/V和输出LN读取，但绕过CNN/T/M与桥接。两边都先在各角色、各模态内池化16槽位，不将同编号槽位当跨光谱真实部件对应。Shared stages/context显式detach，参考路径仍对角色query/projection与其归一化参数可微。

每角色三模态用独立无bias因子，固定rank4、width128；因子投影后做无额外参数LayerNorm，再沿模态相乘或相加、沿rank加权。正因存在LayerNorm，不称严格三线性多项式。没有追加常数1，但上游anchor、adapters等仍可携带常量或重复信息，不能声称严格纯交互或信息隔离。

三条件：add_delta=Ψ_add(processed)−Ψ_add(reference)；joint_delta=Ψ_joint(processed)−Ψ_joint(reference)；joint_absolute=Ψ_joint(processed)。Delta后相同LN，最后用现有1536×384 readout作dense投影到1536。全部都改变了旧structured readout与槽位池化方式；相对原raw semantic的变化不能独归乘积。额外参数589836，三条件参数与初始化相同且主动参与forward。初版用原raw职责目标及单1536维静态检索，不追加分类头、margin、teacher、memory、meta优化、文本或mask资源。

该差分只保证processed==reference时descriptor增量为0；真实初始化的三个算子并非identity，不能保证完整初始行为等于global或旧semantic，更不保证训练后不掉分。归一化乘积可能被共同成分支配或某模态抑制，必须实测。

## 已执行检查与未闭合接口

首次实际CPU组件f85a57 EXIT0：8次synthetic前向、1次反向、0optimizer更新。两差分在相同输入时输出0；翻转未改变其operator增量的另一光谱，乘积变化符合内容条件性，加性仅有1.42e−7舍入差；两个因子参数张量及processed/reference输入都有有限非零梯度，toy global detach无梯度。三条件完全同参数状态。

这没有调用semantic_reference helper、真实CLIP/Mamba、真实图像、完整wrapper、优化器或检索。完整入口尚未实现；真实initializer、作者head/BN状态、梯度所有权、optimizer恰好覆盖一次、8步M0和严格重载仍未验收。不能凭组件通过上正式训练。

同一研究agent源码复核未发现组件阻塞，但不是独立张量审计或完整新颖性审查。已有LMF/V8/V17先例明确保留。三条件仅支持两个条件效应；若将来主张reference与乘积的协同交互，需第四项add_absolute，不用三条件推导2×2交互。

## 后续有限执行顺序

先实现完整builder/driver，一次真实prepare与8步M0核验参考读取内容、shared任务隔离、全部新参数活动、BN状态及optimizer所有权。M0失败先保留实际失败定位，不盲目重跑或改变rank/gain救分。通过后RGBNT201三条件各fresh50/seed42，沿原作者训练配方及相同初始化/RNG与batch order。预计正式约2–3小时，须按真实M0/首轮校正；当前没有启动授权以外的额外服务器或进程。仅26GPU0/1，一个两卡job、条件顺序执行；无25/2/3、温度功率查询或环境重建。

主要比较joint_delta−add_delta，另比较joint_delta−joint_absolute；重用原raw semantic、独立shared-adapter global、无adapter作者来源基础及完整Signal的已封存账本，明确匹配范围与继承来源。项目推进线需mAP至少+0.5点且R1不降，完整Signal目标仍是另一条要求；接近之前不增加seed或调参，不因小正值自动扩三数据集。若项目化参考没有必要性，不将它算原创主模块；若两者都没有净增益，不再扫rank或倍率。

正式结果需同一mAP-best的全部CMC、50轮轨迹、同模型G/C/F、因子及乘积/差分幅度、全query修复/新增错误和身份AP分布、参数/真实墙钟；诊断NN必须另封存范围。所有best/唯一记录保留，完成strict/report后只归档核验再退休重复M0。官方基准已消费、固定模型bootstrap不替代训练seed的不确定性。

目标仍为同一网络模块的三个数据集全面超过Signal，先201。车辆只在机制获得实际支持后按既有数据集配方适配，不能分别挑不同模块赢家。当前没有新的正式分数、独有创新或失败根因结论。
