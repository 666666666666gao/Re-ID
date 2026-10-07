# 固定五best的全局／修正／融合分解

状态：SOURCE_ONLY / NOT_RUN。五端正式50轮与12组配对已闭合；主要推进0/5、全部配对0/12，失败100 repair_keep仍无正式权重或成绩。本计划不重训这些端，也不改变损失、gain、学习率、margin、种子、验收阈值或已选epoch。

明确缺口：当前正式距离文件仅保存fused和真实标签／camera／scene元数据。训练步骤中的修正范数不能证明选定权重在官方query/gallery上的global质量、独立修正能力或纠错／新增错误。因此需要读取原五份mAP-best，补充同权重的g/c/f诊断。

固定范围：201 md_batch_ratio、201 repair_keep、MSVR md_batch_ratio、MSVR repair_keep、100 md_batch_ratio；共5个独立子进程顺序执行。仅26物理GPU0/1、单个split模型、现有conda；不接触25/GPU2/3/功率温度。五份原权重与当前控制、公开初始化和固定协议均按原SHA封存。

实现复用已使用的extract、真实camera/scene评分、paired与describe函数。每条query/gallery记录只执行一次eval/no-AMP前向，同时取shared_global、correction、raw_fused、fused；g/c仅在单独诊断时L2归一化。使用原初始化JSON构造同类模型，严格重载同一best；原模型参数／buffer前后SHA相同、无梯度、0 optimizer updates。fused与原首次严格评价全部四项误差须小于原1e-5个百分点；保持原h=g+gain*c和f=L2(h)的1e-6检查。原距离差值作为描述保留，不借此修改旧反向重复性结论。

全量query/gallery：201为836/836、MSVR为591/1055（含全部合法干扰身份）、100为1715/8575；合计16,926条记录。原标签、排序和camera/scene数组须逐项相同。每端报告同模型g→f、独立global-only→同模型g、独立global-only→f，全部query AP/首位修复与新增错误、身份宏平均AP；另外保存c独立检索、query/gallery的修正能量和夹角。只作固定权重描述，不将同模型g当独立消融、训练范数当检索互补、身份重采样当种子稳定性。

源冻结：保留既有420件源字节，新增诊断脚本、本计划和两件SOURCE_ONLY复审文件，共424件。独立seal保存原五best/official距离/receipt、五初始化、当前45/48输入、公共CLIP与协议的真实SHA；启动前确认原五端控制器EXIT0、无OWN NN、显卡0/1空闲。诊断沿用既有固定best诊断的2GiB磁盘门；预计新增全量特征及距离约0.6GiB，不修改正式训练5,192,548,352B的存储门。

预计10—25分钟：三组件评分／距离、完整前向与重复SHA相较单次正式strict评价有额外开销；唯一observer首读8分钟后，再每240秒。运行期间不做源码/Git/服务器同步，不退役当前五权重或其他消费者输入。出现首个失败即原样记录并停止队列，不挑重试结果、改checkpoint或放宽容差。

分支判读：若g不同于独立global-best，描述其中epoch选择／模型差异而不唯一归因于损坏；若c有判别性却f损失，则支持研究互补学习；若g保留而c/修正几乎没有作用，则支持研究上游证据形成。无论哪种结果都不自动证明P1/P2/P3有效、新颖公式、三集稳定增益或SOTA。Goal ACTIVE_UNMET。
