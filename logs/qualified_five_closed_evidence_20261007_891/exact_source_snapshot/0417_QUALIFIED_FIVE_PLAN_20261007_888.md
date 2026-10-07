# 五个合格端进入正式实验，一项M0失败明确缺失

§886的新六M0已于09:24:19 terminal EXIT1。六端生产训练均为M0_PASS/每端8更新；五端独立增量活动验收通过，只有RGBNT100 repair_keep的第三角色query/key在八步未缩放、autocast禁用的辅助VJP中累计为0，unused=False，c和其余四QK非零。原controller状态仍RUNNING，但已有EXIT1原件；不要用状态字符串覆盖真实失败。新5/6不是三集普适通过，旧0/6亦不追认修改。

不继续原样重试、不改margin/gain/LR/尺度/阈值、不增加工程诊断来救最后一端。新合同显式只训练已接受的五端：RGBNT201两目标、MSVR310两目标、RGBNT100 md_batch_ratio。缺失的100 repair_keep没有正式权重/指标，不补零、不推测、不做三集完整repair_keep主张。模型、原两公式、作者raw任务、公开初始化、seed42、50轮、同best全部CMC、1536部署和NN执行均不变；新的full和report为显式五端入口，旧393/398/402/406/415已执行源码范围不变。

新的full资格依据五份已有真实acceptance与原EXIT1，而不要求把六端父队列改为COMPLETE。逐份检查8步/全参数/BN8/strict reload/初始化与batch匹配已通过的原件SHA、六QK/c正值、probe已退役；明确核验缺失端第三Q/K累计零且没有acceptance。正式五端重新构造公开模型，只继承初始化witness，不加载M0权重。控制仍为六个rawsemantic/global-only正式结果，不重训。250轮、9,839次正式更新，12配对：五端各对semantic/global，201/MSVR repair_keep另对同集MD。保持原全query分析、首位修复/新增错误、身份AP、完整曲线/E50与实际成本。推进线仍mAP≥0.5且R1不下降，不称统计显著性或训练多种子。公式有V26/R2/CIRC先例，不称新颖或完整MDReID复现。

源码范围在独立复审通过后登记，报告子进程显式导入五端panel以保持同一source_map。旧六端full仍NOT_RUN；新五端full也尚未启动。资源仍仅26物理GPU0/1/max1；无25、GPU2/3、功率温度操作，OWN NN活动期间禁止源码/Git/server sync。存储门仍保留5,192,548,352字节，不为这次失败降低；只在所有消费者闭合之后清理明确无用OWN权重，保存曲线/strict原件/best和official距离/失败证据，当前匹配控制、公开/作者和较强路线保留。

原probe的唯一post-eight诊断已经闭合；新failed100 probe不登记新消费者。它们可在保存精确SHA、训练/重载与失败证据后退役。清理后历史fixed9/187/241二进制直接重放需重生成，明确披露。新的正式首轮观察依据匹配历史耗时：2012350秒×2，MSVR1434秒×2，1007174秒×1，合计约4.1小时，增量开销下预计4—6小时；唯一observer首读3.5小时，之后240秒，不按官方中途指标改变固定方案。

这一合同继续已授权研究，保留缺失而非放宽失败条件。M0活动与正式检索有效性分开。Broad goal ACTIVE_UNMET。
