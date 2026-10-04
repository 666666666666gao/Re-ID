# TriFusion 当前 Goal 执行约定

更新：2026-10-04 §41.814。六项角色读取梯度边界对照已真实启动。RGBNT201 semantic 自己的8次有效更新M0通过，281/281张量有非零有限梯度，作者BN计数8、重载差0，匹配初始state/cfg/容量；fresh50已开始。当前M0 1/6、正式完成0/6，没有新的完整50轮分数。只用26GPU0/1，不管功率温度；科学Goal active/unmet。

## 研究目标与当前阶段

推进 RGBNT201、RGBNT100、MSVR310 多光谱 ReID：在匹配协议下超过 baseline，按预训练、外部资源、训练与推理协议比较强方法，形成有必要性、稳定性和完整流程多种子证据的机制。当前九端先回答 native 相对 semantic 的细节增量、semantic 相对独立 global-only 的角色增量。global-only 含共享适配器，不能等同 F1 无模块基础；配方收益、继承能力与模块贡献分别记账。

执行 refine-logs/native_research_v6/EXPERIMENT_PLAN.md。每端先做实际完整 batch 的 8 步 M0，核对有限数值、有效更新、作者 BN、完整 state 和严格重载；通过后从相同初始化 fresh 训练 50 轮，再独立评价。旧反向重复性 FAIL / STOP 记录保留，当前 V6 不把这项修复设为前提，不宣称已解决确定性问题；现有 M0、容量、数值、更新、重载或计分失败仍停止，不能重试挑一次通过。

## 固定研究边界

- 相同公开 CLIP、新 camera/head、匹配作者来源配方、seed42、完整 50 轮；完整 query/gallery 和各数据集正确 camera/scene 过滤。
- 数据集顺序 RGBNT201、MSVR310、RGBNT100；各为 global_only、semantic、native；同一双卡分段；B64/K8、B64/K4、B128/K16 不改，RGBNT100三条件真实B128/K16各自M0与完整50轮均已实测通过；不追认旧parity修复。
- 当前九端不加入 N2/N3、新 loss、文本/SAM/DINO、教师或测试时更新；不因官方分数调整学习率、batch、AMP、倍率、margin、seed 或容差。
- native − semantic、semantic − 独立 global-only 分开报告。+0.5 mAP 和 R1 不下降是项目推进条件，不是统计显著性或 SOTA 证明；有收益后补实际参与计算的同容量语义控制，再验证必要性及完整流程多种子。算子消融需删除其完整作用路径并重新训练。

## 最新执行状态：全九端与原一次CPU报告已收齐

原生研究V6九端全部完成各自真实M0、fresh50和首次严格独立评价；450轮/19,452步，原控制器全量CPU报告仅调用一次并退出0。完整结果、修复/新增错误与冻结八端复核归档；仅26GPU0/1，不管功温。Goal active / unmet。

完整配对、首次严格评价、source314/实际batch顺序/9份best与M0 SHA核对见logs/native_research_complete809_20261004。原报告依赖已闭合，发布核验后再清理九份临时M0；正式best与初始化保留。强baseline/SOTA、机制必要性、完整流程多种子尚未完成。

## 最新执行状态（覆盖下方带时刻的历史观察）

正式8/9。RGBNT100 semantic完整50轮/3129步，E5为83.4395/96.0933；比匹配独立global-only低1.0943mAP/0.5248R1。native自己的B128/K16 M0通过，fresh50已启动；06:43原PID901390存活、1/50轮。只用26GPU0/1、不管功温，Goal active / unmet。

原队列3854540继续最后RGBNT100 native，PID901390/ticks32250108，fresh50启动06:39:20.321738；06:43:17已1轮。全九端各自真实M0通过，不代表旧parity修复。原CPU九端报告尚未调用，保留全部M0探针至verify/报告依赖闭合。末轮和首次严格评价前不填写native正式成绩。完整原始回执与配对分析见logs/native_research_rgb100_semantic808_20261004。

## 资源与当前执行

仅 2026 物理 GPU0/1，一对双卡实验；不使用 GPU2/3，不在 2025 运行训练、conda 或模型，2025 仅同步文字。长任务按 180–300 秒或预计里程碑观察，不因观察超时重启。

用户最新指示“不管功率和温度了”。23:54:04核对PID4013580、UID、启动ticks和完整命令后，仅SIGTERM本项目只读功温观察器；23:58:11确认该/proc已不存在，semantic训练PID4081872仍R，controller3854540仍活。没有发送训练信号，没有设置硬件，不再以温度或功率作为当前执行门，也不继续采样或反复设置。旧权限失败、热暂停及监测数据保留为历史证据。

RGBNT201三个条件均完成50轮、各2649条实际step及独立严格评价：global_only E8为74.2967/78.9474/88.2775/91.8660；semantic E7为71.8981/74.4019/84.3301/90.1914；native E20为72.1273/75.1196/85.1675/89.3541。§805当时正式6/9；MSVR三端各50轮/706step：global-only E38为50.5421/68.0203，semantic E49为50.9636/69.2047，native E38为50.6755/68.6971。

native−semantic为+0.2291/+0.7177/+0.8373/−0.8373个百分点，未达原0.5mAP推进线；native−global_only为−2.1694/−3.8278/−3.1101/−2.5120。只能说201当前单种子独立细节读取小幅正向，但没有抵消完整角色路径相对global-only的损失。项目门槛不是统计显著性；不能宣布无身份价值或三数据集普遍失败，不改已登记配方。

三条件实际training_batch_order.jsonl各5475022B、2649行，SHA均e446282654c74949fb594856631b246642593b73229331c30b555418a0358993。共享初始化字段及真实初始前向已核对，native零出口与semantic初始输出/heads一致。50轮step和loss有限、同一best、严格评分差0，三份best及距离实存SHA通过，每正式端仅一份best。

MSVR native于02:36:51.524761完成首次独立严格评价；三端初始化字段及actual batch顺序同SHA，314科学source未改。RGBNT100 global-only M0于02:40:18.617584验收，B128/K16真实8次有效更新、213/213非零有限梯度、三个作者BN各8批、重载差0。fresh50开始02:40:22.839966，02:45:30 PID385541/R、ticks30816360，已2轮，原controller3854540仍S。只证明global的B128反向容量，semantic/native各自M0仍待执行；不改变batch、累积或重计算，不追认旧parity已修复。

原始文字、SHA/实际batch顺序绑定及分析见logs/native_research_first_dataset802_20261004；前两端历史见§800–801。native末轮mAP69.994009，best回落2.133267；计时与显存按机器分析分账，history.seconds只含循环。三端CPU距离/逐query净修复报告仍等全九端终态后由原入口调用一次。M0探针为该报告verify依赖，暂不清理；01:20:19磁盘约34.94GB可用。

MSVR新正式回执在logs/native_research_msvr_global803_20261004：50轮/706实际step、finite loss、同一E38 best、严格重载分差0、实存best/两份距离/M0探针SHA通过。训练循环451.215330秒，完整训练命令1329.690852秒，独立评价41.866760秒；循环时间不包括逐轮评价/保存。01:49:18磁盘可用31462952960B；全九端CPU报告仍未调用，继续保留其M0依赖。

MSVR前两端实际batch顺序字节相同，各1456580B/706行、SHA9c8a03beccbccedfa065cb13a3b20f5e64673d79693c4971034b54ab94e147e8；共同公开初始化与真实full-batch检查已核对。semantic相对global-only四项为+0.4215/+1.1844/+0.1692/+0.6768个百分点，mAP仍低于原0.5推进线。单种子正配对不是稳定性或SOTA证据；没有按它改变剩余四端。

semantic末轮mAP50.96181677927415，比best低0.0017643847556992；当前MSVR两个完成端均无明显best→末轮退化，不能把201早期峰值/后期回落泛化到所有数据集。原始回执、SHA、706step和配对分析见logs/native_research_msvr_semantic804_20261004。02:14:17可用29708836864B；全九端CPU报告未调用，M0探针仍保留其依赖。

当前细节配对：201 native−semantic +0.2291mAP/+0.7177R1，MSVR −0.2881mAP/−0.5076R1；两集尚未支持稳定额外收益，RGBNT100完整配对仍待完成。MSVR native−global仅+0.1334mAP/+0.6768R1，不能用这个分母覆盖native低于semantic的事实。维持剩余三端原合同，不叠N2/N3或临时调学习率/倍率/种子。

MSVR native best38→末轮回落0.1563519mAP，三端当前均无201那样数点退化。新回执、完整step、三端SHA/batch绑定及RGBNT100实测M0在logs/native_research_second_dataset805_20261004。全九端CPU报告仍未调用，保留M0依赖；02:45:30磁盘可用28481900544B。仅26物理GPU0/1，不恢复功温监控，25文字同步。

## 评价、保留与完成标准

每端完整 50 轮，只留一份 mAP-best，R1 和 RGBNT201 R5/R10 跟随同一 checkpoint，严格完整 state/BN 重载和固定计分规则。保留逐 query AP、首个合法正例名次、修复/新增错误、身份收益分布、50 轮曲线和实际成本；披露官方集参与研发选择，固定模型 bootstrap 不替代训练种子不确定性。

只清理确认无用的本项目重复或临时权重；作者与输入权重、当前正式 best、报告验证仍依赖的 M0 探针及失败证据保留，不触及其他项目。模型、图片和距离大文件不上传，只同步代码、文字与验证摘要；主交接文档本地、Desktop、GitHub、2026、2025 五份一致。

完整三数据集性能、必要性与稳定性要求均实现后才标 complete。M0、工程检查或首个完整端不能完成整个 Goal。


六端完整轨迹补充（§41.806）：201 raw融合范数增加、MSVR下降；best到末轮mAP分别明显回落与接近零，不能统一归因尺度增长。日志未拆CE/Triplet，也未记录细节出口幅度，不补造诊断；完整300个epoch派生证据及文本SHA见logs/native_research_training_traces806_20261004。正式仍6/9，等待原定04:40窗口和剩余3端，未调参。


§41.807：RGBNT100 global-only正式完整验收E7 84.5338/96.6181，50轮3129step；只计算独立global能力，本集角色/细节配对尚缺。semantic自己的真实B128 M0通过8有效更新/285非零有限梯度/三个BN各8/重载0，04:41 PID634130/R已2轮；native M0仍待原队列。原始文本/SHA/步骤见logs/native_research_rgb100_global807_20261004。当前正式7/9、最终9端CPU报告未调用/M0依赖保留；不调参、不重启。


§41.808：RGBNT100 semantic E5 83.43950031243456/96.0932970046997，完整50轮/3129步严格验收，四项均低于同配方独立global-only；mAP/R1差−1.0942839135407496/−0.5247771739959717。native M0通过8有效更新/299非零有限梯度/三个BN各8/重载0，06:43原PID901390存活1/50；九端实际M0均通过，正式8/9。继续原固定实验，不调参、不重启、不新增模型分支；最终报告/清理依赖尚未闭合。


§41.809：九端完整50轮共450epoch/19452step，原控制器全量CPU报告一次/退出0；记录、最终配对与冻结八端advisory复核已完整归档。下一研究问题依据完整证据决定，不以端数通过或loss下界替代检索Goal。


## 完整九端结论与唯一下一问题（§41.809）

1. **独立原生读取未达到三集推进条件（0/3）。** native−semantic在RGBNT201为+0.2291mAP/+0.7177R1，RGBNT100为−0.8785/−0.5831，MSVR310为−0.2881/−0.5076。201有小幅正结果但低于事前0.5mAP条件；不能把它覆盖另两集的负结果，也不能由此断言原生细节永远没有身份价值。
2. **原角色路径仍未优于匹配全局适配（推进0/3）。** semantic−独立global-only在201为−2.3985mAP/−4.5455R1，100为−1.0943/−0.5248，MSVR为+0.4215/+1.1844。MSVR有正结果，尚低于项目mAP推进线；冻结不是此轮的解释，因为视觉共同参与训练。该结果约束当前完整角色系统，不证明某个算子单独无用。
3. **新增证据同时修复和破坏排序，仍需分离其作用位置。** native−semantic首位修复/新增错误分别为201 60/54、100 31/41、MSVR 4/7。对应身份宏平均ΔAP为+0.1012、−0.1649、−0.2824；它与query加权mAP不是同一统计量。既有报告只保存fused距离，不能确定差异来自同模型global变化、角色修正自身效用，还是二者兼有。

先做固定正式best的只读完整query/gallery诊断：同一前向返回g、c、h、f，使用实际训练所得gain检验同模型global→fused的增量，并与独立global-only区分；记录修正能量、方向与native出口活动。它用来定位当前负结果，不重新训练、不改权重/gain/损失、不会升级为因果证明。提案尚未登记/实现/启动；不自动堆入N2/N3或搜索新种子。下一新训练假设依据这份诊断确定。

单seed42、官方集已参与epoch/方法选择、native增加159,296参数、无同容量控制；固定模型identity bootstrap不代表训练种子稳定性。旧反向parity FAIL保留。完整SOTA/必要性/全流程多种子Goal仍未达到。


§41.810：原九端一次全量报告/§809五份发布闭合后，九个精确成功M0探针已退役，实删3196826712B；9正式best/CLIP/回执不变。旧M0依赖校验不可直接重放，不追认二进制仍保留。记录见logs/native_research_m0_retirement810_20261004。完整性能Goal仍active/unmet。


§41.811当前下一执行已登记：refine-logs/native_fixed_best_diagnosis_v1/EXPERIMENT_PLAN.md，六份已定best全量只读g/c/h/f诊断，尚未GPU运行。原九端训练/唯一报告完成，不重启旧队列；M0退役边界不改。新训练待完整诊断后另行登记；Goal active/unmet。


§41.812固定六best诊断完成/父子exit0；同模型g/c/h/f与native真实出口已量化，原fused四项复现、模型/buffer与原61依赖未变。下一训练尚未登记：依据完整六端分开检验global共同训练差与证据读出增量，不重复旧九端、不放宽门槛、不自动叠加N2/N3或扫描参数。完整数据见logs/native_fixed_best_complete812_20261004。Goal active/unmet。


§41.813下一唯一六端：refine-logs/role_input_detach_v1/EXPERIMENT_PLAN.md（固定前向/容量/作者配方，仅角色输入detach），当前prepare/M0/full尚未启动。原V6九端和§812固定best诊断封存；旧M0已退役不能重放其依赖verifier。新端自己的真实8步M0通过后fresh50，原控制保持，不加N2/N3/调参/工程修复。Goal active/unmet。


§41.814六项真实队列运行中，首项201semantic自己的M0 PASS/fresh50开始。具体执行refine-logs/role_input_detach_v1/EXPERIMENT_PLAN.md，原V6九端已经终态封存；新M0自己的验证不调用旧退役M0。首项281/281梯度、BN8、重载差0；正式0/6，不预写新分数。原同容量/初始化/配方，只有角色输入梯度边界改变，global仍可训练。只26GPU0/1、不管功率温度。科学Goal active/unmet。
