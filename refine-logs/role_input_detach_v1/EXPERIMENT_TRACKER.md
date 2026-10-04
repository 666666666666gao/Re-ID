# 角色读取梯度边界执行表

| Run ID | 阶段 | 目的 | 条件 | 数据 | 指标 | 优先级 | 状态 |
|---|---|---|---|---|---|---|---|
| D201-S | M0→fresh50→strict | 对照原semantic的读取梯度路径 | detached semantic | RGBNT201 | mAP/R1/R5/R10 | MUST | PREPARED_NOT_LAUNCHED |
| D201-N | M0→fresh50→strict | 对照原native的读取梯度路径 | detached native | RGBNT201 | mAP/R1/R5/R10 | MUST | PREPARED_NOT_LAUNCHED |
| D310-S | M0→fresh50→strict | 对照原semantic的读取梯度路径 | detached semantic | MSVR310 | mAP/R1 | MUST | PREPARED_NOT_LAUNCHED |
| D310-N | M0→fresh50→strict | 对照原native的读取梯度路径 | detached native | MSVR310 | mAP/R1 | MUST | PREPARED_NOT_LAUNCHED |
| D100-S | M0→fresh50→strict | 对照原semantic的读取梯度路径 | detached semantic | RGBNT100 | mAP/R1 | MUST | PREPARED_NOT_LAUNCHED |
| D100-N | M0→fresh50→strict | 对照原native的读取梯度路径 | detached native | RGBNT100 | mAP/R1 | MUST | PREPARED_NOT_LAUNCHED |

仅26GPU0/1，顺序一组两卡；原V6九端/§812六best诊断均已完成并封存。控制复用其原接受结果，不调用已退役旧M0二进制。没有新训练成绩；每端真实8-update M0通过才从相同初始化fresh50。完整Goal active/unmet。


## §41.814 真实执行起点

六项角色读取梯度边界对照已真实启动。RGBNT201 semantic 自己的8次有效更新M0通过，281/281张量有非零有限梯度，作者BN计数8、重载差0，匹配初始state/cfg/容量；fresh50已开始。当前M0 1/6、正式完成0/6，没有新的完整50轮分数。只用26GPU0/1，不管功率温度；科学Goal active/unmet。

观察：2026-10-04T09:54:15.361783+08:00；首项PID1313574，fresh50始于2026-10-04T09:53:44.383147+08:00。完整证据与不确定性边界见logs/role_input_detach_launch814_20261004。


§41.815：读取输入detach的RGBNT201 semantic完成50轮/2649步并通过首次严格重载：E18为72.7798/76.9139/84.9282/89.2344。相对原semantic，mAP/R1提高0.8817/2.5120，R10下降0.9569；仍低于独立global-only。六项正式1/6、M0 2/6；native已通过自己的M0并开始fresh50。只26GPU0/1，不管功率温度；Goal active/unmet。

原V6历史九端保持封存。新首项actual批次/init/源/实存权重与距离核对见logs/role_input_detach_first_formal815_20261004；不重跑旧M0或历史报告，不用首项局部增益完成科学Goal。


§41.816：读取输入detach的RGBNT201两端各完成50轮/2649步及首次严格评价。native E8为69.4305/72.1292/85.2871/90.7895；较原native的mAP/R1下降2.6968/2.9904，较同批semantic下降3.3494/4.7847。正式2/6、M0 3/6；MSVR310 semantic自己的M0通过并开始fresh50。只26GPU0/1，不管功温；Goal active/unmet。

完整配对/实存SHA/批次证据和只读资源补证见logs/role_input_detach_first_pair816_20261004。旧九端与退休M0边界保持封存，不修改负结果；新的六端终态、原一次CPU报告及同模型分解仍待完成。


§41.817：MSVR310读取输入detach semantic完整50轮/706步及首次严格评价完成，E49为50.7851/68.8663；较原semantic下降0.1785/0.3384。正式3/6、M0 4/6，MSVR native已开始fresh50。只26GPU0/1，不控制功率温度，保留原配方和单一best；Goal active/unmet。

证据logs/role_input_detach_msvr_semantic817_20261004；322源码/61原依赖/初始化/实际顺序SHA核验，非新种子复现。原报告一次与同模型固定best分解待完成。


§41.818：MSVR310 semantic/native均完成自身50轮/706步与首次严格评价。native E38为51.1388/69.8816；较原native变化+0.4632/+1.1844，较新semantic+0.3537/+1.0152。正式4/6；RGBNT100两端按原队列继续。固定best分解计划仅准备，输入尚未封存、模型尚未执行。只26GPU0/1，无功率温度控制；Goal active/unmet。

原始证据、actual order、weight/distance SHA及主指标完整表见logs/role_input_detach_msvr_pair818_20261004。后续固定best计划尚未执行，当前六端CPU报告一次仍待完成。
