# 原队列失败后的未启动RGBNT100配对

2026-10-05 §41.842。原队列完整3端后，MSVR310 native的真实8步M0未覆盖细节stem首层weight/bias非零梯度，退出1。原campaign/日志/8步记录保留；该端不进入full，不延长M0、重换初始化或放宽门槛。

仅继续原来登记但从未prepare/M0/full的RGBNT100 semantic/native两端；单独campaign `logs/deployment_metric_role_pending100_20261005_842`。原失败队列的terminal SHA作为输入，执行前后不变，不修改它的FAILED或PENDING历史。

- 复用原339项封存模型/训练源码和187项控制；只新增该有限队列脚本，其SHA单独登记。
- 模型、joint1536 L2角色metric、raw global任务、车辆各头loss求和、初始化、optimizer、LR、seed42、B128/K16、增强、noAMP和原两卡分段均不变。输出目录改变不改变初始化合同。
- 每端按原函数prepare→真实8步M0→独立fresh50→第一次strict；各自一份mAP-best全部CMC跟随。
- M0或full失败即停止这个队列并保留失败；不重试，不能因读到官方分数换参数。
- 每个formal仅best；自身严格验收/SHA后才能退役自身probe，使用原函数的接受证书/journal。
- 两端best＋最多一probe＋原2GiB reserve，最低3,355,443,200B。训练前只做存储和GPU0/1空闲显存检查；不查询功率或温度，不使用GPU2/3/2025。
- 原六端唯一15对CPU报告需要六份正式接受回执，现在先保留report_invocations=0；不能以新两端的2/2冒充整个6/6。全局最终为至多5份正式结果＋1份M0失败，后续汇总须披露缺失范围。

源码复核：循环只有两个RGBNT100条件；无新的forward/loss、fallback、try/except或原科学源修订。原FAILED状态仅读取；所有完整校验复用原queue函数。此为同合同下独立未启动任务的继续，非失败端救分。
