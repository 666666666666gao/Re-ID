# §952：第一端真实M0通过，原队列进入full50

注册Git 5f43a55101f160bf27e9d54605d70cf3addb8209，同496 source；实际26 CUDA0/1 supervisor983280 / controller983289在2026-10-14T04:10:19.453006+08:00启动一次。第一个semantic_reference prepare EXIT0，共享初始化与原raw semantic匹配，训练参数88424906/280张量；新模式删除589824自由readout参数。

真实M08次有效更新、全部280张量非零有限梯度、BN8、reload最大差0；角色到shared_global梯度unused，实际attention_change获得非零有限任务梯度。当前只第一端通过，不是两端全通过或性能通过。原M0 probe保留，直到first strict/CPU报告关闭其依赖。

新fresh50在2026-10-14T04:11:44.859387+08:00以PID985316进入训练；2026-10-14T04:13:31.418568+08:00一次观测确认原controller/supervisor存活。首节点observer的parent campaign快照steps为空，实际child campaign和M0 receipt已同时采集；这里只在本地读取原child文件，不增加第二次SSH或NN。没有在该节点直接验证train PID存活或采集epoch history，不虚填当前轮数/分数。第二uniform_reference还在原串行队列，0新正式成绩。

唯一near-first observer native7321/localPID12744，出生2026-10-14T04:15:35.197769+08:00，等待2026-10-14T04:53:44.859387+08:00；新计算尚无实测完整时长，此为按旧同类训练约40分钟估计的42分钟节点。只在该节点读取原队列，睡眠240秒，不提前查epoch/GPU/T-P，不因本地观察超时重启训练或另建observer。

固定§950已保存local-only结果的CPU描述：QK mean相比G修复46个首位错误、新增78个，mAP差−1.4954；289query AP提高、350降低。此内容具备不同身份关系，也有明确损伤，不能称已经强于G、可直接成为教师或将其与G拼起来一定有效。训练两端不是按这些query选择test gate，而是训练集raw身份任务下学习真实value读取变化。

后继CPU配对报告代码已准备/AST检查，尚未执行；只在原两端50/首次strict完整通过后检查同批次顺序，报告semantic/uniform相互比较及各自对原raw semantic/global的净增量、首位修复新增错误、身份AP和全曲线。不重复原报告或新NN。所有原正式结果不变。

source496/best/初始化依赖不热改。只有26物理0/1、seed42、原warm环境/raw作者配方；没有功率/温度/seed/gain/margin/车辆或环境变更。闭合的D重复退休1.792GB记录见§951，需要旧consumer时先恢复。三数据集超过Signal Goal仍ACTIVE_UNMET，继续训练授权持续。
