# 原生研究 V6 执行表

§41.804：正式5/9；Goal active / unmet。仅26物理GPU0/1，25只同步文字，不恢复功温监控。

| 数据集 | global_only | semantic | native |
|---|---|---|---|
| RGBNT201 | 50轮＋严格评价，E8：74.2967/78.9474/88.2775/91.8660 | 50轮＋严格评价，E7：71.8981/74.4019/84.3301/90.1914 | 50轮＋严格评价，E20：72.1273/75.1196/85.1675/89.3541 |
| MSVR310 | 50轮＋严格评价，E38：50.5421/68.0203 | 50轮＋严格评价，E49：50.9636/69.2047 | M0通过；02:14:02 PID316812/R，fresh50已2轮 |
| RGBNT100 | 未执行；B128容量待验证 | 未执行 | 未执行 |

201 native−semantic +0.2291mAP/+0.7177R1，低于0.5mAP线；native−独立global-only −2.1694mAP/−3.8278R1。MSVR semantic−独立global-only +0.4215mAP/+1.1844R1，R5/R10为+0.1692/+0.6768，同样低于原0.5mAP线。native尚无MSVR正式结果。保留单种子配对窄结论，剩余四端原计划继续。

MSVR semantic E49→末轮仅回落0.0017644mAP，global E38→末轮回落0.0898254；这批MSVR没有明显后期退化，不能把201现象解释成三集共同根因。前两端各50轮/706条实际step、finite loss、严格重载分差0，一份正式best及两份距离、M0探针实存SHA已核验。实际batch顺序各1456580B/706行同SHA9c8a03beccbccedfa065cb13a3b20f5e64673d79693c4971034b54ab94e147e8；共同初始化字段及真实前向检查通过，314source未改。

原controller3854540运行；MSVR native fresh50于02:12:10.519895开始，PID316812/ticks30647128。只用26GPU0/1。原CPU九端报告尚未调用，M0探针保留到报告依赖闭合；不从当前成绩调学习率/倍率/种子或新增N2/N3。原始文本/分析在logs/native_research_msvr_semantic804_20261004，201完整配对在logs/native_research_first_dataset802_20261004。三集强性能、必要性、稳定性Goal仍未达。
