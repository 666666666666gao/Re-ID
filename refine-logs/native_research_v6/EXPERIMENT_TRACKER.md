# 原生研究 V6 执行表

§41.803：正式4/9；Goal active / unmet。仅26物理GPU0/1，25只同步文字，用户取消功温关注继续有效。

| 数据集 | global_only | semantic | native |
|---|---|---|---|
| RGBNT201 | 50轮＋严格评价，E8：74.2967/78.9474/88.2775/91.8660 | 50轮＋严格评价，E7：71.8981/74.4019/84.3301/90.1914 | 50轮＋严格评价，E20：72.1273/75.1196/85.1675/89.3541 |
| MSVR310 | 50轮＋严格评价，E38：50.5421/68.0203 | M0通过；01:49:02 PID246040/R，fresh50已5轮 | 未执行 |
| RGBNT100 | 未执行；B128容量待验证 | 未执行 | 未执行 |

201 native−semantic：+0.2291mAP/+0.7177R1，R5+0.8373、R10−0.8373，未达原0.5mAP推进线；native−独立global-only：−2.1694mAP/−3.8278R1。保留单种子201窄结论，剩余五端继续原计划，不修改结构或配方。

MSVR global-only包含共享适配，不能等同F1无adapter基础。该端50轮/706真实step、loss有限、严格重载分差0；一份正式best及两份距离、M0探针实存SHA已核验。角色及细节增量尚无MSVR配对结果，不能填0或从201外推。

201三端实际batch顺序同SHA e446282654c74949fb594856631b246642593b73229331c30b555418a0358993，各5475022B/2649行。MSVR global-only actual order为1456580B/706行、SHA9c8a03beccbccedfa065cb13a3b20f5e64673d79693c4971034b54ab94e147e8；另外两端完成后才检查本数据集配对。MSVR真实初始化检查通过，共同state与semantic/native零出口初始行为核验；314source未改。

原controller3854540持续运行；MSVR semantic fresh50开始01:46:04.602622，PID246040/ticks30490536。01:45观察是阶段交接快照，不重启已完成global PID180327。原CPU九端报告尚未调用，M0探针保留到报告依赖闭合。原始文本/分析在logs/native_research_msvr_global803_20261004，201完整配对在logs/native_research_first_dataset802_20261004。三集强性能、必要性、稳定性Goal仍未达。
