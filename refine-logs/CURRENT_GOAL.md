# TriFusion 当前 Goal 执行约定

更新：2026-10-05 §41.835：职责干预六端全部完成50轮、首次strict及原唯一18组CPU报告，300轮/12968步；原父进程exit0，正式推进0/6。RGBNT100 native E26为83.1632/96.7930，相对semantic −0.9271mAP/+0.9329R1。六份已闭合M0探针退役，正式best及全部回执保留。固定best分解332source/187artifact已封存，尚未启动。仅26GPU0/1，不查询或限制功温；2025镜像原I/O待补，Goal active/unmet。

## 完整目标

在RGBNT201、RGBNT100、MSVR310匹配协议下超过baseline，注明预训练、外部资源、训练及推理条件比较强方法；形成机制必要性、同容量控制及完整流程多种子证据。普通配方收益、继承Signal能力与新增角色收益分账，不能弱化baseline凑十点。工程通过、单端收益或原报告完成都不能完成整个Goal。

所有历史在docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md及原archive中保留；本文件只给当前执行状态，旧观察不作为新启动指令。

## 已闭合、不得重跑的前置工作

- 原V6九端：logs/native_research_v6_20261003_794，九端各自M0/fresh50/首次strict及原单次CPU报告完成，450轮/19452步。原native−semantic推进0/3，不能视为稳定细节收益。
- 原V6固定best分解完成§812；随后读取detach六端和固定best分解完成§822–823，300轮/12968步。读取detach不是global保护的充分条件。
- 以上9+6份M0探针已按依赖闭合/回执/SHA退役，只保留M0文字及正式best；不能再调用依赖其二进制的旧verify/report。
- 旧反向parity FAIL与STOP记录保留；当前合同没有把parity修复设为前提，不重启工程修复，也不追认确定性问题已解决。

## 已完整闭合、不得重跑的训练合同

计划refine-logs/global_task_role_v1/EXPERIMENT_PLAN.md；campaign logs/global_task_role_v1_20261004_824，原launch logs/global_task_role_launch_20261004_824。原supervisor2620776、controller2620777；全部六端和原唯一CPU报告闭合前不重启或改源。

- 330项科学source封存不变，原公开CLIP、新camera/head、seed42、作者来源配方，完整50轮和合法query/gallery过滤。
- 双卡完整batch第一6个CLIP block在GPU1、后6及作者head在GPU0；201 B64/K8、MSVR B64/K4、100 B128/K16，maxparallel1。
- 角色读取stages/context/shared_global stop-gradient；原global任务L_g仅更新shared及原作者head。h=sg(g)+gain*c的L_f使用同一head值、detached参数及cloned BN buffer，仅更新roles/readout/gain。持久作者BN每batch更新一次；推理仍Normalize(g+gain*c)。
- 各端prepare→自己的8步真实M0→fresh50→首次strict。保持有限数值、有效参数更新、作者BN8、完整state重载与固定计分检查；失败保留，不重试挑通过。
- 不改LR/batch/AMP/margin/gain/seed/容差，不加N2/N3、loss、文本/SAM/DINO、教师、测试更新或新采样器。

## 最新已验收结果

同一mAP-best报告全部指标，车辆下表mAP/R1，201另有R5/R10完整回执。

| 数据集 | semantic | native | 相对独立global-only的推进 |
|---|---|---|---|
| RGBNT201 | 74.3749/78.8278，E8 | 74.4182/79.0670，E8 | 两端均未达条件；native−semantic +0.0434mAP |
| MSVR310 | 50.5422/67.8511，E38 | 50.5523/68.0203，E38 | 两端均未达条件；native−semantic +0.010120mAP |
| RGBNT100 | 84.0903/95.8601，E5 | 83.1632/96.7930，E26 | 两端均未达；native比semantic低0.9271mAP、高0.9329R1 |

六端既定推进条件0/6：至少+0.5mAP且R1不下降，只是项目门槛，不是统计显著性。职责干预使201/100的部分旧角色表现恢复，但不等于角色已产生稳定新增证据。实际batch顺序、源码、best/距离SHA和全部50轮已按各端核验；均为单seed42、官方基准已参与研发选择。

## 接下来按依赖顺序做

1. 原六端、12阶段、300轮/12968步及原唯一18组CPU报告已全部接收核验，不重跑原训练/评价/report。
2. 当前六份M0探针已在依赖闭合及新seal核验后退役；保留M0文本、六份正式best及必要历史控制。
3. fixed-best v2 wrapper/计划已登记部署，332项source及187项artifact新seal封存。完成四份执行副本核对后，执行六份固定best只读全query/gallery g/c/h/f分解；不得训练或修改原330项source。
4. 依据全六端与同模型分解再登记唯一下一训练假设；不凭小差值堆N2/N3。来源增量若成立仍须真实参与计算的同容量语义控制、完整角色作用路径删除并重训、完整流程多种子及强参照比较。

## 资源、保留和同步

只2026 gaob@172.19.12.138:2026，/data/gaob/Re-ID/Trifusion，复用tri_reid环境与数据。GPU2/3和2025不运行模型；不设置、监测或以功率/温度作为门槛。只清理本项目确认无用的权重，不触及其他项目；正式best、作者权重、初始化与当前依赖保留。

最新清理为本轮闭合六份M0共2,147,192,492B，实测剩余5,067,812,864B。原六端父进程exit0，已无训练活动；正式best、全部187项诊断输入及332source保持SHA。仅待已登记固定best分析启动；旧所有失败及清理记录保留。没有模型、环境或功温修订。

数据、模型、图像、NPY留远端；代码、文本、CSV/SVG及SHA同步本地/Desktop/GitHub/2026并核对。2025 /data2/gb/Re-ID/Trifusion仍有已记录真实I/O错误，文本镜像待补，最后验证§821；不冒称五份一致或擅自做恢复。

每个正式端只保留一份mAP-best，CMC随同一权重；保留原失败、50轮曲线、逐query修复/新增错误、身份收益分布、完整成本。identity bootstrap不是训练种子方差。只有完整性能、必要性和稳定性目标实现才标complete；合法等待不标blocked。
