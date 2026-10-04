# TriFusion 当前 Goal 执行约定

更新：2026-10-05 §41.836：当前职责六端fixed-best g/c/h/f分析全部完成并核验；332source及187原输入、模型参数和buffer不变，原fused距离差全部0，无优化更新。201/MSVR同模型global指标恢复到独立global，但角色增量仍薄；100修正使同模型mAP下降0.2789/0.8022。原六端训练与全部分解已闭合；下一项先检验角色训练度量与部署几何，不叠加N2/N3。仅26GPU0/1，不管功温；Goal active/unmet。

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

1. 原六端fresh50、首次strict、原18组CPU报告及六端fixed-best分析全部闭合。不重跑原训练、报告或诊断；332source/187原输入保持封存。
2. 全六端表明global职责保护已经恢复部分主干检索，但未形成稳定角色增量。唯一下一候选为角色度量直接使用完整1536维L2(h)，原global作者任务、分类头、BN更新、结构/容量/推理保持不变；车辆同时改为联合模态度量，不能称仅归一化。
3. 下一候选尚未实现/登记/运行，先完成最小源码、匹配初始条件和真实M0计划，再开展匹配seed42/fresh50。不扫描LR/gain/margin/seed，不叠N2/N3，不把普通损失接口校正当模型创新。
4. 仍需真实同容量来源控制、完整角色作用路径删除重训、完整流程多种子和强参照；单次源代码或计分通过不能完成Goal。

## 资源、保留和同步

只2026 gaob@172.19.12.138:2026，/data/gaob/Re-ID/Trifusion，复用tri_reid环境与数据。GPU2/3和2025不运行模型；不设置、监测或以功率/温度作为门槛。只清理本项目确认无用的权重，不触及其他项目；正式best、作者权重、初始化与当前依赖保留。

最新清理为本轮闭合六份M0共2,147,192,492B，实测剩余5,067,812,864B。原六端父进程exit0，已无训练活动；正式best、全部187项诊断输入及332source保持SHA。固定best六端已经完成核验；下一候选尚待源码/合同/执行登记；旧所有失败及清理记录保留。没有模型、环境或功温修订。

数据、模型、图像、NPY留远端；代码、文本、CSV/SVG及SHA同步本地/Desktop/GitHub/2026并核对。2025 /data2/gb/Re-ID/Trifusion仍有已记录真实I/O错误，文本镜像待补，最后验证§821；不冒称五份一致或擅自做恢复。

每个正式端只保留一份mAP-best，CMC随同一权重；保留原失败、50轮曲线、逐query修复/新增错误、身份收益分布、完整成本。identity bootstrap不是训练种子方差。只有完整性能、必要性和稳定性目标实现才标complete；合法等待不标blocked。
