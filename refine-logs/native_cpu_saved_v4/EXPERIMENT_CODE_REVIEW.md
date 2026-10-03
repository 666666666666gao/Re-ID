# Native CPU saved-tensor v4：fresh CODE_REVIEW

**结论：PASS（SOURCE_ONLY）。** 没有发现阻塞问题、非阻塞代码缺陷或必须修改的源码。审查时间：2026-10-03 15:35:01 +08:00。

本次按 experiment-bridge Phase 2.5 审查，独立上下文、same-family、provisional。请求配置为 gpt-6-astra / max；实际模型与 reasoning effort 身份未独立验证。本结论不构成运行等价、资源适配或科学成功的证明。

## 范围与依据

审查起始工作树为 `C:/Users/gb/.trifusion_github_publish_22c3bee`，HEAD `27a1d4bd7423a52bc658fb03ac48b92272879e8a`。读取 `refine-logs/native_cpu_saved_v4/EXPERIMENT_PLAN.md` 及新6个Python：`cpu_saved_evidence_clip.py`、`run_native_cpu_saved.py`、`check_cpu_saved_native_pair.py`、`check_cpu_saved_backward.py`、`queue_native_cpu_saved.py`、`report_native_cpu_saved.py`。

旧依赖以 `.codex_tmp/independent_evidence_draft/original_repeat_intake776/source` 的真实runtime快照为准。对照实际 `INTAKE.json`，293项源码与6项primary文本逐一校验，均无长度/哈希不符。6个新Python及封存257个Python均通过AST解析。纯stdlib helper mock在 training/grad 四组合下验证了参数转发、每次仅调用原forward一次、仅 training且grad启用时进入 `save_on_cpu(pin_memory=False)`；未导入torch或模型。

## 正确性判断

1. **改动边界正确。** 16行helper保存原始bound forward，仅在训练且梯度启用时增加saved-tensor CPU上下文。没有activation checkpoint、重算、额外stream、压缩、新Parameter/buffer、模块注册或state key。原 `CrossLayerAdaptedCLIP.forward` 的hook注册/清理、模态顺序与返回路径保持。角色运算、Mamba、native detail、作者heads及loss均在该上下文外。见helper第7–16行、新入口第17–36行及封存 `correspondence_roles.py` 第53–90行。

2. **接口替换和ownership正确。** 新入口在configure前捕获原构造器，随后沿原entry/foundation的动态接口接入；M0、train、evaluate和fresh reload均使用v4构造器，没有串入V2/V3 checkpoint实现。原作者优化器仍接收完整模型，检查全部可训练参数恰好覆盖且无重复。作者BN mode、一次raw-feature路径、一次heads/loss调用保持。seed42、FP16 autocast、GradScaler256、full50，以及201 B64/K8、100 B128/K16、MSVR B64/K4保持；没有将32样本witness偷换成正式批大小。

3. **门禁保持。** 所有初始化、3项full-batch eval pair及9项original/CPU-saved生产AMP对照完成后才进入M0。forward/loss/head的1e-5、gradient的1e-4、gradient-None支持一致、state/buffer/RNG exact、BN1及hook清理断言保持；梯度差与有序输入元数据在数值容差断言前落盘。每个M0独立执行8次有效更新、BN8、完整优化器ownership及严格全state reload；native全部14个detail张量须获得非零梯度支持并在第8步改变。全部9个M0通过才启动full50，失败停止后续阶段且没有自动重试。

4. **科学路径与评估保持。** native159296/14参数、独立detail Q/K/V、原128-token semantic读取和单zero exit未改。raw feature进入作者损失，L2_1536用于部署。单mAP-best checkpoint提供全部既有指标，完整query/gallery、数据集camera/scene过滤和GT评分路径保持，严格reload后与保存指标校验。终端报告保留完整50轮轨迹、实际batch-order一致、repair/new-error及identity AP；+.5 mAP且R1非负只是阶段标准，不能当作显著性结论。

5. **source与队列承接正确。** 新source_map先核验实际293项，再合并11项，其中3项为原有源码、8项新增，完整组装数为301。2026固定路径、物理GPU0–3、启动/调度时显存小于500MiB、最多4个单卡任务和240秒轮询保持，无抢占、无重试。固定预算未改变：campaign估计8,145,338,368B，加2,147,483,648B运行reserve，初始free门槛为10,292,822,016B。

## 部署前提与审查限制

公开 `logs/native_original_repeat_result776_20261003/terminal/INTAKE.json` 在审查开始时尚待发布；必须使用本次真实intake的相同副本。父执行方已确认：保留远端原293项，仅叠加6新Python、计划和本审查MD共8项，随后以既有source_map核验301。该发布工作树有27个仅换行差异和107个缺失的封存路径，未发现实质内容差；不能以整个本地modeling目录覆盖runtime。以上是明确部署前提，不是要求增加兼容代码或新哈希协议。

实际空闲卡须以启动检查为准；历史观测中的GPU1有他人进程，并非全部卡空闲。本次不SSH、不导入模型/torch、不使用GPU、不修改科学源码。stdlib mock只证明包装分支与调用关系，不证明PyTorch数值或内存行为。

V1完整B128 OOM、V2/V3原AMP梯度失败均保持；一次semantic RGBNT201 original/original控制仅支持该次281梯度PASS（max abs 2.9103830456733704e-11、BN1/state/RNG exact），不能重判V3、唯一归因或推出普遍确定性。3个精确旧V1 probe已退役，其文本/哈希保留，二进制直接重放已退役。退役边界free为10,770,534,400B、先前MemAvailable为116,887,232kB，均不是本次实时容量证明。

**无v4真实运行成绩。** 完整B128显存/主存峰值、耗时、实际AMP等价及全部9个M0仍须按现有计划实测；本PASS不声称资源适配、科学收益、鲁棒性或SOTA。

## 必须修改的源码

无。blocking_issues、non_blocking_code_defects、required_source_fixes均为空。

