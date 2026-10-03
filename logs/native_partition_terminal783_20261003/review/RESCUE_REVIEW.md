# V5 终态失败救援复核

时间：2026-10-03T18:28:08.9640862+08:00。结论：**FAIL_STOP_V5_NO_LOCALIZED_REPAIR**。V5保持停止；现有证据没有定位出可直接修复的源码错误。唯一建议是下述一次native原始实现对原始实现的无更新控制，尚未实现或执行。

本次按experiment-bridge的RESCUE_ON_FAILURE进行fresh Codex复核。父代理确认实际spawn请求为model=gpt-6-astra、reasoning_effort=max、fork_turns=none；这不构成独立服务端型号/推理强度核验。review_independence=same-family，acceptance_status=provisional。只有本地读取、JSON重计和既有清单核对；没有SSH、Torch/模型、CUDA、scorer、训练、更新、生产编辑或权重变更。

## 证据与终态

15/15 primary文本的字节数及既有摘要相符；312/312启动依赖与INTAKE一致，LAUNCH的312项来源映射亦一致。源码头499b2cc61e877a284af412e6ea97136c3dd7bfa5。这里仅核对既有收集机制，没有提出新指纹协议。

实际启动17:59:03.277076+08:00，controller3410754，物理卡0/1。父代理提供的首次300秒观察在18:04:27已见父进程不在。18:05:35.635528+08:00的INTAKE同样记录父进程缺失、原状态INITIALIZING/jobs[]、父退出码null。launcher.log记录backward子进程退出1及未捕获CalledProcessError；不能把它当成独立捕获的父退出码。采集工具退出0不是模型PASS。

只有RGBNT201三份初始化、一份完整B64初始pair、三份梯度测量。没有完整dataset backward PASS文件。optimizer更新、M0、正式50轮、终态报告调用和新增权重均为0。其余两数据集的初始化/pair未得到本轮完成证据。

## 逐项重计与勘误

计数来自原pre-assert文件的逐参数fixed_gate_pass；历史完整梯度没有保存，本审查没有重新执行tensor allclose。门始终为atol=rtol=1e-4。

| variant | 梯度张量 | 失败 | 分组结果 |
|---|---:|---:|---|
| global_only | 209 | 0 | Signal155、adapters54均精确 |
| semantic | 281 | 0 | roles70最大差3.2741809263825417e-11；其他211项精确 |
| native | 295 | 90 | Signal80/155失败，最大0.0068359375；adapters10/54失败，最大0.000244140625 |
| native其余 | 86 | 0 | roles70最大4.76837158203125e-7；detail14最大3.5087577998638153e-7；readout2精确 |

三组raw/fused/global/loss记录最大差均为0。native全部295项均非None；detail14只有output.weight的参考/候选梯度非零。其余13项的零梯度通过不能替代后续M0的14项累计活动和参数改变。

完整90项及其参考/候选最大值保存在配套JSON的all_native_failed_gradient_parameters。失败分布：视觉输入6项，block0..11各6/8/7/8/7/7/7/7/6/7/3/1项，adapter stage0六项、stage1四项。最大差在conv1.weight；traceback首先停止于cv_embed，差0.0010986328125。不能只报告首项，也不能定位成仅前六层失败。

**native author-head门已执行并通过。** 实际check_partitioned_backward.py:101–102先检查全部head score/feature的1e-5 allclose，之后109才触发梯度断言。103的梯度键集合检查亦已通过。head精确delta未落盘，不能称bitwise exact。

111–122的buffers、CPU/双CUDA RNG、BN1、完整state和hook验收未执行。生产finally清理临时hook的代码仍运行，但其验收断言未执行。global_only/semantic能进入下一variant，按实际控制流可知各自后续门已通过；仍没有完整dataset receipt。

原ANALYSIS.json boundary及任务简报把native head列入“后续未执行门”有误。父代理已确认保留原稿并勘误；90项梯度失败与0更新/0M0/0权重结论不变。

## 源码链和根因判断

实际队列按RGBNT201三次prepare、full initialpair、backward witness顺序执行；check=True使native异常阻止后续job manifest/M0/formal/report。INITIALIZING在检查之前写入，所以残留状态不代表进程仍活跃。

partition_visual_graph在optimizer之前移动前六视觉层和adapter0，保留Parameter身份及full-state断言。原CLIP residual公式、RGB/NI/TI顺序、role snapshot stack与三角色均值写回不变。两处pre-hook和stage4 snapshot.to保持autograd；没有detach、CPU-save、checkpoint或新stream。固定meta_arch.py:102将camera索引移到cv_embed设备。完整作者raw head/BN/Triplet、roles/native reader/readout在cuda0；独立QKV与唯一零output.weight保留。

这些源码事实不能证明实际反向等价。失败覆盖前后视觉层，角色/细节梯度的小差也不能定位产生点或放大路径。现有证据没有区分调度、atomics、layout、零分支、硬件差异或jitter，不把任何一个写成根因。旧check_native_original_backward_repeat.py只循环semantic；§776的semantic PASS不能外推native。

对“失败并应停止”的证据置信度高；对具体机制的置信度不足，没有已证实的可提交模型修复。V1 OOM、V2/V3 checkpoint失败、V4失败94和§780失败81继续封存；本次未重新执行或重新裁定旧路线。

## Blocking和非阻塞限制

B1：native90/295违反原梯度门，禁止原样重启V5或进入M0/正式训练。B2：完整witness及九端完整作者batch八更新M0均未完成，RGBNT100 B128容量没有证明；前32样本不能替代容量M0。

非阻塞限制保留：双卡峰值只在正常返回后写出；pre-assert缺少head/buffer/RNG等完整比较明细；完整CMC/逐query/最终评价时间汇总仍有缺口。原状态要和实际traceback一起封存，不能表示活跃任务。这些不构成本次梯度修复，不增加异常框架或兼容层。_source没有前审JSON，本次实际读取同次review_native_partition781/EXPERIMENT_CODE_REVIEW.json，并在配套JSON记录来源。

## 唯一下一动作

父代理先封存并同步V5失败和上述勘误。之后只实现并源码复核一个独立native/RGBNT201原始对原始控制；不修改生产模型、训练队列、配方或scorer。本报告没有执行该控制。

- 直接调用captured original_build_core或相同未分段builder，各构造一次两份fresh native模型；不调用partition helper，不装/撤checkpoint。两份起始完整state相等，并匹配既有native初始化state：7f5ff300a45faed287a510dacc68d06c006931504a4510fffd8c47e43d93b852。
- 原B64/K8 loader只取一个batch，固定前32输入；两模型共享冻结图像/labels/cameras/paths，并要求与本次native失败记录的输入摘要和metadata一致。完整目标metadata已写入JSON。state或输入不匹配就停止，不换batch或seed。
- seed42、原训练模式/作者loss/optimizer分组、FP32存储、autocastFP16、GradScaler256均保持。每次forward前恢复相同CPU和两张可见CUDA的RNG。只在2026资源实际空闲时采用原cuda0计算及相同双设备可见性/RNG记账；不抢占或用2025，不改backend、确定性开关、layout或stream。
- 严格限定一次调用、两个原始模型、各一次forward/backward/unscale。没有optimizer.step、scheduler.step、scaler.update、官方评分、候选重跑或自动后续动作。
- raw/fused/global/loss和全部head score/feature仍用1e-5；全部295梯度仍用1e-4且键/None/shape/有限性一致；buffers、CPU/双CUDA RNG、BN1、post-state、参数未更新与原capture清理继续精确检查。先落盘这些已经要求的比较，再断言；保留全部失败，不增加第二诊断臂。

**判读：** 全通过只支持此native原始重复witness；V5仍FAIL，尚无具体修复，不自动重启V5/M0/formal。若失败，原始native在此控制也不满足原门，不能把现有差异单独归因分段；封存并停止GPU parity试验，仅继续源码分析，不放宽阈值。state/input/runtime失败同样封存停止，不用更小batch、不同精度或重复直到通过。

V5完整梯度和pre-forward RNG张量未保存。新控制仅比较新执行的同state/input/RNG两份原始模型；即使起始state/input摘要相符，也不能声称重建了历史梯度或历史RNG实现。这个控制回答此前未测的问题，不能授予容量、科学有效性或完整parity PASS。

科学登记继续保持三variant×三数据集、公开CLIP及新camera/head/sharedadapters、seed42/full50、B64/K8/B64/K4/B128/K16、native159296参数/14张量、完整raw作者heads与L2 1536D部署。全部初始化、pair、parity及九端八更新M0门继续约束正式运行。
