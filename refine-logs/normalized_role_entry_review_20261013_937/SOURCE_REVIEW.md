# 最小角色分类入口：源码与实际配置链审查

**根源源码审查在本项范围内未发现阻塞，第一次实际入口绑定核查通过；仍不具备真实初始化、M0或正式训练成绩。**训练停止与待回复的训练范围问题保持。§936 CPU合成head结果不重跑，也不升级为生产M0。

固定审查父提交`e1983eabe877effd8afeabe89a48980ff72d7173`，规范是用户最小范围要求与§936 COMPONENT_PLAN；使用code-review-excellence/Python指引。只由root审查，**不是独立实验审计、外部模型复现或科学有效性PASS**。

## 具体源码结论

| 需求 | 实际实现及证据 | 范围结论 |
|---|---|---|
| 只支持201 semantic | 新driver build_core显式限定dataset/variant；wrapper限定author direct | 源码符合；CLI/真实构造未执行 |
| 只改变角色分类入口 | normalized=L2(raw)，role_score=detached classifier(detached BN(normalized)) | 符合；BN仍在L2之后，非直接余弦head |
| Triplet保持raw | 新wrapper heads=[(role_score,raw)]，实际绑定原global_task_role loss_values | 符合；原作者损失读取tuple的raw feature |
| global任务独立 | global_score仍classifier(BN(global_feature))；原global与角色loss求和 | 符合；936有限fixture已检查梯度归属 |
| 不新增参数/头 | 新wrapper仅重写forward，无构造器或注册参数；复用原detached_head | 符合；真实参数所有权需M0核查 |
| 保持推理及单次编码 | 非return_aux返回原output.fused，每次仅调用一次forward_features | 符合；936 fixture结果保持，不是全CLIP向量实测 |
| 不持久更新角色BN | detached_head参数detach、buffer.clone，只有global调用实际BN | 源码与936 fixture支持；真实BN运行未测 |
| 不改变采样/配方/分段 | 原builder、loader、optimization/train调用绑定保持 | 实际函数绑定通过；生产数据/optimizer仍未执行 |

未增加fallback、try/except、额外兼容层或假设性分支；不对已封存旧来源作风格重构。wrapper与旧forward之间的小段重复用于独立登记当前单项对照，不能据此修改封存旧入口。

## 实际入口检查

一次warm conda CPU导入／配置检查：2026-10-13T21:30:37.036724+08:00→2026-10-13T21:30:38.964588+08:00，原native `be0939` EXIT0，状态`PASS_ENTRY_BINDINGS_ONLY`。CUDA_VISIBLE_DEVICES为空。原474项运行来源加936四项源码/计划执行前后哈希保持。

实际捕获的六个builder函数对象各不相同，次序为：

`run_normalized_role_classification → run_global_task_role → run_role_input_detach → run_native_research → run_native_partitioned → run_independent_native_evidence`

首次外层configure后，以及正式inner.main会再次执行的inner.configure后，12个实际绑定保持相同，包括inner/foundation build_core与loss_values、AuthorHeadEvidence、新condition、DetachedSemantic初始化类、loader、optimization、分段train；schema为`trifusion-normalized-role-classification-v1`。此次**未调用inner.main本身或六个builder**，没有假构造成功的INITIALIZATION_VERIFIED文件。

调用计数：outer configure1、inner configure1；builder/model/NN/loss/autograd/optimizer/prepare/M0/train/evaluate均0。没有GPU任务、真实数据、权重写入/删除或新指标；配置函数调用不计作真实训练。

## 尚未完成的资格与科学问题

本项只排除了实际配置链覆盖新wrapper／恢复旧损失绑定的担忧。公共CLIP与camera/head初始化、完整state和optimizer覆盖、真实输入下BN运行及角色参数活动、8步更新/重载、完整50轮仍待实际核查，不能写成“生产资格全通过”。训练范围问题尚无人类回复，不依赖自动Goal continuation启动上述工作。

§936分类入口本身是诊断对照，仍不是论文创新；旧训练空间差异已在Oct5提出。合成共同偏移例子不证明它解释实际损失，也不证明新入口会超过raw semantic/global-only/Signal。原失败、best、官方集已消费及单seed42边界保持；不增加seed/gain/margin/层数搜索或车辆队列。

三个数据集全面超过Signal与有必要性的原创证据机制仍未达到，Goal为ACTIVE_UNMET。下一项依赖明确训练范围回复；若保持停止，就只可做有新问题依据的离线工作，不原样重复本次绑定检查。没有本次无用权重需要清理。
