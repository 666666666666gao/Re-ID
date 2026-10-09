# 冻结文本上下文 pilot：生产源码审查 R1

**结论：REVISE。2 项 BLOCKING；未发现其他确定的生产逻辑缺陷。**

这是 experiment-bridge Phase 2.5 的 fresh-context、same-family / provisional 源码审查。请求模型 gpt-6-astra，reasoning=max；实际模型及effort身份为 UNATTESTED。只读本地源文件并进行stdlib AST解析；0 SSH、0远端动作、0 NN/GPU执行、0安装、0源修改、0子代理。方法R2仍是有限pilot值得检验、paper REVISE；本结论不替代prefix/M0/正式成绩。

## BLOCKING

### B1 · P1：正式50轮缺少已登记的新参数更新与读取诊断

[新入口](C:/Users/gb/.codex_tmp/independent_evidence_draft/text_semantic_prior900/candidate_code/tools/run_text_semantic_prior.py:117) 的优化器post-hook、isolated role-gradient及诊断落盘都限定mode=m0（117–148）；正式train没有这些记录。[text forward](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/text_semantic_prior.py:117) 不保存text/context统计，[实际Q/K读取](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/slot_competition_fp32_roles.py:10) 不保存Q/K、logit_std或读取熵。[报告](C:/Users/gb/.codex_tmp/independent_evidence_draft/text_semantic_prior900/candidate_code/tools/report_text_semantic_prior.py:39) 读取steps后只汇总历史和计数。

继承的[损失诊断](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_role_input_detach.py:22)已有g/c/f范数，[训练循环](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_foundation_recipe.py:229)确实执行优化器；因此本项不是断言ψ/W没有训练。缺口是无法交付[计划B3](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/text_semantic_prior_v1/EXPERIMENT_PLAN.md:46)明确要求的正式ψ/W实际更新、text/context与Q/K/logit_std/读取熵轨迹。只保留best权重不能事后恢复这些观察，须在正式训练前接入。

最小修复：仅在新入口给现有ψ/text/W与query/key projection添加只读观察；在实际allocation_weights调用旁统计实际scores/weights并原样返回原weights；正式optimizer post-hook记录5个新tensor的实际梯度、初值delta及effective updates，报告消费记录。复用原g/c/f记录，不改变训练数学、初始化、RNG、步数或配方。

### B2 · P2：未验收集成训练中冻结T的完整149项状态

[冻结检查](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_foundation_recipe.py:165)只遍历named_parameters，训练前后分别在195/259调用；但T的[positional/template embedding](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/text_semantic_prior.py:74)是持久buffer。新入口67行只记录初始text_state_sha256，94–114行只验收eval/no-grad与新参数活动；[队列78行](C:/Users/gb/.codex_tmp/independent_evidence_draft/text_semantic_prior900/candidate_code/tools/queue_text_semantic_prior.py:78)接受的是frozen_parameters_unchanged。

[prefix66行](C:/Users/gb/.codex_tmp/independent_evidence_draft/text_semantic_prior900/candidate_code/tools/check_text_semantic_prefix.py:66)对独立合成组件做完整state比较是正确的，不能替代实际集成M0/正式训练的终态比较。完整严格保存/重载也已经实现，但不证明训练前后的两项buffer不变。[计划26–30行](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/text_semantic_prior_v1/EXPERIMENT_PLAN.md:26)明确要求冻结T全state不变。未发现现有forward主动修改buffer；这是明确验收缺口，不是已发生污染的结论。

最小修复：复用已有_module_state_sha256和初始text_state_sha256，在实际训练模型M0/正式训练结束时比较完整text.state_dict的149项并落盘；队列和报告验收此结果。不要把可训练BN buffer纳入冻结T判断，不引入新hash框架。

## 已核对且正确的部分

- 导入/configure链到达实际构造器与训练器；prefix显式ROOT/modeling路径修复存在。新builder在原common state完成后附加T，common hash比较没有混入新增状态。生产源码仍未安装，不能宣称运行导入通过。
- 普通causal forward_ori、12层FP32、EOT11、两模板、固定随机包填充顺序与私有CPU RNG符合提案。T参数冻结但保留输入autograd；augmented context没有再被父类detach。ψ/W的592,000参数/5tensor及cuda0位置与两GPU原分段一致。
- 原作者raw global/role任务、detached functional heads及克隆BN buffer、optimizer恰好一次覆盖和冻结参数排除正确。作者model/optimizer/loss/三套config/metric的本地快照hash与当前SOURCE_SCOPE精确匹配；保留各数据集原batch/K/LR/scheduler。
- M0的firstC有限非零梯度及更新、首步ψ/W预期0、steps2–8各新增tensor至少一次有限非零角色任务梯度及累计更新、原全部trainable活动、AMP有效更新和BN8次检查均接通。BatchOrderLoader在yield之后记账，实际8步日志正确；不存在第9批误记问题。
- 全持久state保存和strict load正确；M0不会温启动fresh50。单一fused mAP-best及同权重CMC指标、同一次visual forward提取own-global，原完整query/gallery与camera/scene GT过滤不变。
- 固定六端顺序、每端M0后fresh50、端级失败保留缺失并继续其余固定端、旧六控制复用、空间规划/已验收probe退役和原始报告一次调用及exit记录正确。完整CPU配对使用全部合法query的AP/first rank、修复/新增错误和身份宏平均，不补造缺失分数。

## NONBLOCKING 与执行边界

没有新增的确定NONBLOCKING代码缺陷。AST解析覆盖四个候选工具及core共5文件；审查前后这5个文件hash未变。真实prefix/VJP、GPU容量、逐端M0、实际共同初值/数据流、正式50轮及严格评价均未由本审查执行。私有队列引用的PRODUCTION_SOURCE_SCOPE.json仍需在原部署准备流程中装配，不把暂存文件当作已安装入口。

父执行者提出的最小hook/post-hook修复能闭合上述缺口。需保留原allocation_weights的autograd，仅统计用detach/no_grad；GPU标量读取会同步CUDA、增加墙钟成本，应如实计入。已有M0在foundation.train:268再build fresh模型做strictreload，新build不能覆盖原训练模型的观察对象/终态state比较，也不能把reload的eval forward计为优化步骤。修复后复核真实diff，无需新方法轮。

## 实际审查版本

以下仅记录已读取文件的实际SHA256；JSON附带42个审查输入的路径、字节数和hash，未创建新的版本绑定机制。

| 文件 | SHA256 |
|---|---|
| run_text_semantic_prior.py | dca3fd96a07d36714fbc7369c1124c4f19765c43e1cfd4dae914ac00015f1412 |
| check_text_semantic_prefix.py | f0b720cad3abaa7b337365e25380711cbd5a1964d2a93bd9800eeb25b844ecf9 |
| queue_text_semantic_prior.py | 739a8752ffa511d665d6cddca71965559584a859836120c84c2803093a2ac878 |
| report_text_semantic_prior.py | 960cb0eace0e3fcab20fd78f0a31a04ebdd8e11324ed99ec24a214d85f10eaef |
| text_semantic_prior.py | 4b1c5448929881f2914113fcf6fbaba18112da0eae0f9a7264bfe6b2e66d7382 |

机器可读记录：SOURCE_REVIEW_R1.json。

