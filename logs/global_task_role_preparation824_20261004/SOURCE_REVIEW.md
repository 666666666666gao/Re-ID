# 执行者源码核对（非独立审计）

核对原AuthorHeadEvidence、读取detach入口、partition配置链、新模型/runner/queue/report和已有控制seal。
新模型没有新增持久参数或head；global正常head调用一次，fused以detach参数及clone buffers进行stateless调用。
CPU两种作者head布局证实初始state/参数量/输出相同，Lg/Lf梯度职责分离，BN每次只累积global调用。
配置链CPU检查证实真实inner/foundation build/loss/condition绑定与双卡placement保留；不是完整模型M0。
六端各自prepare绑定与匹配读取detach控制相比仅允许architecture/entry/scope/objective_policy改变；
实际state、camera、head、数据顺序、作者配方、seed、budget和已有role_input_gradient_policy必须一致。
报告复核当前六端、旧V6同variant及独立global三项比较，全量query/图库过滤不变，不调用退役M0验证。
这是一项联合训练职责控制，不能把直接global任务与head共适应的因果作用分别识别，也不能提前证明有效。
尚无真实prepare/M0/full50或性能结论；旧失败不改判。禁止依分数改margin/gain/LR/seed或中途加N2/N3。
