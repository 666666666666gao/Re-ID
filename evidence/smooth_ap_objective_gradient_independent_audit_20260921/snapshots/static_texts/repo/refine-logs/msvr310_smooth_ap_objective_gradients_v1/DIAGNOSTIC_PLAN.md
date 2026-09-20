# 同角色上的fused排名/其余13项固定状态梯度诊断

状态REGISTERED_NOT_RUN。0优化器更新，不是新训练。复用已封存Smooth-AP六个epoch20终态，原Q1_FAIL不变；只访问每折source图像。所有参数保持固定，每批恢复状态入口的全部buffer；train模式、真实criterion、原来源采样与像素、原角色入口RNG及512/8历史分组保留。

第一阶段：六端各8个来源batch，预热2步，总48批；与原登记前8个batch的记录及像素逐一核对。此阶段改变预热仅用于已声明的工程预检，不当作原训练复现。每端首个历史group作当前/历史两侧完整图梯度与分组VJP比较，原继承相对L2容差0.005。保存该步189个encoder参数的fused/other/full三组梯度张量，约480MB总计，留远端供独立数值重算。必须六端全部状态/像素/RNG/缓冲/有限性/分解检查通过，才允许第二阶段。

第二阶段：六固定终态各20轮原来源序列260批，共1560批；65步预热、512唯一记录、8批最大年龄。历史候选始终由固定当前参数及其原始view编码，全部历史field重放验证旧保存单位向量与fresh逐位相同；只跳过上游确为零的VJP组。只有当前64个anchor。control每步hard；candidate前65步hard，之后Smooth-AP tau0.01；其余13项及真实权重不变。

定义gF=梯度(weight*fused_current)+梯度(weight*fused_history)，gO=真实其余13项加权和的梯度；gTotal=当前完整目标梯度+历史梯度。其余13项通过将fused标量替换为常量0后调用原weighted_training_loss计算，避免总量相减的抵消。各角色使用同一参数块，记录gF/gO的norm、cosine、差范数、gF对gTotal、重复反传噪声；不直接比较CNN与Mamba异义坐标。

每步检查当前侧及完整侧gF+gO与gTotal一致：差范数除以两分量范数之和<=0.005，保留实际总量与差范数便于评估抵消。此分母用于数值恒等式误差，不是性能归一化。实际参数导数计算沿用AMP与固定SCALE=256；不恢复原训练各步动态GradScaler。首历史直接全图检查范围只有每端一个group，不能写成每一步完整图复核。全程梯度范数/点积为运行见证；除预检指定步外，不保存全量参数梯度张量。

所有1560批完整索引/像素/loss账本/距离/角色梯度统计保留，零范数和零历史情形也记录；无源身份筛选，0heldout/official前向。固定终态模型诊断不能解释为重建训练过程每一步的参数状态。与候选覆盖诊断分别报告，不叠加新损失、不据此宣称泛化提升。

运行入口：python -B -m tools.probe_msvr_smooth_ap_objective_gradients --contract configs/MSVR310/TriFusion-smooth-ap-objective-gradients-v1.json --root PREFLIGHT_ROOT --mode preflight。完整source使用同合同、不同ROOT、--mode source --preflight PREFLIGHT_ROOT/summary.json。每阶段持久进程/日志/退出码，失败保留现场，观察超时不重启。先实际预检核对和记录耗时，再登记完整运行的观察里程碑。

不建立或重建环境；复用tri_reid已核实Python3.10.14/PyTorch2.5.1+cu121/RTX3090，W&B=false。输出卷当前约6.87GiB。预计预检梯度张量约480MB、完整距离约120MB，文本与缓存另计；无新checkpoint。工程预检耗时尚未实测，不承诺全量完成时间。
