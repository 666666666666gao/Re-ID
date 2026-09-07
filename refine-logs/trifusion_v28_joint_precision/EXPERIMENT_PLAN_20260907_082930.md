# V28 单批次 Mamba 梯度精度诊断

登记于2026-09-07。此诊断不恢复原M0，不执行Q1，不更改原M0_FAIL。
根因候选按顺序为：(1)AMP小梯度舍入，(2)真实输入/目标的局部低敏感性，
(3)融合CUDA与非融合反向执行差异。当前仅第一项是较可信推测，尚无数值证明。

原M0容量8步及固定100步均仅缺joint.mixer.dt_proj.weight非零梯度。
全局图已产生非零修正，T0实际FP32 Mamba全部参数连通，但这不能替代真实batch证明。

固定fold0合法V12初始化、seed42和前两批已存增强SHA。仅执行与原容量第1步
完全相同的一次optimizer更新以打开零输出投影；第2批只计算原损失及反向，
不再更新参数。两批loss须与原日志逐bit相等。不执行100步或任意更换样本。
捕获真实joint输入、三个输出的同一已缩放上游梯度、一次更新后的joint state；
持久化为远端诊断fixture，约230MB，不保存完整检索模型或访问heldout。

以相同输入、参数和上游梯度，独立重放joint子图：
A原AMP融合fast路径，B关闭autocast的FP32 fast路径，
C关闭autocast的FP32非融合路径(use_fast_path=False)。
A输出及dt_proj.weight梯度须与真实整图相同；记录全部16参数、输入梯度、
输出差异，不只报告目标参数。不扫描loss scale、LR、batch、宽度或loss权重。
若B/C仍为零则不能认定AMP根因；若B与C一致恢复非零，支持局部精度损失；
若仅C恢复，则需继续定位融合kernel，不能泛称AMP。

预算：仅1次重建optimizer更新，2次真实source模型前向，3次joint子图导数比较。
这是额外诊断成本，绝不计入原116步或写成0次真实更新。
BN buffers恢复，比较阶段参数SHA不变；原配置/源权重/失败日志不修改。
诊断源码tools/diagnose_v28_joint_precision.py。预计30–60秒。
输出只能为数值机制证据，不能带来新的M0通过、检索成绩或晋级。
