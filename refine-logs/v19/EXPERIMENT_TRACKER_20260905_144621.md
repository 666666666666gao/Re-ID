# V19 执行记录

| Run ID | 阶段 | 目的 | 对比 | 范围 | 指标 | 状态 |
|---|---|---|---|---|---|---|
| V19-T0 | T0 | 新wrapper/optimizer行为 | 冻结/可训练私有尾部 | 远端单元测试 | 输出/重载/storage/更新范围 | DONE_PASS |
| V19-M0 | M0 | 真实工程合同 | 三折配对、两端容量、实验端过拟合 | source-only | 梯度/显存/溢出/过拟合/冻结 | DONE_PASS |
| V19-Q1 | Q1 | 完整身份隔离主比较 | 三折×两端×20epoch | 3126gallery/571query | 五路mAP/R1/R5/R10/全部AP | DONE_Q1_FAIL |
| V19-AUDIT | 审计 | 全部结果与来源核验 | 全部端 | 完整Q1 | 完整性/科学门 | DONE_WARN_Q1_FAIL |
| V19-D1 | D1 | 同协议dev主结果 | 当前最佳与exact Signal | 141-fit/30-dev | >=65mAP及固定严格胜出 | NOT_RUN_Q1_FAIL |

最新终态（2026-09-05）：三折×两端×20epoch全部完成，3360步，Q1_FAIL。
fused增益+0.256035、fold增益+1.619524/-0.900867/-0.001279、Mamba增益-0.422824，
21身份bootstrap下界-1.615129；五项科学门仅fused胜过同checkpoint各路通过。
全部3126gallery/571query保留，overflow0、strict reload、冻结及配对检查通过。
执行端已重算全部数组/bootstrap并在远端核验29文件SHA与六receipt；完整证据与
五路/三折/21身份表见results/TRIFUSION_RGBNT201_V19_PRIVATE_TAIL_2026-09-05.md。
完整Q1独立审计已完成：工程PASS、integrity WARN、科学FAIL。V19封存，不执行D1/dev/official或任何超参/层数/epoch扫描。
以下带时间的RUNNING条目均为历史进度，不能覆盖本终态。

2026-09-05 12:26:41 CST：远端T0四项测试通过，6.95秒；M0全部检查通过。
M0两端203/311训练张量均有有限非零梯度，峰值reserved6478/6814MiB，
overflow0；固定100步超额损失比0.0595140216，冻结state不变。
原始M0快照：`evidence/trifusion_v19_m0_seed42_4b749cd.json`；远端SHA核对
见`evidence/trifusion_v19_m0_transfer_receipt_20260905.json`。
源提交4b749cd，12:15:41在screen `18809.v19_private_tail_4b749cd`启动；
Q1当前fold0对照第9/20epoch，尚无完整最终检索结果。首端约35秒/epoch，
六端连同评价预计75–90分钟，按真实后续时长修正，不因中间值改变预算。
只允许seed42、远端GPU；无消融/扫描/official。D1/dev/official访问仍为0。
冻结方案为同目录EXPERIMENT_PLAN.md；既有V17/V18固定方案与记录保持原样。

M0独立审计已完成，见`EXPERIMENT_AUDIT_V19_M0.md/json`：工程完整性PASS，
overall/integrity WARN（远端字节持有和未完成Q1范围限定）。这不代替表中的
V19-AUDIT完整Q1终态审计。原始报告与证据均保留，训练配置和晋级条件不变。

12:48:21 CST进度：fold0两端均完成20epoch/580步，overflow0，strict reload
及只读评价通过，配对sample/augmentation/initial-state/baseline输出相同。
保留该折全部1000gallery/190query；当前fold1对照第2/20epoch，共完成2/6端。
Q1状态仍RUNNING，整体科学门等待全部端终态；预计13:36–13:51完成。

13:14:21 CST进度：fold0/1四端完整完成，共2280优化步；全部strict reload、
只读评价、overflow0、冻结状态和两端采样/增强/初始状态/baseline配对检查通过。
前两折保留2051gallery/369query；当前fold2对照第3/20epoch，27batch/epoch。
剩余两端完整预算预计1080步，最后一折1075gallery/202query必须全部保留。
Q1仍RUNNING，等待全六端统一科学结论；下一阶段观察13:21–13:22。

Q1独立审计：EXPERIMENT_AUDIT_V19_Q1.md/json。审计员实际使用NumPy2.5.2
重算所有数组、标签mask、科学门和10000次身份bootstrap，数值一致；远端大权重/
数据仅有receipt与两项本地CRLF/LF来源字节差异的限制保留。Q1_FAIL与无D1不变。
随后六端只读表征诊断全部完成329.784541秒，全部原五路检索数组精确复现；
七个来源分类头和来源fused均100%，诊断报告另列全部模态几何及因果解释限度。
该诊断无训练/dev/official，尚待其独立范围审计；不能并入已完成的Q1审计范围。


### 独立诊断审计闭环（2026-09-05 14:43 CST）

EXPERIMENT_AUDIT_V19_GEOMETRY.md/json已完成；工程完整性PASS，overall/integrity
WARN，scientific_qualification FAIL，V19 Q1_FAIL不变。审计员从原始JSON独立
重算60个fold指标行、20个aggregate指标行、42个fold分类头、14个aggregate
分类头、108个模态对、24个分组几何与5路配对距离统计，汇总差异为0。
源码记录的fold均值与逐样本float64重算最大差9.68e-8，属于记录精度差异。
source6252为fold-model memberships，唯一物理记录3126；每身份恰在两折来源
训练集合中。权重/数据远端字节仍只经receipt绑定，独立审计未读取本地模型。
报告不支持唯一因果、未来干预收益或dev/official/SOTA主张。审计原文及调用
trace完整归档于.aris/traces/experiment-audit/2026-09-05_run05。
