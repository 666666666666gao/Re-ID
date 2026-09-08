# 单独补充审计：saved-source positive-relation coverage

完成时间：2026-09-08T23:20:33.774161+08:00。**审计结论：PASS，限定于完整保存来源快照的描述性统计及声明边界；same-family / provisional。** 确定性检查PASS；不产生新的科学晋级状态，不改变核心报告 `full_response.md` 中 Q1_FAIL、两组0/5、整体WARN的结论。

本补充由 fresh Q1 auditor 按追加请求单独完成；未将执行侧已有诊断结论作为判定依据。原源码、plan、原始JSON、聚合生成器、报告和aggregate都直接读取，并复制到本审计目录 `positive_supplement_input_snapshots/`。下文 file:line 以该快照目录为准；完整绝对路径与SHA见 `supplement_audited_input_hashes.json`。

## 独立执行与输入证明

运行 `remote_supplement_positive_replay.py`，远端只读、NumPy CPU、2线程环境限制，不导入训练或模型实现。重新哈希18份来源文件（六端各 training.json、memory_steps.jsonl、memory_distances.f32），逐项与Q1 CPU清单匹配；另外将CPU receipt SHA与本次核心审计值绑定，将protocol SHA明确绑定到注册值 `4ff4c60bca3d019929add5788212c526387d93d535a2c52aa7b1c3acfd387cb4`。所有20个远端输入与原诊断 `inputs` 完全一致。

原worker按排序负例+searchsorted计数，且仅首batch显式比较384个anchor（`role_set_source_positive_coverage_20260908.py:43-52,78`）。本审计采用独立方式：**对全部99,840个anchor建立实际正负距离显式广播比较**，累计214,672,384对，逐行重新构造全部指标。不调用原worker函数，也不复用它的Counter结论。

实际检查1,560步、116,501,504个四空间stream元素。诊断使用其中fused空间的29,125,376个距离元素；另外三空间被完整读取和有限值检查，不参与正例反序定义。成功耗时15.731秒；optimizer updates=0、model forwards=0、checkpoint loads=0、heldout rankings read=0、image reads=0、remote writes=0。原始stdout/stderr保留于 `remote_supplement_positive_replay.attempt1.json/.stderr`，一次成功，无本补充的失败重试。

全部6端×4阶段原始计数与原诊断JSON完全相同，全部2端×4阶段聚合与作者aggregate完全相同。phase为all1–260、warmup1–65、post_warmup66–260、last65196–260。独立重放还核对training记录的step与采样顺序，而非只读取其哈希。计算收据：`SUPPLEMENT_POSITIVE_COVERAGE_REPLAY.json`。

## 清单与判断

| 检查 | 状态 | 证据与判断 |
|---|---|---|
| 输入来源和完整范围 | PASS | worker第8–25、27–40、75–78行；独立收据读完所有stream并检查EOF；20个SHA匹配，3折2端各260步，无采样尾部替代全程 |
| 严格反序/等距 | PASS | 第46–49、62–63行使用left/right searchsorted，分别代表d(p)>d(n)及d(p)=d(n)；独立逐对`>`/`==`复算完全一致 |
| 非最大定义 | PASS | 第46、49、65–66行的hp取全部真实正例最大值，nonmax严格`pd<hp`；不是“未被argmax挑中”，全部最大值并列都排除 |
| identity/scene/self masks | PASS | 第31–45行与protocol逐项匹配；只排除自身当前位置，负例始终为不同identity，跨scene仅限制正例子集 |
| 重复曝光/历史位置 | PASS | 第60–61、67、69行分别计位置、每anchor去重record曝光、历史位置及同record其他视图；全部独立匹配，未混写成独立样本数 |
| 聚合账本 | PASS | `report_role_set_positive_coverage_20260908.py:13-29,31-35` 的三折求和及decomposition逐项匹配；作者Markdown表数值与对应聚合字段一致 |
| 直接距离导数与总参数梯度 | PASS（声明有必要限定） | plan第13段附近及作者报告“证据含义”段明确仅指本anchor该fused项的直接正例距离偏导；未宣称全模型零梯度 |
| 训练快照与最终评价 | PASS（声明有必要限定） | 作者报告开头及结尾明确不是固定最终模型的全source评价、不是新Q1/官方结果，不改变Q1 FAIL |

严格反序anchor计数与旧运行统计 `expanded_wrong_order_anchors` 的逐步相等还需要解释：旧统计是hp>=hn，补充定义是hp>hn。**本次所有正负相等距离对实际为0，因此两者在这份数据相等**。不能推广为带tie数据上永远等价，也不能把该断言本身当作严格定义正确性的证明。本审计的显式`>`/`==`比较覆盖了这一区别。

全99,840个anchor中仅fold0 role_set有1个anchor的最大正例距离并列，且它存在严格反序；其余五端无最大值并列。该并列的所有最大位置都正确纳入max类，不误入nonmax。正负等距对六端均0。这是对实际保存浮点值的精确比较，不使用容差将近等距改成tie。

## 关键数值的独立复算

预热后每端三折585步、37,440 anchor曝光；末65步每端三折195步、12,480 anchor曝光。

| 计数 | 预热后control | 预热后role_set | 末65 control | 末65 role_set |
|---|---:|---:|---:|---:|
| anchor 曝光 | 37,440 | 37,440 | 12,480 | 12,480 |
| 存在严格反序的 anchor | 4,432 | 4,478 | 941 | 962 |
| 存在非最大正例反序的 anchor | 2,602 | 2,618 | 472 | 474 |
| 有跨 scene 正例的 anchor | 15,376 | 15,376 | 5,152 | 5,152 |
| 受影响正例位置 | 12,003 | 12,104 | 2,239 | 2,272 |
| 其中严格非最大正例位置 | 7,571 | 7,625 | 1,298 | 1,310 |
| 受影响跨 scene 正例位置 | 9,441 | 9,505 | 1,848 | 1,857 |
| 其中严格非最大跨 scene 正例位置 | 5,900 | 5,925 | 1,061 | 1,060 |
| 全部严格反序正负对 | 43,499 | 43,723 | 6,638 | 6,728 |
| 非最大正例严格反序对 | 22,949 | 23,109 | 3,238 | 3,308 |
| 有跨 scene 正例但所有最远正例同 scene 的 anchor | 2,013 | 1,999 | 745 | 739 |

因此末65 candidate受影响正例位置2,272，其中1,310严格低于本anchor所有正例的最大距离；其中1,060为跨scene。这些数值及control的2,239/1,298/1,061完全一致于原报告。它们表明所记录训练快照中仍有未位于正例最大值的严格反序位置，不构成最终固定模型的全source检索性能。

非最大是在全部真实正例上定义的，之后才筛选cross-scene子集。因此“非最大跨scene正例”不是“低于跨scene正例内部最大值”。这一区别对“所有最远正例都同scene、但存在跨scene正例”的1,999次candidate预热后anchor曝光尤其重要。

## 为什么直接正例距离导数的限定成立

注册两个fused目标对正例距离的入口都是hp=max(all positives)，角色集合只扩展负位置；详见核心快照 `snapshots/tools/msvr_role_set_relations.py:25-55`。严格小于hp的独立距离坐标d(i,p)对本anchor该目标的直接正例距离偏导为0；无margin严格反序也不等于带0.3 margin的hinge全体活跃关系。这份补充只统计`d(p)>d(n)`，没有把它当作全部hinge违反数。

核心审计已经对相同全部保存矩阵验证两个目标的距离坐标导数；这为上述partial derivative定义提供独立计算依据。该partial以其余距离坐标固定来解释。实际d(i,p)由共享embedding和encoder参数产生，同一record可作为其他anchor/candidate，也有其他13项损失与角色路径监督，因此不能据此推断该图片、特征、encoder参数、某角色或总梯度为0，更不能断言模型永远无法间接改善这些位置。

“最大值并列均从nonmax排除”是保守的类别划分，不宣称每个最大并列位置一定获非零导数：当前max/min的首索引tie约定，以及当前/历史极值分摊约定仍按核心目标实现执行。额外的零梯度、正负抵消、优化器变换都未由本补充重建。

## 结论边界与时间记录

这份诊断合理支持“已有source训练快照中存在严格非最大正例反序，跨scene子集亦存在”的有限描述，不足以证明多正例/AP损失更好、最大正例有噪声、当前方法存在唯一失败原因、或已有负关系扩展造成某种泛化结果。已审核作者文字保留了这些边界，未发现通过位置曝光放大独立样本量或将本诊断升级为新benchmark成绩。

该计划的当前文件mtime早于结果文件mtime，和“运行前写计划”一致；本审计没有不可变、独立可信的注册时间戳，因此不将本地mtime当作强预注册证明。无论该时间断言如何，本诊断发生在已知Q1 FAIL之后，是事后机制描述；没有借此重新选择checkpoint、门槛、模型或训练。

本补充不要求修改实验、不选择温度/间隔/候选预算、不建议新的训练。核心Q1结果和完整性审计原文保持独立。供调用方记录：supplement_status=`PASS_DESCRIPTIVE_SAVED_SOURCE_SCOPE`，review_independence=`same-family`，acceptance_status=`provisional`。

机器可读审计：`SUPPLEMENT_POSITIVE_COVERAGE_AUDIT.json`；完整补充原文：`supplement_positive_coverage_full_response.md`；独立数值收据：`SUPPLEMENT_POSITIVE_COVERAGE_REPLAY.json`；输入清单：`supplement_audited_input_hashes.json`。
