# RGBNT100 Signal R2 完整内部基线终态

登记时间：2026-09-06T13:56:31.328098+08:00。状态 **COMPLETE_BASELINE_NOT_METHOD_QUALIFICATION**；执行侧完整文件和标量核验均 PASS。
固定三折30epoch全部完成，内部query加权 **89.5241750420 mAP / 96.8299711816 Rank-1**；
Rank-5 97.3832853026、Rank-10 97.5792507205。这是独立Signal基线，尚无RGBNT100三角色成绩。

## 数据与指标边界

仅RGBNT100官方训练中的8675条三模态拼图、50个身份，按原排序标签round-robin三折。
每条记录只在所属heldout fold参与一次query/gallery评估；各fold独立建立检索距离，
gallery分别3125/2950/2600，总8675，不跨fold拼接特征坐标。
严格只删除同身份且同camera记录，其余同camera身份仍保留为负例。
官方1715 query/8575 gallery及RGBNT201固定dev访问均为0。
不能将本表与Signal论文86.3或PMKD论文91.6直接相减，更不能称为官方复现或SOTA。

| Fold | source身份/记录 | query=gallery | mAP | Rank-1 | 有效更新 | source记录曝光 | 跨camera正对/同ID正对 |
|---|---|---|---|---|---|---|---|---|
| 0 | 33/5550 | 3125 | 82.7329880360 | 95.3600000000 | 2479 | 158656 | 475793/555296 |
| 1 | 33/5725 | 2950 | 90.2915446509 | 96.6101694915 | 2581 | 165184 | 497516/578144 |
| 2 | 34/6075 | 2600 | 96.8159900603 | 98.8461538462 | 2734 | 174976 | 529619/612416 |
| 合计/查询加权 | 每折身份隔离 | 8675 | 89.5241750420 | 96.8299711816 | 7794 | 498816 | 1502928/1745856 |

三折相差明显，合计是8675条query的加权结果，不是三fold均值。
三个source分别全部覆盖5550/5725/6075条记录和33/33/34身份。
完整训练跨camera正对占 **86.0854503464%**；此前M0的85.9384%另保留。
这是采样覆盖事实，不能单独证明覆盖率造成性能差异。

## 执行与工程定义

执行commit def7b9b7ecd9e7e37821716a13fdb2580b2d955c，wrapper87066/child87070。
实际启动2026-09-06T12:14:01.080628+08:00，完成2026-09-06T13:22:06.744688+08:00；
wrapper4085.661751532927秒，summary4080.2730049155653秒，exit0，三折0 AMP下降。
每折新CLIP/分类头初始化，与R2 M0初始state相同，没有加载M0训练权重；
固定epoch30唯一heldout终点，没有中间epoch选择或seed修改。

R2在已定位的真实Gram零点反向异常后登记：局部FP32 Gram后使用sqrt(abs(det).clamp_min(1e-12))。
下限以下volume为1e-6、对det梯度为0，不声称与无保护原公式严格等价。
原Gram/Patch权重各0.1、四组ID/Triplet、作者30epoch噪声cosine及各LR组保持。
B64/K8、workers4、同步几何/独立擦除为项目条件，非作者B128/K16原预算。
复用相同R2源字节的T0真实回执；没有再次执行T0/M0。
此前AMP停止、第34步捕获、FP32算子失败、稳定Gram回归、M0写盘失败均独立保留。

## 完整终态核验

远端CPU核验12.97555秒，0模型/图像/反向/更新：
3完整checkpoint内容、3特征距离数组、8675完整argsort和排名、10项目/21Signal源文件、
Signal commit/diff/整个CLIP权重、作者30epoch学习率与90条epoch记录全部通过。
三折保存距离与CPU重算逐元素相同；初始化同M0初始但不同M0训练后state。

本地JSON/stdlib核验7.94961秒：全部8675 query AP/首正例rank、50身份、
90epoch及7794逐步日志重放；7794/7794原分组FP32损失组合完全相等，
最大指标差1.4210854715202004e-14，epoch均值差0。
Python double重分组的最大1.13919e-6差异另列，不是原FP32损失不一致。
27份完整JSON/JSONL/文本/无损压缩排名共94457434字节逐文件SHA收取；
权重、features/distance张量留远端，没有本地模型、张量或图像运行。

- [完整原始summary](../evidence/rgbnt100_signal_v1_r2_baseline_receipts/baseline/summary.json)，SHA 549476408e82f085e7987fe087f6784773a54cd8f7eaf82e4e01faefd85711d5。
- [远端文件核验](../evidence/trifusion_rgbnt100_signal_v1_r2_baseline_terminal_files_verification_20260906.json)，SHA cfc819b5be9d51923f66d2f3f3cf005b6eb25add874bd8c3b49584a2ad72d230。
- [全部查询和训练标量重放](../evidence/trifusion_rgbnt100_signal_v1_r2_baseline_terminal_scalar_verification_20260906.json)，SHA 49f1d9a72617961967b54ed674d9c48c2a20197361d93fc7d802e7e9a4617706。
- [执行侧闭合记录](../evidence/trifusion_rgbnt100_signal_v1_r2_baseline_executor_closure_20260906.json)。
- 远端产物根：/root/trifusion-storage/artifacts/rgbnt100_signal_source_oof_v1_r2_storage_retry_seed42_20260906。

独立审计run20因服务额度限制无最终报告，仅涉及更早Gram工程153输入/973项重放；
不覆盖本次R2 M0/B0，不能将执行者核验标成独立审计PASS。

## 观察纠正与下一步

12:35首次观察时fold0排名文件仍在写入：当时7554299字节，关闭后16867041字节。
此前“完整排序文件已产生”只证明文件出现；原观察和纠正记录全部保留。
13:21已见90epoch/7794更新但最后检索仍在执行，以13:22:06真实terminal为完成时间。

本终态解除RGBNT100原完整三角色的B0前置条件。新配置将绑定真实epoch30 Signal及数组SHA；
先完成固定M0及全部核验，再新初始化执行三折20epoch比较。
V23/V24/MSVR310科学失败保持封存，RGBNT201 dev58.4050与主目标未达状态保持。
