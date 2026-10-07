# Review Summary

日期：2026-10-07；完成2/2轮，最终6.35/10，REVISE；CALIBRATION:none。锚点逐字保持；完整研究目标ACTIVE_UNMET。

## Problem Anchor

- 最终问题：让新增分支为 RGB–NIR–TIR 身份检索提供 global 尚未解决的判别证据，最终在 RGBNT201、RGBNT100、MSVR310 三集上稳定超过匹配强基线，并按预训练资源与协议核实 SOTA；单集、工程通过和来源内拟合不能替代完整目标。
- 必须解决的瓶颈：分支活动、独立识别能力与最终排序净收益脱节。当前角色区域证据尚未证明能够在未见身份上修复 global 的错误而少添新错。
- 非目标：重复已闭合的容量、原生读取、轻量重建、slot/uniform、hard mask、独立头、detach、joint-L2、增量损失、gain、seed 或阈值搜索；不为论文预定三个有效模块，不把数学辅助目标改名当原创。
- 约束：仅 2026 物理 GPU0/1，一次一个 first6/last6 分段模型；现有 conda、数据与公开 CLIP 起点；不访问25/GPU2/3，不查询或设置温度功率，不安装环境。每个合格端独立真实8步M0后重新初始化完整50轮，同一 mAP-best 报告全部指标；失败端保留缺失；官方基准已参与开发须披露。必要权重与封存证据保留，无消费依赖的自训无用权重可退役。
- 成功证据：匹配配方、初始化、正式主训练批次和完整合法 query/gallery 的净收益；当前推进线保持 mAP 至少 +0.5 个百分点且 R1 不降。这只是研发决策线而非显著性；最终仍需强参照、完整流程多种子、机制必要性与实际成本。新单轮对照不等于完整目标完成。

## Round-by-Round Resolution Log

| Round | Main Reviewer Concerns | Change | Resolved | Remaining Risk |
|---|---|---|---|---|
| 1 | 同模态对应不等于身份互补；pre-key影响V；原Pad边界与RNG；真实Q–K链 | 保留0新增组件，锁H2a；修订梯度/支持/状态与own-g | 定义处理，证据未完成 | 新颖性弱、身份效用未知 |
| 2 | 定义改善不代表新机制；视图参数需固定 | 固定每模态batch共享flip/整数平移、顺序、pad值；删无学习信号三端完整控制 | 协议就绪，非源码PASS | 生产实现/M0/资源/完整成绩未有 |

## Final Status

方法可作为低置信度固定可证伪pilot，尚不是论文主贡献。没有文本、教师、分割器、队列、额外投影或第二损失。对应或注意力代理不替代身份指标，单seed三集方向不称训练稳定。后续只有原主要配对过线才另立监督内容必要性/完整流程种子合同；不按旧已消费官方结果调参救分。

原始评审请求/回复与第一次transport中断留在私有 prepool_method_review895。请求Astra/max；实际运行身份UNATTESTED/same-family/provisional，绝不声称跨模型验收。两轮不再增加。CPU40像素/面积参考、随后Torch40案例只证明几何实现；Torch首次归一化索引失败已保留，最小修正后通过，生产NN/optimizer均0。FINAL_PROPOSAL是唯一干净方案，源码尚待实施。

## Output Files

- 最终完整方案：FINAL_PROPOSAL.md。
- 每轮完整稿：round-0-initial-proposal.md、round-1-refinement.md。
- 公共评审摘要：round-1-review.md、round-2-review.md；完整原回复与prompt私有。
- 得分轨迹：score-history.md。
- CPU标签计算证据：logs/prepool_geometry_closed_evidence_20261007_895。

## Pushback / Drift Log

接受同图几何不是H2必要前提、K路径同时改变V、实际读取仍未被证明、人工pad边界支持修正。没有把锚点改成空间匹配问题，也没有为了READY9分添加新技术。REVISE保留，不以工具通过补论文评分。

## Next Step

依据固定低置信度效用合同实施，仅3新端M0→fresh50；不是对研究原目标完成的替代。实现/source审查、实际初始化、数据/RNG/BN中性和M0尚待验证，没有正在运行的NN。
