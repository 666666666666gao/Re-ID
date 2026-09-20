# 支持感知梯度平衡工作记录

2026-09-21。READY_FOR_REGISTERED_T0_M0，仅seed42；尚未部署或训练。

| 项目 | 状态 | 内容 |
|---|---|---|
| 前序cross-scene Q1 | COMPLETE / CLOSED_WITH_LIMITS | 原科学FAIL封存，独立确定性核验通过 |
| 固定合同与实现 | REGISTERED | 完整R/A、支持状态、有界EMA、头部保持和CPU重算 |
| 独立代码审查 | PASS_WITH_LIMITS | fresh same-family/provisional；无剩余阻断；T0覆盖缺口已修源码、未运行 |
| T0 | NOT_RUN | 780来源batch合同与合成梯度检查 |
| M0 / CPU | NOT_RUN | 248更新及完整重算 |
| Q1 / CPU / 终态审查 | NOT_RUN | 三折两端1560更新，原两组五门 |

两端同cross-scene AP，仅balanced应用角色梯度组合。静态审查不是工程门或检索结果。分类头按原梯度更新，无支持步EMA保持；控制器统计含完整历史排名梯度。后续按T0→M0→M0_CPU→Q1→Q1_CPU持久队列执行，任一阶段失败停止，完整终态后独立审查。原失败不扫参、不挑checkpoint，无多seed或官方测试调参。
