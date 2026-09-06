# V25 experiment tracker

更新时间：2026-09-06T22:33:37.459788+08:00。IMPLEMENTED_REGISTERED_NOT_RUN。

| 阶段 | 状态 | 完整范围 |
|---|---|---|
| SOURCE-METADATA | COMPLETE_PASS | 三折两端20epoch，3360batch，旧序列全部匹配 |
| T0 | PASS_METADATA | 全身份/全记录覆盖、两组跨相机、相同曝光预算 |
| M0 | TODO | 六端48只读前向，16容量更新与100步固定batch |
| Q1 | TODO_AFTER_M0_PASS | 六端20epoch，3360优化步，五输出全图库 |
| D1 / official / ablations | NOT_QUALIFIED_NOT_RUN | 原门槛保持 |

原始重放：468da3813828c49487c904249012f9c33b4b81c29fea7ccee1b6c4f242de2073；两端正对占比
8.070790816% →
12.344547194%。
模型训练与检索尚未开始，不能用元数据通过宣称模型有效。
完整计划SHA256：0399da7522f92c152bb453b3e3a6b5ebd6674e80c1fe02ba93e2e97aea47cc9f。
外部独立审计服务不可用，执行器核验与独立审计分开。
