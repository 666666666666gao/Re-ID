# RGBNT100 全训练集主比较跟踪

更新时间：2026-09-06T17:44:01.445667+08:00；PREPARATION_ONLY_NOT_RUN。

| 运行 | 目的 | 范围 | 状态 |
|---|---|---|---|
| T0 | 全官方文件/标签/掩码合同 | 8675train+1715query+8575gallery，0模型 | 入口AST通过，NOT_RUN |
| Signal M0/B0 | 全50类完整同源基线 | fresh首epoch工程门→fresh固定30epoch | 驱动待实现，NOT_RUN |
| 角色M0/主训练 | 原三角色完整模型 | 8+100工程门→fresh固定20epoch | 驱动待实现，NOT_RUN |
| 官方完整比较 | 同协议Signal对照及公开差距 | 1715query/8575gallery、全部五输出与50身份 | NOT_RUN |
| 原内部OOF | 前置完整正结果 | 5195更新/8675query；mAP+1.792365pp | 已完成；不再训练 |
