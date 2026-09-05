# V22 initialization full-gallery diagnostic tracker

更新时间：2026-09-05T21:17:51.497936+08:00；状态COMPLETE_READONLY_INDEPENDENT_AUDIT_PENDING。

| 步骤 | 状态 | 证据 |
|---|---|---|
| 执行前冻结 | DONE | INITIALIZATION_FULL_GALLERY_DIAGNOSTIC_PLAN.md/preregistration |
| 三个固定初始化完整检索 | DONE | evidence/trifusion_v22_initialization_full_gallery_20260905.json |
| 全45指标行/21身份/五路配对重算 | DONE | evidence/trifusion_v22_initialization_full_comparison_20260905.json |
| 30项完整文件SHA与state/梯度检查 | DONE | evidence/trifusion_v22_initialization_diagnostic_observation_210912_20260905.json |
| 独立诊断审计 | PENDING | 将交原始代码、计划、数组及来源绑定 |
| 新训练、checkpoint选择、D1/dev/official | NOT_RUN_NOT_AUTHORIZED_BY_DIAGNOSTIC | 不改变原V22资格 |


共同初始化的完整图库只读诊断已结束：原PID42325退出码0，74.088494秒，
3模型/26次模型调用/3126triplet前向，optimizer0、checkpoint写入0、dev0/official0。
三个初始state及binding均同时等于该fold两个终态保存的初始绑定；
评价前后state和30项source文件SHA不变，所有参数grad均None，baseline所有数组精确相同。

同完整图库/同输出的初始化fused80.590328 mAP/R1 83.712785，
普通终态80.640677/83.712785，mAP差+0.050348；MCNL终态差-1.605874。
普通终态相对初始化CNN/Transformer/Mamba差+0.729564/+0.539850/-1.044532，
MCNL分别-0.167748/-3.725590/-0.726907。不能用单一分支取代整体比较。
旧V12约88是eligible-only gallery下的residual/bank结果，不能直接与当前fused相减；
本次直接可比证据不支持普通继续训练使融合mAP整体下降，也不证明任何唯一失败原因。

全部45个fold/阶段/输出指标行、全21身份及两端五路逐query变化已经JSON/NumPy复算，
最大数值差0。原始758929字节 SHA21a73baacca91834eb5f47ec0c129731cfdb42ff92a5b90c2d712bef40f334ca，
日志6351字节 SHA756aa7a7eb67c8bde7c1ce41d62274677a62522cd01864bb4549f45d362c2a55。
结果及完整范围见results/TRIFUSION_RGBNT201_V22_INITIALIZATION_FULL_GALLERY_2026-09-05.md，
配套evidence/trifusion_v22_initialization_full_comparison_20260905.json保留所有21身份全部输出。
独立诊断审计待完成；不选择初始化作为部署checkpoint，不改变V22 Q1_FAIL/禁重训规则。
后继需要新的表征干预依据，当前未设计或启动V23；best dev与未达总目标状态不变。
