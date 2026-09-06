# RGBNT100 原完整三角色 M0 完整终态

更新2026-09-06T14:11:01.774271+08:00。**PASS_ENGINEERING_ONLY**，执行侧全部权重/源绑定/124步标量核验通过。
这是工程及固定batch优化能力检查，heldout/dev/official为0，不是三角色检索增益。

执行9e908c8e82b0653164f94e2e7572ac781e802eb5，wrapper92186/child92190；
14:01:21.442232启动，14:05:20.261521完成，exit0。
wrapper238.8160716秒，summary232.2851100秒；计划300秒首查，实际14:06:33.454192观察，
距启动312.011960秒，完成时间采用terminal而非观察时间。

## 固定范围及实际结果

三fold分别fresh角色，各8步B64/K8；再fresh fold0对同一个固定增强batch恰好100步。
合计124有效更新/7936source训练记录曝光，0 AMP下降；每阶段203个可训练张量全部有限非零梯度。
完整Signal及冻结state不变，角色state实际变化；三fold保存和strict reload五输出逐元素相等，
独立Signal与baseline前缀真实source检查PASS。
clean角色72条记录、独立Signal24条，heldout0，不是官方评估。

| fold | 总参数 | 可训练参数 | 有限非零梯度 | peak reserved MiB | 8步容量秒数 | source前缀/严格重载 |
|---|---|---|---|---|---|---|
| 0 | 97022989 | 6248460 | 203/203 | 5974 | 10.659862 | PASS |
| 1 | 97022989 | 6248460 | 203/203 | 6230 | 9.666065 | PASS |
| 2 | 97052173 | 6274572 | 203/203 | 6230 | 9.732790 | PASS |

实际可训练参数与预登记6248460/6248460/6274572完全一致。
分类头对应33/33/34个source身份，不把类别数变化说成新增主干。
三个分支都执行共享tail，以上不是FLOPs/部署延迟，也不能仅用可训练参数声称低推理成本。

固定100步：initial loss2.8034026622772217，final0.4944220185279846，
七头label-smoothing下界0.4908334477121887，final excess ratio **0.00155176796144928**。
原门0.1通过；不选中间最小值，不延长步数或重新抽batch。
该阶段耗时108.9309988秒，peak reserved6234MiB。

## 完整核验与证据

远端原核验器11.5175259秒、wrapper14.1869111秒：三个完整checkpoint内容、
31项项目绑定、21项Signal绑定/commit/diff/完整CLIP以及所有124步原记录通过。
三个保存Signal state每个张量逐元素等于B0。
本地原JSON/stdlib核验0.0715102秒：所有124条source索引、8×8采样、
203项gradient finite、损失原FP32分组、冻结状态/解析下界/4条epoch日志全部重算，
124/124损失完全相同、epoch均值差0。23份原始文本/JSON/JSONL共5451771字节按SHA收取。
权重、张量和图像不下载到本地运行。

- [原M0 summary](../evidence/rgbnt100_original_roles_v1_m0_receipts/m0/summary.json)，SHA b2ba13b644033f2888f2c7eb4535e1b117f9175e2cef412f06aa2c438466a2f8。
- [远端完整权重/文件核验](../evidence/trifusion_rgbnt100_original_roles_m0_files_verification_20260906.json)，SHA ce71f8ed6363cec4e3dd65de1a5f71bc13e6feb2c27d37d97cf1e3876dcd851c。
- [本地全部标量核验](../evidence/trifusion_rgbnt100_original_roles_m0_scalar_verification_20260906.json)，SHA 49d50e12358f01072e9515150114c0001f8e6242f964d32e3dcf06a016de5f10。
- [执行侧闭合](../evidence/trifusion_rgbnt100_original_roles_m0_executor_closure_20260906.json)。
- 远端根：/root/trifusion-storage/artifacts/rgbnt100_trifusion_source_oof_v1_seed42_20260906。

独立审计服务不可用，未有本实验独立verdict；执行者核验和独立审计分开标注。
执行者曾把工具显示的换行转义误读为格式问题，AST实际码点10，原四核验器没有该错误，
源码未改、未使用转换脚本，见verifier_output_format_check证据。

## 已满足的下一步条件

按同一原合同和配置，三fold重新构造相同初始化SHA的角色，不加载M0训练权重，
各完整20epoch后一次完整图库/五输出比较，43375 query-output/50身份，
原五科学门不变。容量步实测约1.208–1.332秒/更新，
正式总成本仍按约5190实际采样更新+全图库检索估计105–125分钟，日志决定实际预算。
当前正式比较尚未启动；没有改变B0、模型/损失或任何已封存科学失败。
