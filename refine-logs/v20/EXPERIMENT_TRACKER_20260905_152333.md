# V20 Experiment Tracker

更新时间：2026-09-05T14:54:11.240053+08:00。状态：M0_PASS_Q1_RUNNING。

| ID | 内容 | 状态 | 证据/依赖 |
|---|---|---|---|
| V20-T0 | 三项远端CUDA数学单测 | DONE_PASS | evidence/trifusion_v20_t0_20260905.json |
| V20-M0 | 六模型配对及新损失梯度、两端容量、100步过拟合 | DONE_PASS | evidence/trifusion_v20_m0_seed42_3cea5bf.json |
| V20-M0-AUDIT | M0阶段独立完整性审计 | DONE_WARN_ENGINEERING_PASS | .aris/traces/experiment-audit/2026-09-05_run06 |
| V20-Q1 | 三折两端20epoch完整OOF及五路全量评价 | RUNNING | 原screen v20_cross_modal_3cea5bf |
| V20-AUDIT | 全量终态证据复算及独立审计 | TODO | 完整终态 |
| V20-D1 | 条件性141-fit refit与一次30-dev | LOCKED | Q1全部科学门通过后另行固定合同 |

source3cea5bfc17e214b1829c020527699d939efa221d；14:48:31CST启动，M0在228.975805
秒内完成，14:52:27观测Q1已进入新建第一端。只允许原进程继续完整20epoch。

两端各98,800,141总参数、7,841,292可训练参数、203训练tensor，推理新增参数0。
六模型配对初始state/五路输出/增强/绑定一致。每专家新损失单独梯度存在；
三折CNN/T/M非零encoder张量数42/54/93、42/54/92、42/54/91。
两端8步峰值reserved6062MiB，203/203完整梯度覆盖、overflow0、冻结state不变。
100步总损失1.8860781193→1.1006586552；基础ID/Triplet0.6110473871→
0.5803135037，新损失5.1001229286→2.0813806057；解析总下界1.0982433065，
超额损失比0.0030658060<=0.1。M0无heldout检索结论，不代表Q1科学通过。

M0不可变snapshot340010字节，SHA5fd4922a7a7036f6905c54397809faed18387666b1df18aa39e5429cd10876a0，
本地与远端相同；保留原RUNNING状态与空fold数组。当前无完整Q1检索结果，
dev/official/D1访问0。Q1初估60–80分钟，等待第一端完整epoch时长修正ETA。
旧V19 Q1_FAIL封存不变。无扫描、消融、新种子或本地模型执行。


M0独立审计及字面字段/记录范围校对已完成：EXPERIMENT_AUDIT_V20_M0.md/json，
工程PASS，overall/integrity WARN，scientific_qualification not_evaluated。
熵下界与比例独立复算差0，损失分量最大舍入差5.96e-8。两轮原文及完整
request/response/meta保存于trace run06。最终MD SHA
28964c3a11d900db7300671fdd07772c4217710a751daef078c4df15e709de8a，
JSON SHAb3e717f453ee813ea45b4b1abdb6004ae29337b1a140a6b45c974b54ba06892a。
字面键名、3126源码断言/48正例记录推导、preflight无optimizer状态保存、
容量8不同batch/过拟合100固定batch等描述已精确区分，数值/门槛不变。
本地criterion.py及protocol只在LF规范化后匹配SHA，远端大权重/图像独立持有
范围有限；M0审计不覆盖后续Q1终态。不作无关换行符重构或运行中源码改动。


15:16:12CST确认第一折两端各完成20epoch/580步，共1160步；strict reload完整
state SHA、只读评价、overflow0、冻结state、203训练tensor梯度覆盖及完整
采样序列/前8增强/初始state/baseline输出配对检查均通过。保留1000gallery/
190合法query。完整snapshot889149字节，SHA
1f71ee488494019937a2f8a9d76b7ec29a611ac10f4037ae6ac5ff8f42a0eb0c，
本地与远端一致；见evidence/trifusion_v20_first_paired_fold_20260905.json。

第一折fused相对实际对照-1.087608 mAP；固定“各折均非负”条件已有失败项。
CNN+0.418478、Transformer-4.332104、Mamba-0.534771。整体仍RUNNING，
按合同完成剩余4/6端，不能删分支/改温度权重/缩预算；所有五路Rank1/5/10、
逐query AP/rank及源/终态权重绑定完整保留。无D1/dev/official访问。
预计六端15:56–16:06完成，下一观测窗口15:33–15:34临近第二折配对完成。
