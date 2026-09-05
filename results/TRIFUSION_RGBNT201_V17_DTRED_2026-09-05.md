# V17 DTRED：完整训练终态与全 gallery 补评

V17 已完成预注册的全部三折、两个 endpoint、每个 endpoint 20 epoch；原始 Q1
终态为 FAIL，D1 未执行。不是训练中断或少跑 epoch。本节原始记录均来自实际
运行源码 `535ef2f305668493c0d07095ab17bb66e9997db6`；本次2026-09-05接续核查
时已无训练进程，发现终态尚未进入交接文档，因此补齐证据与报告。

## 当前模型与执行规模

V17 固定 Signal 的完整3072D检索输出和 V8 CNN/Transformer/Mamba 三条专家。
三专家各给出1536D三模态残差；唯一新增 TriadicCorrection 将每支残差投影到
256D，并结合另外两支的逐元素乘积及三支均值，输出三组残差修正。最终融合
为3072D exact Signal前缀与等能量4608D残差银行的拼接，共7680D；各专家输出
为3072+1536=4608D。Signal-only始终可从同一checkpoint输出。没有Router、
reranking或测试时训练。

V17总参数101,823,245，可训练参数3,010,816、22个tensor。M0真实B64/K8的
8步训练22/22梯度有效，overflow0，峰值reserved1808MiB；100步固定批
floor-aware excess ratio=0.000693508，冻结状态不变。三折初态、采样和前8批
增强tensor配对通过。M0耗时152.17秒。

Q1共3360步：fold0每端580，fold1每端560，fold2每端540；六个endpoint全部
20/20 epoch且22/22梯度有效，overflow0。总耗时2113.85秒，峰值reserved5656MiB。
六份最终checkpoint均保留在远端。所有13项integrity receipt为true，dev0/official0。

## 原始预注册Q1结果（限制gallery的资格协议）

| Fold | weight0 fused mAP | DTRED fused mAP | fused Δ | CNN Δ | Transformer Δ | Mamba Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 82.467856 | 81.867644 | -0.600212 | -0.450206 | -0.899701 | +0.255723 |
| 1 | 94.619150 | 94.316678 | -0.302472 | -0.644239 | -0.164442 | +0.048449 |
| 2 | 89.324807 | 89.200165 | -0.124641 | -1.001929 | +1.230013 | -0.842926 |
| 571-query aggregate | 88.702857 | 88.364223 | -0.338635 | -0.706213 | +0.084211 | -0.197918 |

21身份cluster、10000次bootstrap的fused增益95%下界为-1.120318 mAP。
所有科学分项门失败，仅integrity通过；`next_phase_authorized=false`，
`d1_executed=false`。这不是85.3官方测试目标的达成结果，88.364223不能与其混排。

训练中fused负关系violation由对照0.00852508降到0.00534845（降低37.26%），
但正关系violation由0.000282664升到0.000303692（增加7.44%）。该指标为
最后一个epoch在线批次loss均值，不是对同一固定诊断批次的终态因果比较。
它支持“负关系约束响应，但正关系和跨身份检索没有同步改善”的有限观察。
不能据此宣称阈值、loss系数或训练时长是已证实原因。

## 评估范围问题与本次补评

原runner先调用`select_cross_camera_records`，再以该列表同时构造query和gallery。
因此每折只评价7个跨摄像头身份，对应190/179/202查询；其余40个留出身份也从
gallery被移除了。该协议对所有endpoint一致，原有相对负结果保留，但不覆盖
各折全47身份gallery，不能描述成完整留出集检索结果。

本次独立只读脚本`tools/audit_v17_full_gallery.py`重新加载全部六个最终checkpoint，
严格核对整模型final-state SHA以及冻结来源，遍历三折全部留出样本。所有具备
跨摄像头正样本的查询参与mAP/CMC；无合法正样本的记录仅从query分母排除，
仍作为gallery干扰项保留。逐查询AP、first-match rank、文件/身份/camera manifest
进入回执，五路输出统一比较。此补评不是新训练、独立新数据验证或改变原门槛。

首次补评六个endpoint的推理均完成，但结果写出时漏把ClusterBootstrapResult
转换为JSON数字字段，因TypeError没有生成最终JSON。修正提交`0888f454`只修改
结果序列化；保留原失败日志，执行一次相同checkpoint和数据的只读重放。两次
均optimizer0、checkpoint writes0、dev0/official0；训练源和六个checkpoint不变。

## 后续研究边界

V17不得做D1、official、消融、多seed或width/LR/loss/epoch/checkpoint扫描。
原始失败和全gallery补评应一同保留，不能挑一个协议中的最好数值汇报。

下一步先把全gallery缺失干扰项问题变为后续实验固定协议要求，再用保存的逐
查询结果对所有三折/所有分支做错误归因：正样本排序破坏、首个错误匹配和
专家互补是否被correction抹平。需要图像级证据后才选择新的表征改动；目前
没有证据把背景污染、分辨率、骨干容量或学习率宣布为主因。

新主实验必须是新的表征或关系监督结构假设，在完整gallery、全分支、固定
seed42/终点的同协议比较中证明增益。多次使用同一21个跨摄像头身份的结果
只能作为开发诊断，不得称为新独立验证。65 mAP开发门、官方固定训练与评估、
RGBNT100/MSVR310扩展和SOTA验证均未完成，总目标继续有效。

证据：

- `evidence/trifusion_v17_dtred_m0_seed42_535ef2f.json`
- `evidence/trifusion_v17_dtred_q1_seed42_535ef2f.json`
- 远端原始目录：`artifacts/trifusion_v17_dtred_{m0,q1}_seed42_535ef2f/`

## 全 gallery 补评终态（2026-09-05）

补评成功，耗时127.24秒。三折gallery为1000/1051/1075，共3126条，全部141个
留出身份各出现于且只出现于其所属fold；合格query为190/179/202，共571条。
另外2555条没有同身份跨摄像头正样本，因此不进入query指标分母，但全部保留
在gallery中参与排序。

| 输出 | weight0 mAP / R1 | DTRED mAP / R1 | mAP Δ | DTRED R5 / R10 |
|---|---:|---:|---:|---:|
| baseline_only | 77.487603 / 79.334501 | 77.487603 / 79.334501 | +0.000000 | 89.492119 / 93.520140 |
| fused | 80.614939 / 84.063047 | 80.286024 / 83.537653 | -0.328915 | 88.791594 / 92.469352 |
| cnn | 79.952578 / 82.837128 | 78.433784 / 81.436077 | -1.518794 | 88.441331 / 90.542907 |
| transformer | 78.123409 / 82.311734 | 78.853236 / 82.837128 | +0.729826 | 90.192644 / 94.921191 |
| mamba | 78.858043 / 82.311734 | 79.064729 / 81.961471 | +0.206686 | 88.266200 / 91.243433 |

三折fused Δ分别为+0.129220 / -0.906481 / -0.248032 mAP；aggregate=-0.328915，
21身份bootstrap95%下界=-1.124616。加入原先遗漏的2555条干扰样本后，DTRED
fused mAP从88.364223降至80.286024，绝对下降8.078199；原先的完整数据集
措辞必须撤回。两种协议中的相对fused结论均为负，但fold0符号改变，Mamba
aggregate也改变符号，说明gallery范围会影响逐分支归因，不能只保留有利协议。

DTRED比同fold Signal-only高2.798421 mAP，但weight0高3.127336；因此增益不能
归因于DTRED关系包络。CNN受到最大的额外伤害（-1.518794 mAP），全gallery下
Transformer/Mamba aggregate为正仍不足以证明三专家共同受益。

六端全state SHA严格等于原训练final-state SHA，读取后全state与checkpoint SHA
均不变，exact Signal前缀通过。原始Q1科学gate原样保留为false，没有调用D1。
`status=PASS`仅表示本次只读重载/评估/序列化成功。

回执：`evidence/trifusion_v17_full_gallery_fixed_20260905.json`，SHA256
`1f812b7688b8186b87b1e5ec9dd37c137edd0c98928c277ee0fb719dd59cf79a`。


## 验证与错误分布

全gallery逐query fused AP改善/受损/不变=165/184/222，Rank1修复/破坏=2/5；
CNN为138/238/195及6/14；Transformer为211/153/207及6/3；Mamba为190/177/204
及6/8。mAP与Rank1响应不同，也不应只选择提升的一项。

远端最终专项回归13/13通过。M0、原Q1和补评JSON的远端/本地字节SHA逐一相等，
完整回执和两次补评日志SHA记录在`evidence/trifusion_v17_terminal_verification_20260905.json`。
独立审计报告见`EXPERIMENT_AUDIT_V17.md/json`；科学失败与完整性问题分开记录。
