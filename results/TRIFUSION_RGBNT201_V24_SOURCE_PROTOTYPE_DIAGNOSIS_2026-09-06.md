# V24 完整 source 原型与样本几何只读诊断

记录时间：2026-09-06T03:34:48.367820+08:00。状态 ALL_NINE_SOURCE_MODELS_DIAGNOSED_READ_ONLY，原PID59375 exit0。
本地完整数组与日志核验通过；独立 source 诊断审阅 PENDING。V24 Q1 已单独封存为 Q1_FAIL，
其训练终态与独立审计不由本诊断替代。

本次观察支持“干净 source 上的原型分类及身份间隔已经充分拟合”。
初始化、对照终点和候选终点的九个模型，global/environment 原型分类均100%；所有非自身
同身份正例的最小余弦仍高于最近异身份负例。这个结果不是未知身份泛化证据。

## 固定执行范围

- 代码4452a9801c69bc0bfbdbe012ff9311e4afc04f46；Q1源码6a4ac2cd95af2ca1a9122d1f79aabd3a83e4fe33。
- 三折各94个source身份、108个身份/相机成员，记录数2126/2075/2051。
- 每折共同初始化一次、对照最终模型一次、候选最终模型一次；共9模型/18756条source记录/153个batch前向。
- 真正跨相机source身份每折14个，相应query381/392/369；全部其余source记录保留。
- 仅现有确定性clean变换、FP32、7680D fused；无 autocast。
- 优化0、反传0、checkpoint写入0、heldout/dev/official图像前向0；无跨fold距离。
- 每次模型状态前后SHA不变；原始初始memory被精确重建。耗时312.011035秒。
- 本地只核对JSON与标签/索引/数值，不加载模型、特征张量、checkpoint或图像。

## 九个模型的完整结果

下表CE是global/environment两个未平滑交叉熵的均值；“缓存”为V24保存的memory，
“新原型”为当前clean source特征按原相机等权公式重新求均值。
最后一列是每条source记录的“最远非自身正例余弦减最近负例余弦”的全局最小值。

| fold | 模型 | source记录 | 缓存 CE | 新原型 CE | global/env正确率 | 最小正负余弦间隔 |
|---|---|---:|---:|---:|---:|---:|
| 0 | initial | 2126 | 0.000456865 | 0.000456865 | 100 / 100 | 0.105179071 |
| 0 | ordinary_two_view | 2126 | 0.000381496 | 0.000279468 | 100 / 100 | 0.139325976 |
| 0 | environment_identity_prototype | 2126 | 0.000337099 | 0.000246343 | 100 / 100 | 0.148987055 |
| 1 | initial | 2075 | 0.000513873 | 0.000513873 | 100 / 100 | 0.075873971 |
| 1 | ordinary_two_view | 2075 | 0.000488707 | 0.000357764 | 100 / 100 | 0.098958969 |
| 1 | environment_identity_prototype | 2075 | 0.000415883 | 0.000304135 | 100 / 100 | 0.106595457 |
| 2 | initial | 2051 | 0.000368214 | 0.000368214 | 100 / 100 | 0.115347624 |
| 2 | ordinary_two_view | 2051 | 0.000340796 | 0.000247233 | 100 / 100 | 0.137706935 |
| 2 | environment_identity_prototype | 2051 | 0.000316143 | 0.000229235 | 100 / 100 | 0.154448211 |

所有同身份正样本集合均排除了query自身；这不是通过self-match得到的正负间隔。
另外单独核对了真实跨相机正例子集。这里只报告几何诊断，没有执行或声称一项新的source检索mAP评估。
global分类均包含全部94个真实身份；environment候选身份/相机成员数依实际相机变化，
三折分别最少2/3/3、最多72/71/69，不能把environment的100%当成统一94类任务的难度。

## 样本负例与身份均值原型的差别

| fold | 模型 | 样本最近负例减新原型最近负例余弦均值 | 缓存/当前clean原型平均余弦 |
|---|---|---:|---:|
| 0 | initial | 0.032806725 | 1.000000029 |
| 0 | ordinary_two_view | 0.033116064 | 0.980625602 |
| 0 | environment_identity_prototype | 0.034141293 | 0.980414765 |
| 1 | initial | 0.031819968 | 1.000000035 |
| 1 | ordinary_two_view | 0.032229789 | 0.981822311 |
| 1 | environment_identity_prototype | 0.033088632 | 0.981561862 |
| 2 | initial | 0.031350697 | 1.000000041 |
| 2 | ordinary_two_view | 0.032075797 | 0.979374946 |
| 2 | environment_identity_prototype | 0.032604688 | 0.979281531 |

样本最近负例通常比均值原型更接近query，说明均值表示弱化了样本级竞争；
但这九个模型的所有clean source样本仍满足严格正负分离，因此当前没有直接证据证明
只扩大同一批干净source的负例池就能解除泛化瓶颈。此判断是执行者依据上述观测作出的研究推断，
不是对XBM、困难负例学习或其他训练阶段的普遍否定。

最终缓存与clean均值的平均余弦约0.979–0.982；缓存来自弱增强、当前均值来自clean视图，
所以差异包含视图分布及模型变化，不能全部归为陈旧缓存，更不能据此扫描EMA/温度/刷新周期。
初始化平均余弦略大于1是浮点舍入；原初始memory状态SHA精确相同，没有对余弦作截断美化。

## 解释边界与下一步约束

新原型包含被评分source样本自身，prototype分类是in-sample；距离正例则排除了self。
source身份本来就参与过合法初始化和训练，这项100%不能替代heldout/dev/official。
clean诊断没有重放强增强训练分布，不能说强增强目标“完全没有梯度”或没有困难例。
全Q1记录中的强视图原型损失与本次clean损失应分别表述。

V24对照和候选保持相同双视图、原目标、memory计算和更新，只有原型损失系数不同；
完整Q1 fused增益+0.491307pp但四项门失败，不能用更好的source间隔把它升级。
未来主假设需要解释环境变化下的未知身份区分，并保留相机/模态专有身份细节。
当前不登记或执行新的训练、扫描、消融、D1/dev/official评价。

## 完整可核对证据

- 原始全逐样本/全94身份均值/全108成员数组：evidence/trifusion_v24_source_diagnostic_20260906.json。
- 原始SHA：a66f17a450fb0eca2404fd23721545eed0dd7b061550230a7d51368da24fa271。
- 固定计划：refine-logs/v24/SOURCE_PROTOTYPE_DIAGNOSTIC_PLAN_20260906.md/json。
- 远端执行：tools/diagnose_v24_source_prototypes.py；脚本SHA aedaa9a1366a37ebcf65c3a24a1a8708c67971cff9163ff82f23565ddab148a8。
- 本地数组核验：evidence/trifusion_v24_source_diagnostic_array_verification_20260906.json。
- 完整18756条记录、1692行identity均值核验，均值/分位最大差1.42108547152e-14，
  概率和FP32代数最大差2.98719532443e-08；耗时1.271381秒。
- 日志九模型事件及最终计数完全一致：evidence/trifusion_v24_source_diagnostic_log_verification_20260906.json。
- 远端权重和特征未导出到本机，独立审阅需保留“原文件哈希回执与本地原张量持有”的区别。
