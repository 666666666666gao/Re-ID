# MSVR310 实例覆盖实验跟踪

固定执行104506b72193b6cdbcd237044be97c2410fd7a45；run msvr310_instance_memory_v1_seed42_104506b；合同40f44b0e6c12771c53a283a5b65ca6a45556e0293b7d48decf5fd518ad771a3d。完整工程/CPU/本地重算PASS，科学Q1_FAIL，配对与相对Signal均0/5。计划与合同没有修改。

| 阶段 | 固定范围 | 终态 |
|---|---|---|
| T0 | 780采样batch/数学 | PASS，21:25:45退出0 |
| M0 | 6×8容量+2×100过拟合，248更新 | PASS，21:33:28退出0 |
| M0 CPU | 全248步/1,236,480距离元素/compact权重 | PASS，21:33:37退出0 |
| M0全文本 | 27文件4,903,487字节SHA及248步重算 | PASS |
| Q1 | 3fold×2端×260步/600query | COMPLETE，22:10:39退出0 |
| Q1 CPU | 全1560步/29,125,376训练距离/2,069,520检索位置 | PASS，22:10:55退出0 |
| Q1全文本 | 59文件71,974,384字节SHA、全部训练/600query/60ID重算 | PASS |
| 科学配对/Signal条件 | 原两组五项门 | Q1_FAIL，0/5与0/5 |

fused52.12111331→51.90495689，配对−0.21615642pp，fold−1.568153/+0.298665/+0.752976，身份bootstrap下界−1.16083775。C/T/M分别+0.57593551/+0.22432572/−0.27768033；相对Signal fused−1.22442367。

全部203/203梯度、0overflow、frozen state、exact Signal、compact reload及780配对像素/采样通过。实际新增困难实例和参数梯度存在；候选训练Triplet并未比control更低，早中期漂移明显但末期减小，不能把陈旧直接定为唯一原因。报告results/MSVR310_INSTANCE_MEMORY_V1_Q1_2026-09-07.md；完整证据evidence/msvr310_instance_memory_complete_q1_20260907/。

原wrapper178471与Q1/CPU已结束，22:11核实GPU空闲；未重启、未消费official。下一新实验尚未登记；先区分真实当前实例困难与缓存近似，禁止在本次失败结果上扫描参数或修改门槛。长期Goal active/unmet。

22:33完整只读挖掘诊断R4通过：1560步/29,125,376距离/99,840anchor曝光，0模型前向/更新。候选新增hinge的负例份额87.8%–88.3%，负例赢家age1–3占63.5%–66.8%，不是旧缓存误差的因果证明；全部20epoch/CSV核验PASS。3次脚本接口错误及修正原样归档，不改训练。下一训练仍未登记，详见完整MINING_DIAGNOSIS报告。Goal active/unmet。
