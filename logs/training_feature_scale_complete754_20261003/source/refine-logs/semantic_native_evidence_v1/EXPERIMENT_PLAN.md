# EV1：保留语义后的原生细节增量

2026-10-02。前置事实：N1 全六端训练满50轮，五端正式验收通过，MSVR310 low 原重载失败且正式指标为空；单独失败收尾报告已完成并接受 fresh same-family/provisional WARN 审计。原 N1-A 三数据集均失败，N1-B 为201失败、100通过、MSVR不可比较。这些终态不变，不重跑旧训练或修改原验收容差。

## 唯一新假设与两项新条件

原 N1 将 CNN 的 CLIP 语义 value 替换成原生图像细节，同时把语义 key 候选从128插值到512；其失败不能区分语义被移除与细节本身无益。本次检验：保留512候选上的语义 value 后，附加原生细节是否能提供可迁移的检索增量。

- `semantic`：CNN 卷积后的语义网格插值到512位置，同一网格产生 key 和 value；没有原生细节 stem。
- `combined`：同一语义 key、同一512候选，将语义网格与 stride8 原生细节网格逐位置相加，经过同一个 value 投影一次。没有新增融合系数或独立 value 投影。

两条件均保留原 CNN→Transformer→Mamba、原 query、读出、1536维 embedding 和 global 加性融合。图像、增强、视觉训练策略、损失、优化器与训练预算保持 clean-public 配方。新接口不是已验证的语义对应、共享/私有分解或新架构成功。

`combined` 比 `semantic` 增加93,248个真实可训练 stem 参数；不得称这两条件容量相同，不在 semantic 中放置闲置参数伪造容量匹配。前置构造证据必须证明 combined 的全部初始状态与原 N1 high 相同，semantic 的全部状态与其共同部分相同；三个数据集均核对公开视觉152张量、新 camera、共享 backbone/neck/classifier。新增 stem 在独立 RNG 范围内构造，不改变其他初始化及训练数据 RNG。

## 固定训练与验收

三个数据集×两条件，共六个新正式端；seed42，从公开 CLIP 和重新初始化 camera/heads/modules 联合训练。FP32视觉LR5e-6，新参数LR3.5e-4，AdamW wd1e-4，warmup5+cosine50，B64/K8，原PLAIN_V8训练增强，CE smoothing0.1+Triplet margin0.3。没有 N2/N3、遮蔽M3、局部ID、外部文本、SAM、DINO、记忆库、新采样器或系数扫描。

每端跑满50轮；选择 fused mAP 最大的一份 checkpoint，平分选较后轮，全部 CMC 跟随同一 checkpoint。完整 state 保存和严格重载，原1e-5计分一致性门保持；失败保留，不重选 checkpoint、不重试或放宽门。GT、完整query/gallery、camera/时间过滤及干扰身份沿用原作者协议，不 rerank。

所有条件、所有训练轮次均公开；已消费官方基准继续参与逐轮选点，单seed不能证明无偏泛化或稳定性。

## 事前科学门

EV-A（主要配对）：combined−semantic 的三个数据集 mAP>0、R1>=0；201与MSVR分别至少+0.5 mAP。全部条件同时满足才支持本次“保留语义后的细节增量”继续进入下一模块。

EV-B（绝对能力）：combined−已封存同配方旧clean roles 三集 mAP>0、R1>=0，201与MSVR至少+0.5 mAP。旧SUMMARY固定 SHA `784e3938a0203ea043543804c6e2933b43877405d7ca624116d59e59d35538fd`。此差含容量及128→512候选变化，不能当作原生细节的单独因果效应。

两门都通过才晋级。combined−已封存 N1 high 只作恢复语义的历史参照，不重新运行旧 N1 high，也不把跨运行差值宣称为严格同期配对。源失败SUMMARY固定 SHA `819ed74b86864786bfc25fac439a1bc8022cdf473e3ab26f021b46648ef04458`；低MSVR不补指标。

报告包括六端完整轨迹、修复/新增首位错误、身份宏平均AP、训练实际区间与参数/显存。任何失败不由一次小幅正差或工程通过替代；既定Goal仍ACTIVE_UNMET，不承诺+10或SOTA。

## 执行顺序

fresh gpt-6-astra/max experiment-bridge 代码复核，same-family/provisional；原生环境 AST/import；三数据集实际初始化对照；201两项各8真实batch M0；其余端先M0再正式50轮。前置结果必须完整通过后才能登记六端，不抢占其他训练，240秒调度，无自动重试。继承现有无重试队列与完整保存/评价，不修改原238个封存来源文件。

预计约6–9 GPU小时、四卡约3–4小时；以实际每端速度更新观察里程碑。本地只保存源码、文本与报告，模型/图像/距离留远端。N1旧成功报告调用数仍为0；新的报告与目录独立。
