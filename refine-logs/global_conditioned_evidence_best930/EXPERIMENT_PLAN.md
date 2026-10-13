# 原梯度／随机best的同模型证据分解

2026-10-13，源草案；尚未执行。本轮主机制结果已在§929封存，不重新训练或改变其失败判定。

## 问题

gradient−random已为−0.105086 mAP/−0.478470 R1，不能通过层号、排除比例、gain、margin或seed继续救分。原两端训练teacher标量一致，但目前只保存fused距离，尚没有各自global/correction/fused分解。`independent_upstream_metrics_equal`是同一fused排名经独立评价函数复核，不是自己的global分数。

这项补测仅回答：两份已选E8 best的global是否真的保留了独立global能力，角色修正给自己global增加还是破坏了哪些合法身份关系？它不研究新比较器、区域MaxSim或新模块，不重放旧§918/919五分数队列。

## 固定合同

- 只使用原两端mAP-best，不改选轮、训练、初始化、分类头、部署维度、推理mask、数据或评价过滤。
- 真实原builder及initializer绑定，strict加载；各端query/gallery每个batch一次编码，从同一次forward保留g、c、h、f。两端顺序，两卡0/1同时仅一个job。
- 使用原1536维余弦距离及完整合法gallery；fused重算距离与原first-strict数组最大绝对差及四项指标均≤1e−5。失败保存，不重试挑通过；不得覆盖原官方指标或验收状态。
- 同模型global、correction单独、fused和已保存独立global距离分别记账；同模型global不是独立消融。
- 报告same-model global→fused、independent global→same-model global的完整query AP/首位修复与新增错误/身份宏AP；修正相对范数和角度只是几何描述，不是唯一失分原因。
- 模型参数与buffer执行前后摘要一致，所有grad均None；保存约100MiB的新特征和诊断距离，供后续CPU分析，不产生任何新权重。
- 分开记录含模型构造/文件校验的总时间和含加载/CPU传输的query-gallery提取时间，不写成隔离kernel基准。

## 来源与执行边界

来源469及15个M0输入维持full926原pin；追加一个只读诊断工具和本轮controller的单独源码pin。绑定原controller EXIT0、两个first-strict receipt/checkpoint/training/initializer/protocol/public CLIP及既有独立global距离。只复用warm tri_reid，不安装、改变环境或读取功率温度；不用25或GPU2/3，不抢占。

源复核通过后才封存具体INPUT_SEAL并执行一次。预计两端共2–4分钟，按原first-strict各约40秒并加文件验证/输出余量估计；等待180–300秒，未到节点不查询日志。原两端训练已终止，本次仅有限固定best诊断；不是新训练队列。原失败和所有best/必要依赖均保留。

## 决策

若global保留而fused增量仍薄，进一步收窄到证据的实际关系用途，不能继续把修正能量或局部独立能力当净增量。若global不同，要先分离底座差距与修正效果；此处不据官方分数修改模型或搜索参数。无论结果怎样，梯度相对随机的原失败不改判；全三集超过Signal仍未实现，Goal ACTIVE_UNMET。
