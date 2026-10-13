# 新入口的实际配置链核查

2026-10-13，固定父提交e1983eabe877effd8afeabe89a48980ff72d7173。使用code-review-excellence技能检查§936新增wrapper/driver的具体实现与COMPONENT_PLAN，不以常见风格偏好重构封存来源。

本次只核查入口导入与配置函数的实际绑定，未获新的训练范围回复。已有合成head检查不重跑，训练仍停止；不调用模型builder、数据loader、loss、autograd、optimizer、prepare、M0、train或evaluate。

实际链为normalized分类→global任务→读取detach→native research→模型分段→independent native。生产inner.main在外部入口configure后再次调用inner.configure，这是当前代码明确存在的配置边界；需要检查不会覆盖候选wrapper、原global/role loss、raw metric policy和dataset限定。没有证据支持加入fallback或新兼容层。

检查按既定warm conda执行一次CPU-only模块导入，CUDA_VISIBLE_DEVICES为空：

1. 六个被捕获的original_build_core函数对象各不相同，模块次序与上述源链一致。只检查对象，不执行函数。
2. outer configure后检查inner/foundation的build_core、loss_values、condition、AuthorHeadEvidence、DetachedSemantic初始化类、loader/optimizer/train入口及schema绑定。
3. 调用实际inner.main将调用的inner.configure，再检查同一组绑定保持。
4. 原474项运行来源、§936四项新增源码/计划及当前Windows本地来源在执行前后保持字节；Git其他修改保持。
5. 成功只称PASS_ENTRY_BINDINGS_ONLY；它不覆盖真实模型构造、公共CLIP初始化、全参数optimizer所有权、8步M0、首次重载、50轮、数据及科学有效性。

源码审查同时检查：201 semantic限定；f=L2(h)先进入detached BN/classifier，heads的metric部分仍raw h；global task保持；无新增注册参数；推理仍是原fused；单次forward_features；根源head parameter detach/buffer clone。这些字段与936组件证据分开注明，不把模块导入冒充生产资格。

新发现应按真实检查失败留证并修复具体实现，不原样重复整个检查碰运气；不改变科研推进线、缩减合法图库或按测试数值调参。scope确认未到前不继续任何依赖训练工作。Goal仍ACTIVE_UNMET。
