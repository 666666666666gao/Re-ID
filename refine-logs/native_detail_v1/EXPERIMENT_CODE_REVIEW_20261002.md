# N1 fresh源码复核

gpt-6-astra/max，same-family/provisional，实际回复原文如下；运行时检查尚未执行。

**BLOCKING：无。当前实现符合已登记的 N1 源码合同，源码审查 PASS，无需修改代码。** 这不是 GPU M0 PASS。

核查结果：

- [native_detail_roles.py:24](/C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/native_detail_roles.py:24)：high 直接读取同一批预处理图像；low 先 `avg_pool2d(2)`，两者使用相同的 93,248 参数 stem。两种方向均得到 `[B,3,512,128]` 候选接口。原 role0 跨层投影与 CNN 继续生成 semantic key，native CNN 生成 value，没有断开原 key 路径。role1/2、桥接、static token 和 readout 保持原逻辑。
- [native_detail_roles.py:77](/C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/native_detail_roles.py:77)：新增构造位于 CPU RNG fork 内，原 roles 状态逐项覆盖后 strict load。现有 Mamba factory 默认 CPU 构造，未发现额外消耗训练 RNG 的路径。
- [prepare_native_detail.py:36](/C:/Users/gb/.trifusion_github_publish_22c3bee/tools/prepare_native_detail.py:36)：每个数据集先恢复原构造器，校验旧初始化 binding，再实际比较原模型与 N1 全部非 stem 状态、高低两端全部状态，以及新增参数量。
- [run_native_detail.py:55](/C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_native_detail.py:55) 和 [queue_native_detail.py:59](/C:/Users/gb/.trifusion_github_publish_22c3bee/tools/queue_native_detail.py:59)：模块全局替换闭合。训练、checkpoint condition、strict reload、M0 验收、worker 命令及 expected binding 均使用 N1 合同；未发现递归替换、旧条件残留或 worker 回到旧入口的问题。
- 复用 loop 保存完整 `state_dict`，包含 stem；正式训练重新构造，不加载 M0 权重。八批全部可训练张量累计非零梯度、视觉/camera 更新、完整 50 轮、同一 mAP-best 权重对应全部 CMC、真实 GT 和原过滤链均保留。
- 实际启动链采用 subprocess 参数列表，未发现 shell 命令拼接漏洞。240 秒调度、失败后停止新增启动及无自动 retry 的行为保持一致。
- 四个新 Python 文件 AST 解析通过；检查的八个复用核心文件与 COMPLETE721 封存源码 AST 一致。

**NON-BLOCKING：证据边界。**

尚无 N1 实际构造、梯度数值、显存或 reload 结果。源码能确认梯度路径存在，不能证明真实 AMP 下八批梯度门已经通过。初始化见证目前也是待执行检查。完整 CPU 配对报告和 fresh 结果审计仍待六端完成，队列没有将工程完成冒充科学通过。

下一步按现有计划执行实际三数据集初始化比较、两端 RGBNT201 M0，再由队列完成其余四端 M0 和六端 full50；保留原门槛即可，不需要新增防御逻辑。

审查为 **gpt-6-astra/max、same-family/provisional**，不构成 cross-family acceptance。既有 J1 FAIL 保留；N1 仅是研究推进，三数据集 baseline/SOTA 总目标仍未完成。
