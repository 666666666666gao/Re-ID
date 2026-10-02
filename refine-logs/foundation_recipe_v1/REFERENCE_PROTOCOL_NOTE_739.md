# UGG与ProxyTTT：已读源码边界

2026-10-02，只读文献/作者源码提取；不是本机复现或完整实验审计。UGG信息提取由fresh gpt-6-astra/max完成，父执行器复核20份完整源码的SHA256及Git blob。没有运行作者模型或scorer，也没有下载权重、图片或数据。

UGG固定作者提交为`eaf1e8e50d04f34ee3e471440f70d335cc67b2c1`。其MSVR数据入口使用`bounding_box_train/query3/bounding_box_test`，遍历全部图库身份；实际评价排除同身份且同scene，另两集排除同身份且同camera。与固定Signal scorer的活跃排除表达式AST相同。当前MSVR协议中的103个query未出现身份、464条干扰图库记录必须保留。源码规则和当前元数据相符，仍不证明作者历史运行的文件内容或顺序相同。[数据入口](https://github.com/wanxixi11/UGG-ReID/blob/eaf1e8e50d04f34ee3e471440f70d335cc67b2c1/data/datasets/msvr310.py#L67)，[评分代码](https://github.com/wanxixi11/UGG-ReID/blob/eaf1e8e50d04f34ee3e471440f70d335cc67b2c1/utils/metrics.py#L68)。

UGG普通入口采用eval/no_grad，没有读到测试时参数更新循环。最终ori+moe表示为3072维，整体L2归一化后用平方欧氏距离；实际入口未启用re-ranking。GPGR在training分支前调用随机采样，包括eval；可训练系数w零初始化，但未检查训练后checkpoint，因此不预言实际评价波动大小。[推理入口](https://github.com/wanxixi11/UGG-ReID/blob/eaf1e8e50d04f34ee3e471440f70d335cc67b2c1/engine/processor.py#L148)，[图传播](https://github.com/wanxixi11/UGG-ReID/blob/eaf1e8e50d04f34ee3e471440f70d335cc67b2c1/modeling/GPGR_UGMOE/GPGR_UGMoE.py#L537)。

公开树仅有RGBNT201配置，40轮/seed1111；车辆YAML未提交。最佳权重保存代码被注释，周期保存50大于该配置40轮；测试入口加载作者绝对路径，strict=False。需要补足实际配置、权重和完整载入见证，才能进行可核验复现。这不构成对作者论文成绩的否定。[配置](https://github.com/wanxixi11/UGG-ReID/blob/eaf1e8e50d04f34ee3e471440f70d335cc67b2c1/configs/RGBNT201/UGG.yml)，[测试入口](https://github.com/wanxixi11/UGG-ReID/blob/eaf1e8e50d04f34ee3e471440f70d335cc67b2c1/test_net.py#L43)。

ProxyTTT之前的API403接收失败保持历史记录。之后Git元数据确认固定提交`92fb0fa33d74813566e06820e56e8d8f48ca1205`，过滤checkout失败；固定raw接收仅test.py/train.py两份通过blob校验，读取processor时超时。未完成被调用do_test的源码审计。test入口传入optimizer，不能仅凭这个入口断言实际更新行为；完整PESA的测试时更新边界来自已读AAAI论文。没有新的代码复现分数。

访问失败与部分传输均存于私有证据目录，不以不完整源码冒充完整审计。UGG源码目录`C:/Users/gb/.codex_tmp/ugg_protocol_20261002_2045/`，含source_catalog、extraction、access_manifest和verification。ProxyTTT部分目录`C:/Users/gb/.codex_tmp/proxyttt_pinned_raw739_20261002/`，原失败记录仍保留。

本次事实不改动运行中的F1六端/source249或既定50轮选点；完整F1报告和fresh integrity/claim review尚待终态。资源池已获用户授权，两台服务器共八张GPU，下一批独立端使用两台，维持单端batch和协议。
