# 已封存SIM选择语义与RGBNT201成本解释

源码的两类mask取并集，再将未选patch乘为零；没有删除token，也没有给MHA传key_padding_mask/attn_mask。因此固定top-k不是最终保留位置数，也不是attention实际序列长度。零位置仍进入softmax归一化：即使投影bias为零也占概率质量；训练后还可能有共同的K/V bias输出。这是忠实作者实现的结构性质，不是这次指标下降已定位的根因。

当前没有保存真实mask覆盖率，不能声称RGBNT100实际全选或选择已失效。当前all-patch控制保留同一交互器和头，仅用原patch替换token_selection的输出。masked与global比较另外混入交互器、额外训练头和1536→3072部署宽度。

在已经闭合的RGBNT201上，masked/all-patch同容量，训练CLI比值与loss-loop比值见FACTS；mAP/R1跟随各自单一mAP-best。时长来自顺序单次共享服务器运行，不是多次孤立benchmark；阶段还需M0、初始化、firststrict及历史失败成本。所有三个数据集完成后仍只执行原登记全query报告，不据此改top-k或另起训练。
