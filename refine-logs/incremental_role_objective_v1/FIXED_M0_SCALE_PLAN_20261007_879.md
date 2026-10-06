# 固定M0权重单批反向尺度检查

原首端md_batch_ratio201训练八步production M0_PASS、281/281总loss张量活动、BN8/reload0；新增未缩放的孤立梯度门FAIL，原controller2208860/supervisor2208857 exit1，五条件未运行。新代码和正式队列SOURCE_ONLY PASS均不改变此事实。

当前明确的实现区别：生产foundation.train使用GradScaler init_scale256，孤立检查则直接autograd.grad未缩放loss。记录不能区分未使用与数值零，也不能证明Transformer/Mamba在真实scaled训练中没有收到新增梯度。先检查量测口径，不改目标/参数/门槛。

仅使用原保留M0 probe及原initializer，严格加载，一份实际首来源B64批次，一次train-mode AMP forward。同一图分别对loss×1和loss×256取g/c/六QK VJP，报告除以scale后的norm/max以及unused标志。无optimizer、无参数更新、无正式评价或检索数字；BN等buffer暂时由实际train forward更新，随后恢复并校验完整model state与前值精确相等。

只26物理GPU0/1、max1；旧393来源不改，checkpoint/initializer/receipt/sourceSHA在启动前固定。明确固定已训练八步的状态，不能把新单批结果追认为旧八步全通过。无loss rescale、学习率/gain/seed/margin修改、无重复旧训练。若存在尺度依赖，下一版M0诊断应与实际AMP反传一致；若仍无活动，目标资格失败封存。正式六端仍不可启动。
