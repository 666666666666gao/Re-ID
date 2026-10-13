# §954：第一端正式弱增量后的诊断补充

第一semantic_reference已完整50/首次strict，mAP74.3111/R1 78.9474，较独立global仅+0.0145mAP/R1不变。uniform_reference尚在原队列训练，不能说两端均无效。496训练source仍冻结，无追加训练或热改。

终态CPU报告从原training_steps记录按epoch汇总真实读取变化、实际region/c范数、修正/G比和两项task loss，保留原完整检索曲线。固定best诊断再比较六项原Q/K128×128权重的初始化/选中best范数及变化；这是实际参数使用描述，不将变化小直接追认成weight decay根因。

同时在原单次编码中用read-only末层attention hook测量实际原512维value的每图patch均值与patch偏差能量，保存image-mean1536向量及完整合法检索。主模型仍照旧使用原value，统计不改变其前向、参数或任何test统计。新增读取不再次编码图像。

理由是实际算法逐行attention差值和近0：固定Q/K条件下，value的整图公共加项会被差分读取抵消。若原value的独立身份能力主要在每图公共成分，不能因其独立mAP72.8就推定它会成为好的增量。现在只提出这个可测问题，不预报它已经解释当前失败，不用dataset均值去改部署，也不假定改QQ必有收益。

当前§953诊断与报告源码已随新增统计重新完成warm CUDA-hidden import，当前SHA以956 CPU proof为准；先前953/954证明保留为各自版本。没有真实诊断/report运行。更新只触及独立终态工具，原496 source前后均核一致。预计两端缓存约200MiB，原256MiB预算仍保留；实际输出需记录大小SHA。

只有两端正式终态+原supervisor EXIT0+一次CPU完整配对报告后才允许原各best一次NN诊断。相关实现为root审查，非独立审计；per-role/mean-value独立成绩不是新正式fused成绩，也不是算子必要性的重训消融。

外领域边界：[ClearCLIP原文§3](https://arxiv.org/html/2407.12442v1)以分割为对象经验选择QQ，其特征熵跨空间与通道，不能直接替代注意力权重熵。当前Re-ID实测QK/QQ不同，故借分解思想而不照搬其选择或噪声归因。PDF网页读取因15MB长度失败，本地HTML缓存另因SSL失败；通过arXiv v1 HTML已核读方法段，不宣称PDF已读或完整附录逐表验算。
