# 固定已保存特征的相加/直和几何诊断

登记于2026-10-06，所有row正式训练和原fixed-best6/18/24诊断已经闭合。该计划只读CPU数组，不加载checkpoint、不执行NN、不更新参数、不挑选另一个best，也不搜索gain。

问题：相加的余弦内积包含g与c的跨分支交叉项。Signal原SIM源码则有独立训练头，推理拼接global1536与SIM1536。其总收益不能全部归到选择器，也不能称为与当前1536残差接口相同。

输入：六模型原original模式保存的g、c、h及原读出gain，完整query/gallery元数据、original距离、own_global分数、原array SHA。即使五份checkpoint已退役，这些原数组均保留；本诊断没有权重消费者。

一次CPU处理全部六行，原距离计分保持原camera/scene过滤。首先用保存h在CPU归一化和原distance_matrix重建additive，对原四指标要求1e-5点一致，并核h=g+gain*c；任何失败原样记录且停止，不扩大容差或重试挑一次通过。只有通过才计算固定直和Normalize([g;gain*c])及完整逐query/身份比较：直和−additive、直和−同模型global。其3072维仅为几何诊断，不能静默替代当前1536部署接口或与Signal原方法等同。

没有新loss、外部数据、参数、seed或正式训练成绩。直和表现好也仅支持固定向量融合形式值得进一步控制，不证明新增内容稳定互补；若不能优于同模型global，则优先研究内容形成，不按官方分数调整直和权重。+0.5mAP且R1不降只作原项目推进线参考，固定post-selection描述不计新正式端或种子证据。

保存新距离与全部逐query/身份表、source/input/output SHA及原失败/终态。预计CPU60–180秒，首次观察launch+180秒；原handle超时不重启。只2026，CUDA_VISIBLE_DEVICES为空，没有GPU/功率/温度查询；2025不访问。分析运行期间不热同步Git/source。
