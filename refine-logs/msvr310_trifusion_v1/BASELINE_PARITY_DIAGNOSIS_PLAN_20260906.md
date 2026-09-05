# MSVR310 第0折Signal特征一致性停止后的固定只读诊断

登记时间：2026-09-06T06:37:21.609289+08:00。状态PREPARED_NOT_RUN。
原比较wrapper70422 exit1，第0折20epoch/260更新已完成；严格重载后提取360条gallery，
在比较baseline与原B0保存features时断言失败，尚未计算AP/Rank、未开始fold1/2。
原checkpoint、summary中的RUNNING字段、日志/exit和全部training/receipt原字节保留；不补写不存在的检索终态。

核对已经被原失败提取访问的fold0全部360个gallery记录，使用原clean loader和相同的五个B64及一个B40尾批。
四条固定路径：原独立Signal；载入原第20epoch角色checkpoint后的独立Signal；
hierarchical baseline；完整三角色模型的baseline输出。
每条都与原B0保存的相同360条features比较，并比较同进程内四路径。
只报告逐元素一致性、最大/均值差、direct/SIM分段差、相对L2与余弦数值差；不计算检索排序、AP或Rank。
记录backend flags前后值，但不更改或扫描任何backend flag、dtype、batch、尺寸、seed或模型设置。
模型/Signal state必须原样；单独保存诊断特征文件于远端，0optimizer、0backward、0checkpoint写入。
共1440次记录前向（720独立Signal、360hierarchical、360full role），不同原记录360；所有图像与张量操作仅远端。
诊断结果只确定实现/计算差异，不能预先解释为算法失效或以数值很小为由放宽原逐元素一致性门。
后续行动依据实际结果单独登记；不重训已完成第0折、不追改原M0/比较门或篡改原结果。

执行前范围修订：原首64条诊断方案从未执行，已另存FIRST64_NOT_RUN。实际失败比较的是360行矩阵；
原loader明确定义五个64批及40条尾批，所以固定覆盖全部原行以区分整批/尾批现象，不依据任何未读差异挑选样本。
