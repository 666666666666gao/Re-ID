# §41.958 保留实际value公共分量：一个固定seed条件

§957两个完整50轮端的global均74.2967，差分c仅39.6920/37.9780；实际T/M读取L1明显，并非全部没有选择。原512维value每图均值独立73.3979，patch偏差能量仅约2.40%/1.67%/3.07%。差分的零行和在固定读取权重下消去该均值。是否应保留其中的身份证据，是待验证的具体问题，不是已经确定的唯一失败原因。

新路径仍让三角色产生原读取权重a，改为读取aV，而非(a−uniform)V。实现利用原差分加回同一张图、同一模态的value均值，等价于绝对加权读取；该均值由原前向产生，不是test数据集均值、标签选择、教师或融合系数搜索。global、value detach、role输入detach及global/role梯度职责均保持。该改变检验原约束是否过度删除内容，不是再次提高attention锐度。

原自由readout仍删除；参数和初始化与已闭合uniform-reference一致，88424906个训练参数/280张量，没有新头或额外模型。gain初值仍0.1，作者raw CE+softmargin Triplet、B64/K8、Adam、AMP、完整50轮、seed42和原数据/相机过滤保持。新增读取的行质量为1，原差分变化的行和仍近0；训练日志中的attention_change继续描述相对uniform的变化。

CPU组件实证：绝对权重直接读V与实现最大差5.96e-8；每图公共value平移响应最大差4.77e-7；六个Q/K投影及三个输入均获得有限非零梯度。真实入口factory/schema/build绑定已检查，没有构造完整模型或运行正式训练。以上不替代生产M0，也不证明ReID提升。

本次仅登记一个RGBNT201条件。原完整uniform-reference 74.2974/78.8278作为同参数、同初始化对照，不重跑旧训练。prepare必须实测共享/视觉/camera/head及完整初始state与它相同；真实8步M0必须通过所有280张量活动、有效更新8、BN8、严格reload与role到global梯度隔离，再fresh50及首次strict。任何原失败保留，不重启NN或放宽门。

主要比较新绝对读取−旧uniform-reference、独立global-only及完整Signal。仍需报告四项同mAP-best指标、完整50轮曲线、最佳到末轮下降、首位修复/新增错误、身份AP，以及实际value读取用途。读出/能量或独立分支分数不代替最终净增量。只有完整终态才执行一次保存距离配对报告和必要的固定best诊断。

§957归档退休记录已核验D大小/SHA：两个闭合M0 probe和semantic-reference两个诊断向量缓存，远端重复共798135762B；所有best、官方距离、唯一receipt和原source保留。原消费者需要退休文件时必须先精确恢复。当前free3369496576B，单端预算2GiB+3×384MiB，按实际source/存储guard启动。

执行仅2026物理GPU0/1、一个双卡NN，沿用现有conda，预计prepare/M0约2分钟，完整50轮约42分钟；首M0观察不早于180秒，随后临近实际train起点+42分钟观察。必要复查180–300秒，不提前查epoch，不因本地观察超时重启。

这是项目问题驱动的受控假设，绝对attention/value读取是已有运算，不能独立声称原创主模块。若没有最终增量，否定此假设，不用增加gain/改seed救结果。其他两数据集与整体创新/三集超过Signal目标尚未完成，Goal ACTIVE_UNMET；结果接近前仍不做多seed/调参。
