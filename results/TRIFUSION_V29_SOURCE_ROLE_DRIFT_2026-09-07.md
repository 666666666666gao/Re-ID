# V29完整来源角色漂移与真实身份关系复核

结论：原角色坐标变化存在，联合修正也普遍接近角度边界；但这次完整来源数据没有证明完整分支或融合的可靠排序被普遍破坏。更明确的现象是平均身份间隔普遍缩小，其中约80–81%的融合间隔净缩小来自同前向joint贡献。不能把向量旋转直接解释为身份能力丢失，也不能直接宣布保留loss已经必要。

V29既有Q1仍FAIL0/5，配对fused只提高0.219534mAP。本诊断没有新训练、AP/CMC、dev或official评估。

## 完整执行和证据

- 实际诊断提交1d52c1e；1680批/10080固定前向，三fold580/560/540批全部完成，source每fold94身份、2126/2075/2051记录。
- 原数学159455、GPU159463、CPU163560、report163862、wrapper159453及queue160653全部正常退出且已确认结束。GPU16:03:55结束，CPU16:14:47通过，报告16:16:23完成。
- 全部743178240个相似度值、7741440个向量观测经完整CPU核验；identity42147840/cross-camera3401664个关系曝光，共45549504供各状态/view重复比较，不是独立样本数量。
- 完整12关系格、564身份关系格、648向量格、20304身份向量格均保存；42份文本48248745B接收并核对全部SHA/CSV行数。六最终权重与原数组再次远端SHA核验，图像/模型/NPY没有传回本地。
- 首次SFTP连接被重置（10054），已保留失败记录；重新核对29份已收文件，使用32请求预取接收剩余13份，全部核对通过。仅传输续接，没有重跑模型或CPU分析。
- CPU最大向量GPU/NumPy代数核对误差2.7046498019e-07。固定模型状态不变、grad None、采样/曝光和原两端前8批像素回执匹配。执行器核验不是外部独立审计。

## 来源排序仍然接近饱和

以下是全部active+inactive、完整来源批内关系；original含训练增强，registered_style追加注册V27计划。baseline在两个条件及三个模型中完全相同。

|协议|输入|输出|control非正关系|candidate非正关系|可靠关系新增排序错误|可靠关系新增间隔违规|平均间隔变化|
|---|---|---|---:|---:|---:|---:|---:|
|identity|original|fused|0|0|0|4293|-0.037126166|
|identity|original|cnn|0|0|0|2245|-0.004762028|
|identity|original|transformer|0|0|0|2183|-0.014144747|
|identity|original|mamba|3|1|0|2634|-0.003227111|
|identity|original|pure_bank|0|0|0|523|-0.014755924|
|identity|original|pure_cnn|0|1|0|1707|-0.009524056|
|identity|original|pure_transformer|0|0|0|1068|-0.028289496|
|identity|original|pure_mamba|6|5|0|2204|-0.006454222|
|identity|registered_style|fused|1|1|0|8031|-0.036155944|
|identity|registered_style|cnn|3|2|0|3963|-0.004524566|
|identity|registered_style|transformer|3|3|0|5032|-0.013909779|
|identity|registered_style|mamba|8|5|0|5069|-0.003178873|
|identity|registered_style|pure_bank|9|9|0|3042|-0.014408811|
|identity|registered_style|pure_cnn|26|22|0|5908|-0.009049131|
|identity|registered_style|pure_transformer|19|10|0|7941|-0.027819558|
|identity|registered_style|pure_mamba|44|40|0|7873|-0.006357745|
|cross_camera|original|fused|0|0|0|963|-0.035438480|
|cross_camera|original|cnn|0|0|0|536|-0.004452697|
|cross_camera|original|transformer|0|0|0|479|-0.011888830|
|cross_camera|original|mamba|1|0|0|659|-0.003440242|
|cross_camera|original|pure_bank|0|0|0|124|-0.013187846|
|cross_camera|original|pure_cnn|0|0|0|325|-0.008905396|
|cross_camera|original|pure_transformer|0|0|0|212|-0.023777660|
|cross_camera|original|pure_mamba|0|0|0|484|-0.006880484|
|cross_camera|registered_style|fused|0|0|0|1863|-0.034472056|
|cross_camera|registered_style|cnn|0|0|0|891|-0.004250877|
|cross_camera|registered_style|transformer|0|0|0|1099|-0.011776883|
|cross_camera|registered_style|mamba|1|0|0|1262|-0.003379948|
|cross_camera|registered_style|pure_bank|0|0|0|605|-0.012938470|
|cross_camera|registered_style|pure_cnn|1|0|0|1051|-0.008501753|
|cross_camera|registered_style|pure_transformer|3|0|0|1399|-0.023553765|
|cross_camera|registered_style|pure_mamba|4|3|0|1579|-0.006759894|

原输入fused正确关系没有丢失，注册扰动下的1次非正关系也未新增或修复。完整分支、纯角色/银行在可靠集合上均没有新增排序错误。九个单模态槽位存在少量可靠排序损害，全部18输出见完整表，不能把它们等同于整角色身份能力损失。
可靠集合由比较起点在原/扰动输入都满足余弦正间隔及单位欧氏Triplet0.3定义；从可靠变为hinge>0不等于排序已经反转。fused原输入4293/42121471约0.01019%，扰动8031/42121471约0.01907%可靠关系重新违反间隔。

## 坐标变化不等于判别关系破坏

以下按所有fold和模态的实际观测数加权求余弦均值；分位数仍保留各fold/modal原表，没有平均分位数冒充总体分位数。每个角色、每个输入和比较均有322560槽位观测。

|比较|输入|CNN均值|Transformer均值|Mamba均值|
|---|---|---:|---:|---:|
|initial_to_control|original|0.846338217|0.750147875|0.868878251|
|initial_to_control|registered_style|0.840925064|0.744667054|0.862729849|
|initial_to_bounded|original|0.842683504|0.744224066|0.865165377|
|initial_to_bounded|registered_style|0.836865645|0.738348082|0.858469673|
|control_to_bounded|original|0.926475227|0.873853832|0.952974415|
|control_to_bounded|registered_style|0.926204424|0.873529607|0.952829971|
|bounded_h_to_y|original|0.894431634|0.894431618|0.894439346|
|bounded_h_to_y|registered_style|0.894431560|0.894431619|0.894437908|

同前向h到y各fold/modal中位数范围约0.894427–0.894444，证明近边界更新在全来源分布中很普遍。Transformer的控制到候选坐标变化最大，但其Q1平均mAP仍略升；这再次说明不能只用夹角解释检索退化。初始化到普通终点也发生大量坐标变化，不能将全部变化归因于joint。

## 间隔缩小是普遍现象，但仍需解释它的作用

全部282个fold-source-identity组合在原输入和扰动输入下的fused平均间隔均缩小；cross-camera有合法关系的42个fold-source-identity组合也全部缩小。它们不是282/42个独立身份：不同fold来源身份有重叠。其余240个cross-camera零关系行完整保留。
原输入246/282组合出现可靠间隔违规，扰动264/282；来源最大的两个组合只贡献5.17%/3.93%的违规，因此这一现象不由两个身份支撑。完整60912身份关系行已读取，全部明细保留。

按现有实际相似度分解，Signal不变，fused变化等于原角色h-bank变化的一半加上同前向joint贡献。

|协议|输入|总平均间隔变化|原角色部分|joint部分|joint占净缩小|
|---|---|---:|---:|---:|---:|
|identity|original|-0.037126166|-0.007377962|-0.029748216|80.127%|
|identity|registered_style|-0.036155944|-0.007204405|-0.028951550|80.074%|
|cross_camera|original|-0.035438480|-0.006593923|-0.028844568|81.393%|
|cross_camera|registered_style|-0.034472056|-0.006469235|-0.028002832|81.233%|

这只是固定模型输出的精确标量分解，没有构造新检索头、计算反事实mAP或扫描融合权重；占比不是Q1性能下降的因果占比。来源已有很大正间隔（原输入control均值约0.6642，candidate约0.6270），缩小后依然能够分对，不能据此推出未知身份必然退化。

## 下一步边界

1. V29失败继续封存，不扫描b、seed或checkpoint。可靠关系保留的损失尚未新增，也不因夹角变化自动启动DELTA/PCGrad/教师。
2. 优先从已经保存的完整来源相似度中检查joint新增变化是否包含新的身份判别关系，或主要表现为相似度的共同偏移/压缩。后者目前只是待验证解释，不能写成已证实；无需新GPU前向或官方数据。
3. 依据完整检查再登记一个新的训练假设，并保持匹配对照和全部晋级条件。V27的环境扰动正证据、RGBNT100基线增益继续保留；MSVR310/RGBNT100的后继验证仍属于总目标，不用RGBNT201内部成绩代替它们。
4. 三数据集超过baseline并达到资源/协议注明的当前SOTA尚未实现。用户恢复后的长期Goal已active，完整目标与执行边界见[研究Goal](../docs/TRIFUSION_RESEARCH_GOAL_2026-09-07.md)。

## 完整文件

- [全部18输出、全部分层和身份CSV](../evidence/v29_source_drift_complete_20260907/complete_scalar_report/README.md)
- [完整CPU验证](../evidence/v29_source_drift_complete_20260907/complete_source_drift_verification.json)
- [本地全量文件与身份复核](../evidence/v29_source_drift_complete_20260907/local_full_text_identity_digest.json)
- [完整间隔变化分解](../evidence/v29_source_drift_complete_20260907/complete_margin_change_decomposition.json)
- [接收证明及全部文件SHA](../evidence/v29_source_drift_complete_20260907/intake.json)
