# V29 完整来源角色漂移与关系变化

这是完整固定来源诊断的标量汇总，没有新增训练或检索成绩。V29 Q1_FAIL0/5保持。

实际诊断提交：1d52c1e9ca957737050072e8e2ace427417cbe14。原始前向10080次、1680个批次，全部CPU核验已完成。
identity关系42147840次，cross_camera关系3401664次；它们包含重复曝光，跨模型和输入条件复用，不是独立试验数量。
原输入包含注册训练增强；registered_style额外施加既有V27计划，其中未激活批次两输入完全一致。
每个模型在自身坐标系内计算相似度，只跨fold累加标量。来源身份在不同fold有重叠，不计算独立样本置信区间。

## 候选相对匹配控制的全部18输出

以下合计active与inactive。间隔为正负余弦差；hinge是单位向量欧氏Triplet margin0.3。
可靠集合由比较起点在两输入条件都满足margin>0且hinge=0定义。单纯间隔下降不等于排序错误。

|协议|输入|输出|起点非正|终点非正|破坏正确|修复|可靠关系|可靠正确被破坏|可靠间隔违规|平均间隔变化|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|identity|original|baseline_only|16|16|0|0|41947364|0|0|0|
|identity|original|fused|0|0|0|0|42121471|0|4293|-0.0371261662|
|identity|original|cnn|0|0|0|0|42112440|0|2245|-0.00476202811|
|identity|original|transformer|0|0|0|0|42125768|0|2183|-0.0141447472|
|identity|original|mamba|3|1|0|2|42100220|0|2634|-0.00322711103|
|identity|original|cnn_RGB_residual|35434|35206|4400|4628|40931811|0|61453|-0.0111055746|
|identity|original|cnn_NI_residual|53059|52074|6766|7751|40320696|0|92008|-0.00971531011|
|identity|original|cnn_TI_residual|32730|31632|3391|4489|41186526|0|48246|-0.00775128421|
|identity|original|transformer_RGB_residual|33865|33086|7324|8103|41152267|0|95108|-0.0299634178|
|identity|original|transformer_NI_residual|58039|55852|12271|14458|40549603|2|157972|-0.0344227789|
|identity|original|transformer_TI_residual|27581|26352|6210|7439|41416203|0|81487|-0.0204822905|
|identity|original|mamba_RGB_residual|43117|42134|5161|6144|40917884|0|62478|-0.00613385992|
|identity|original|mamba_NI_residual|62385|60500|7886|9771|40282836|0|94442|-0.00665714881|
|identity|original|mamba_TI_residual|39130|38782|4843|5191|41153156|0|54736|-0.00657165879|
|identity|original|pure_bank|0|0|0|0|42129787|0|523|-0.0147559242|
|identity|original|pure_cnn|0|1|1|0|42114672|0|1707|-0.00952405622|
|identity|original|pure_transformer|0|0|0|0|42127367|0|1068|-0.0282894957|
|identity|original|pure_mamba|6|5|1|2|42099842|0|2204|-0.00645422213|
|identity|registered_style|baseline_only|16|16|0|0|41947364|0|0|0|
|identity|registered_style|fused|1|1|0|0|42121471|0|8031|-0.0361559441|
|identity|registered_style|cnn|3|2|0|1|42112440|0|3963|-0.00452456566|
|identity|registered_style|transformer|3|3|0|0|42125768|0|5032|-0.0139097786|
|identity|registered_style|mamba|8|5|0|3|42100220|0|5069|-0.00317887283|
|identity|registered_style|cnn_RGB_residual|59755|59834|7526|7447|40931811|0|104781|-0.0105098026|
|identity|registered_style|cnn_NI_residual|86915|85600|11174|12489|40320696|0|149857|-0.00923555857|
|identity|registered_style|cnn_TI_residual|44447|42978|4685|6154|41186526|0|70294|-0.00740203238|
|identity|registered_style|transformer_RGB_residual|66296|65026|14799|16069|41152267|7|195275|-0.0292135582|
|identity|registered_style|transformer_NI_residual|106199|105100|24328|25427|40549603|4|302561|-0.0336455306|
|identity|registered_style|transformer_TI_residual|41723|40336|9479|10866|41416203|0|131005|-0.0205995861|
|identity|registered_style|mamba_RGB_residual|72181|71299|8881|9763|40917884|0|108568|-0.00602165575|
|identity|registered_style|mamba_NI_residual|104298|101188|12943|16053|40282836|0|155569|-0.00661465569|
|identity|registered_style|mamba_TI_residual|54909|54691|6881|7099|41153156|0|82021|-0.00643692534|
|identity|registered_style|pure_bank|9|9|1|1|42129787|0|3042|-0.014408811|
|identity|registered_style|pure_cnn|26|22|4|8|42114672|0|5908|-0.00904913137|
|identity|registered_style|pure_transformer|19|10|3|12|42127367|0|7941|-0.0278195581|
|identity|registered_style|pure_mamba|44|40|4|8|42099842|0|7873|-0.00635774522|
|cross_camera|original|baseline_only|7|7|0|0|3352966|0|0|0|
|cross_camera|original|fused|0|0|0|0|3396158|0|963|-0.0354384804|
|cross_camera|original|cnn|0|0|0|0|3394222|0|536|-0.00445269707|
|cross_camera|original|transformer|0|0|0|0|3396940|0|479|-0.0118888302|
|cross_camera|original|mamba|1|0|0|1|3391648|0|659|-0.00344024188|
|cross_camera|original|cnn_RGB_residual|3983|3846|475|612|3251994|0|7170|-0.00941045594|
|cross_camera|original|cnn_NI_residual|7598|7447|1032|1183|3149683|0|11345|-0.00893230714|
|cross_camera|original|cnn_TI_residual|5671|5511|542|702|3274195|0|7005|-0.00837342443|
|cross_camera|original|transformer_RGB_residual|3837|3516|772|1093|3278681|0|11337|-0.0303254258|
|cross_camera|original|transformer_NI_residual|8786|7922|1580|2444|3187949|0|17793|-0.0283119929|
|cross_camera|original|transformer_TI_residual|3431|3328|900|1003|3315435|0|10499|-0.0126955609|
|cross_camera|original|mamba_RGB_residual|5241|5193|777|825|3252973|0|9610|-0.00799180608|
|cross_camera|original|mamba_NI_residual|9283|9065|1332|1550|3135989|0|13329|-0.00690031613|
|cross_camera|original|mamba_TI_residual|6022|5953|745|814|3275729|0|6266|-0.00574932959|
|cross_camera|original|pure_bank|0|0|0|0|3398444|0|124|-0.0131878461|
|cross_camera|original|pure_cnn|0|0|0|0|3396246|0|325|-0.00890539649|
|cross_camera|original|pure_transformer|0|0|0|0|3397689|0|212|-0.0237776599|
|cross_camera|original|pure_mamba|0|0|0|0|3393394|0|484|-0.00688048367|
|cross_camera|registered_style|baseline_only|7|7|0|0|3352966|0|0|0|
|cross_camera|registered_style|fused|0|0|0|0|3396158|0|1863|-0.0344720558|
|cross_camera|registered_style|cnn|0|0|0|0|3394222|0|891|-0.00425087672|
|cross_camera|registered_style|transformer|0|0|0|0|3396940|0|1099|-0.0117768825|
|cross_camera|registered_style|mamba|1|0|0|1|3391648|0|1262|-0.00337994794|
|cross_camera|registered_style|cnn_RGB_residual|6965|6831|858|992|3251994|0|11777|-0.00883310413|
|cross_camera|registered_style|cnn_NI_residual|14321|14367|1946|1900|3149683|0|18448|-0.00845800342|
|cross_camera|registered_style|cnn_TI_residual|6622|6433|693|882|3274195|0|9234|-0.00821415117|
|cross_camera|registered_style|transformer_RGB_residual|8629|8084|1781|2326|3278681|1|23070|-0.02962243|
|cross_camera|registered_style|transformer_NI_residual|17375|16606|3384|4153|3187949|0|35084|-0.0280589654|
|cross_camera|registered_style|transformer_TI_residual|4658|4569|1230|1319|3315435|0|14743|-0.0129798995|
|cross_camera|registered_style|mamba_RGB_residual|8783|8969|1366|1180|3252973|0|15122|-0.00791462542|
|cross_camera|registered_style|mamba_NI_residual|17646|17299|2352|2699|3135989|0|22076|-0.00695830715|
|cross_camera|registered_style|mamba_TI_residual|7618|7540|938|1016|3275729|0|8660|-0.00540675014|
|cross_camera|registered_style|pure_bank|0|0|0|0|3398444|0|605|-0.0129384701|
|cross_camera|registered_style|pure_cnn|1|0|0|1|3396246|0|1051|-0.00850175282|
|cross_camera|registered_style|pure_transformer|3|0|0|3|3397689|0|1399|-0.0235537654|
|cross_camera|registered_style|pure_mamba|4|3|1|2|3393394|0|1579|-0.00675989406|

## 同一次候选前向中的联合贡献

比较.5Signal+.5原角色h银行与实际fused；这是保存矩阵的标量分解，没有新检索头或反事实mAP。

|协议|输入|原h分量非正|实际fused非正|破坏正确|修复|可靠关系|可靠正确被破坏|可靠间隔违规|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|identity|original|0|0|0|0|42120500|0|3614|
|identity|registered_style|1|1|0|0|42120500|0|6731|
|cross_camera|original|0|0|0|0|3395768|0|688|
|cross_camera|registered_style|0|0|0|0|3395768|0|1424|

## 完整明细与解释边界

所有比较和来源身份见下列CSV；空比率表示分母为0，不表示0错误率。无合法跨camera正例身份保留。
向量分布保留每fold/输入/角色/模态/比较/计划的精确分位数；未平均分位数冒充总体分位数。
向量旋转本身不是身份证据丢失；应结合模型内部的真实关系变化。当前h到y有界不限制初始化到最终h的漂移。
本报告从已核验JSON导出，不重新计算模型特征；执行器核验不是外部独立审计。

|文件|完整行数|字节|
|---|---:|---:|
|[relation_groups.csv](relation_groups.csv)|1296|333461|
|[identity_relations.csv](identity_relations.csv)|60912|11123926|
|[joint_groups.csv](joint_groups.csv)|24|6603|
|[joint_identity_relations.csv](joint_identity_relations.csv)|1128|213873|
|[vector_groups.csv](vector_groups.csv)|648|189212|
|[identity_vectors.csv](identity_vectors.csv)|20304|5904360|
|[relation_totals.csv](relation_totals.csv)|216|63605|
|[joint_totals.csv](joint_totals.csv)|4|1608|
