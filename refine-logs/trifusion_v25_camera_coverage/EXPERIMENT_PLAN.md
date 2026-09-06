# V25：真实跨摄像头正例覆盖的完整配对主比较

登记时间：2026-09-06T22:33:37.459788+08:00。状态 IMPLEMENTED_REGISTERED_NOT_RUN。
本次全量元数据重放已经通过；M0和Q1尚未执行。只登记这一固定采样改动，保留所有已封存结果与原门槛。

## 研究依据与解释边界

原RGBNT201每fold94个source身份中14个跨摄像头、80个单摄像头；
旧1680batch真实跨摄像头有向正对占比8.070791%，1580batch只有一组跨摄像头身份。
V24改变了双视图原型监督，保持物理采样不变，其完整Q1未通过。
本次单独研究实际训练样本的跨摄像头正关系覆盖，未沿用V24原型/双视图、V23 Adapter、
V22 MCNL、V20对比损失、V19私有尾部、V18投影、V17关系包络或早期Router。

IICI等来源只能提供环境内身份区分的动机；本采样器为项目内实现，不声称复现其方法。
公开代码与许可边界见已存 docs/IICI_XBM_SNR_CODE_AND_SAMPLING_CONSTRAINTS_2026-09-06.md。
不同数据集之间覆盖与增益的关联不作为因果证明；不按RGBNT100官方测试结果调参。
内部21个身份已经反复开发，本次仍为重复使用的开发资格验证，不把bootstrap当新独立验证。

## 唯一干预与公平性

control：原CrossCameraIdentitySampler；two_cross_camera：冻结的数据重放候选规则。
两端B64/K8、每fold20epoch、seed42、workers4、相同source集合和29/28/27 batch每epoch。
候选只对跨摄像头身份建立两份原分组，每批两组跨摄像头身份、六组单摄像头身份。
保持组内一个主相机锚点、一个其他相机锚点和六个随机剩余样本，不采用4+4平衡。
候选增加跨摄像头记录重复并减少其他身份相对曝光，这是该完整采样干预的一部分。
全量重放已确认20epoch保留每折全部94身份和全部source记录。
整体跨摄像头正对占比从8.070790816%
提高到12.344547194%。

两端必须分别逐批匹配各自预先冻结的1680个采样记录。
两端样本顺序不相同，不要求图像增强tensor或样本顺序跨端相等。
每端完整记录实际路径、索引、相机正关系数及全部loss；终点逐记录曝光数应与重放一致。
相同随机seed不等于相同样本，实验只能归因于整体采样规则，不能拆分重复曝光与环境贡献。

## 原网络、权重和损失

每fold均恢复原合法V12 source-only Signal与V8角色头最终权重；两端从相同完整模型SHA开始。
M0和任何既有后继训练权重不用于Q1。新控制端属于本次配对比较，不改写旧控制端或重跑封存干预。
冻结Signal和共享CLIP尾部，训练CNN/Transformer/Mamba角色及七组分类头。
两端总参数98,800,141，可训练7,841,292，203个tensor；实际M0逐项核验。
原3072D Signal、三路1536D残差、4608D残差银行、7680D融合、固定等能量拼接保持。
新增推理参数0，推理结构及原完整三模态需求不变。

保留七组ID/Triplet、14项原loss及权重；smoothing0.1、Triplet margin0.3。
复用既有构造与优化函数，仅新采样器和逐批采样核验/完整终态数组写入。
单视图SharedGeometryTripletTransform：256x128、padding10、flip0.5、erase0.5和原归一化。
AdamW LR0.00035、wd0.0001、warmup5/cosine20、AMP scale256、原gradient checkpointing，无梯度累积。
禁止增加新损失、改学习率/宽度/epoch/seed/相机配比或中途停负fold。

## T0、M0与完整训练

T0已由固定三折两端20epoch全量采样重放完成，3360批全部数据门PASS。
本次不增加复刻实现细节的单元测试；真实loader和模型接口由M0检验。

M0：三fold、两种sampler，各8个真实source batch，共48次只读模型前向。
逐批校验对应登记的源记录、ID、相机和顺序；两端完整初始权重SHA相等，模型状态不变，Signal前缀有效。
fresh fold0控制/候选各8个不同batch优化；另fresh候选固定第一个batch100步过拟合。
M0共116优化步，全部203tensor在各完整阶段须获得非零有限梯度；冻结state不变、AMP overflow0，
两种容量峰值reserved均<24576MiB。
ID熵下界F=0.75*H_94(0.1)=0.57838292104621；Triplet下界0。
固定第1和第100次更新前loss，(L100-F)/(L1-F)<=0.1，不选中途最小值或更换batch。
任何M0门失败则保存M0_FAIL并停止Q1，不把它解释为检索失败。

M0通过后同一持久进程按fold0-control/candidate、fold1-control/candidate、fold2-control/candidate完成。
六端均20epoch，分别580/560/540优化步，总3360步、215040次样本曝光；全部120个epoch均保存统计。
每端从原起点新建模型，保存唯一epoch20终点、strict reload后再执行完整held-out检索。
保存六个终点权重、每一步训练日志、五路所有特征/距离/完整图库排名数组和全部query指标。
仅最后固定epoch评价，不使用中途检索选点，不加入reranking/TTT。

评价每fold完整图库1000/1051/1075，合法query190/179/202，合计3126图库、571query、21身份。
无跨摄像头正例的2555条记录仍留作图库干扰，只从query分母排除。
同ID同摄像头过滤保持，无跨fold坐标距离；30-dev与官方访问0。
全部六端结束后逐文件SHA、完整排序/AP/R1/5/10和身份bootstrap复算，
并输出全21身份与571query的改善、退步和R1修复/新增错误；不得抽样替代全量核验。

## 五项科学门保持

1. candidate aggregate fused mAP比本轮control至少+1.0个百分点。
2. 三fold fused增益分别非负。
3. CNN/Transformer/Mamba各aggregate mAP增益非负。
4. 全21身份、seed42、10000次identity cluster bootstrap的95%下界严格>0。
5. candidate fused严格超过同checkpoint Signal与三种专家的mAP。

全部满足才Q1_PASS。失败封存本规则，不扫描配比、分组份数、seed、epoch或损失救回。
通过后D1/refit另行登记；主结果未成功前不做消融、多种子或额外官方调参。
RGBNT201/MSVR310/RGBNT100完整目标不变，现有RGBNT10083.284770/96.151603不代表已达SOTA。

## 环境、时间、磁盘与溯源

服务器tri_reid/RTX3090运行；本地仅文本/JSON/AST处理，不运行项目模型或图像/权重tensor。
22:28复查数据盘free 21.782GiB，
系统盘free 10.365GiB。
只存六个终点与完整评估数组，预计新增工件低于4GiB，不保存每epoch/resume副本。
已清理24个冗余resume权重的记录保持，必要基线/模型不删除。
预计M0约5–10分钟，完整Q1约1–1.5小时；根据实际耗时修正。
长任务在180–300秒或估计里程碑观察，观察超时不能触发重跑。
当前外部审计服务额度不可用，执行器验证不得标为独立审计。

配置SHA256：cc8178de4173e01115a39505115bbc3e67ce322c08cd59cafff952e965116e4b。
模型/训练/数据代码旧文件使用22:28服务器实测SHA；新文件逐字节登记在配置SOURCE_FILE_SHA256。
冻结的数据重放SHA256：468da3813828c49487c904249012f9c33b4b81c29fea7ccee1b6c4f242de2073。
训练执行前提交配置、计划、源码与登记，并同步GitHub/服务器/桌面交接；实际执行commit与启动信息另存。
