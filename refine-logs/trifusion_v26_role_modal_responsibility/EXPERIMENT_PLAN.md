# V26：固定采样下的角色—模态排序责任训练

登记时间：2026-09-07T00:35:14.340790+08:00。状态 IMPLEMENTED_REGISTERED_NOT_RUN。
本文件在真实模型M0/Q1之前固定；数学T0亦未执行。V25已完整Q1_FAIL并封存。
当前只有这一项新主干预；完整RGBNT201/MSVR310/RGBNT100基线与SOTA目标仍未达到。

## 动机、来源和主张边界

RGBNT100已经证明原三角色相对本机Signal的基线收益，不能抹去该正证据。
融合超过Signal、角色互补、融合超过最强角色是不同问题。
V25增加跨摄像头曝光后fused下降0.460638pp，Transformer下降而Mamba提高。
本次在固定OLD采样下检查：融合尚未解决且槽位贡献不利的真实关系，能否提供新的有效排序监督。
BIER文献及作者固定代码cd04edf已核对，见docs/BIER_ROLE_MODAL_RESPONSIBILITY_SCOPE_2026-09-07.md。
本方法受其不同学习职责的思想启发，独立实现以下公式；不复制GPL代码，不声称BIER复现或已证明的新颖性。

## 唯一干预与固定公式

两端都使用原CrossCameraIdentitySampler。control原14项loss；responsibility保留原14项并增加L_R。
每批B64/K8；所有有向行三元组T={(q,p,n):id(q)=id(p),q!=p,id(q)!=id(n)}。
每批448个有向正对、56个负例/正对，共25088个三元组曝光、225792个槽位三元组曝光。
身份0合法；相机不改变真实身份标签。同一记录在不同增强行的重复保留，并报告重复正对数量；
这些是训练曝光数量，不是同等数量独立图片或独立统计样本。

使用实际输出L2归一化后的融合cosine间隔m_F及各角色模态槽位cosine间隔m_s。
w_s=stopgrad[sigmoid(-m_F/tau)*sigmoid(-m_s/tau)]。
ell_s=tau*softplus(-m_s/tau)。
L_R=(1/(9*|T|))*sum_s,sum_t w_s(t)*ell_s(t)。
L_candidate=L_original+lambda*L_R；L_control=L_original。
固定tau=0.1、lambda=1.0，不从任何本轮或官方指标选择。
分母是所有合法行三元组数量；不按权重和归一化，避免将极少支持自动放大。
权重与loss计算FP32，反向继续使用原AMP scaler。该零间隔平滑排序辅助项非负，下界0。
原Triplet margin0.3及14项权重都不改变；本轮不是替换Triplet或Smooth-AP。

九槽位均归一化且非零时，s_F=.5*s_0+(1/18)*sum_s s_s。
实际raw fused必须先归一化；不得直接用raw特征点积声称该等式。
M0/全训练记录原槽位范数及实际融合分解最大误差，固定工程容差0.005。
有限精度误差完整保留，不声称tensor级完全相等。

## 固定模型、数据、初始化与预算

原冻结Signal/V8、3072D基线、三个1536D残差、4608D银行、7680D融合；
全三模态、稠密三角色、固定拼接。新增模型参数0、新增推理计算0。
总参数98800141，可训练7841292、203个tensor，真实M0重新核验。
保留七组ID/Triplet、原ID权重/平滑0.1、原Triplet权重/margin0.3。
不启用Router/HFER、V23模态MLP、V24原型、V25采样或旧失败干预；
不加入XBM、PCGrad、SNR、Smooth-AP、隐藏交换、对齐或蒸馏目标。

三fold每折94个source身份、14跨相机/80单相机；source记录2126/2075/2051。
每端使用合法V12 source Signal与V8角色最终权重，相同完整初始化SHA；
不重训既有Signal，不从V25或M0继续，不混合fold坐标或身份。
B64/K8、seed42、workers4、20epoch、29/28/27 batches每epoch；
每端580/560/540updates，六端3360updates/215040曝光/120epochs。
单视图SharedGeometryTripletTransform原几何、归一化及erase不变。
AdamW LR0.00035/wd0.0001，warmup5/cosine20，AMP初始scale256，无梯度累积。
两端全1680batch记录和顺序相同；每端逐批核对原冻结元数据，前八批增强输入SHA跨端相同。
原跨摄像头正对8.070790816%两端完全一致；全部source记录保留。

## T0与M0：证明新信号真实存在

T0在远端CPU执行，无模型/数据/权重访问：
用4行、两个含class0的真实标签集合、九槽位数值例子，
独立NumPy解析式计算loss及归一化余弦对原向量的梯度，与Torch自动微分比较；
检查fused/baseline只影响停止梯度权重、无辅助反向，以及行重排一致性。
T0不是模型效果证据。

M0共48个只读真实前向：3fold*2arms*8个source batch。
两端模型初始SHA、输入与输出SHA、责任统计匹配，模型state不改变。
fresh fold0控制8步、候选8步；另fresh候选固定第一个batch100步过拟合，共116updates。
203个trainable tensors各阶段得到非零有限梯度、冻结state不变、AMP overflow0、reserved<24576MiB。
七头ID下界F=.75*H_94(.1)=.57838292104621，新辅助下界0。
固定(L100-F)/(L1-F)<=.1，不能选择中途最低loss或换batch。

所有步骤记录每槽位权重min/mean/max、有效样本量、槽位/融合非正间隔数量、
联合不利关系数量、辅助loss及九槽位辅助梯度范数。
控制端也只读计算该辅助量及槽位梯度以便同等诊断，但优化器只使用原总loss。
候选M0八个容量更新同时在相同189个encoder参数上计算三种目标梯度：
fused两项、角色及残差十二项、L_R；CNN42/Transformer54/Mamba93个参数tensor。
记录同角色fused-vs-role与base-vs-auxiliary梯度余弦、范数及新增/原梯度范数比。
复用真实训练前向与保留图，不增加额外训练模式前向以免改BatchNorm状态。
候选须九槽位辅助梯度及三角色新增encoder梯度非零；是否梯度冲突仅诊断，不是预设成功条件。
M0失败就完整保存并停止Q1，不扫描温度/权重/步数救回，不将其标为检索失败。

## 完整Q1与原五项科学门

M0通过后同一持久原进程执行三fold control/responsibility全部六个固定epoch20终点。
每端保存唯一final权重，strict reload后评估原五路输出；不按中途检索选择epoch。
完整图库1000/1051/1075，共3126条；合法query190/179/202，共571、21个身份。
无跨相机正例记录仍作完整图库干扰，同ID同camera过滤保持。
30-dev与官方访问0，test-time更新/重排序0。OOF身份已反复开发，非新独立验证。

五项条件全部原样保留：
1. candidate fused aggregate mAP比本轮control至少+1.0pp。
2. 三折fused增益全部非负。
3. 三个角色aggregate mAP增益各非负。
4. 21身份cluster bootstrap、10000次seed42、95%下界严格>0。
5. candidate fused严格超过其Signal及三个完整角色输出mAP。
六端全部完成才判定Q1_PASS/FAIL，不中止负fold。
完整保存全部3360训练行、120epoch均值、五路特征/距离/全图库排名和每query统计；
随后全数组、权重SHA、训练曝光和原指标/身份bootstrap复算。
统计全21身份及全571query改善/退步、R1修复/新增，不挑展示身份或补写未读取图像成因。
结果失败则封存该固定配置；通过后D1/refit另行登记。
主结果成功前不做旧版本消融或救回扫描，不把本次版本比较称规范消融。
未来直接BIER、FIFO-XBM、Smooth-AP等对照需在主方法成立后单独登记。
实例记忆须先有增强实例新增难例及漂移证据，本次没有执行。

## 环境、磁盘与溯源

远端tri_reid/PyTorch2.5.1cu121/RTX3090运行；本地只处理文本、JSON、AST、解析算术。
00:25实测数据卷free22626324480B，系统卷free11129257984B，GPU无进程。
此前24个冗余resume权重已删24.900344GiB；基线、有效终点、核验数组保留。
仅六个final和全量数组，预计新增低于4GiB，不存每epoch或重复resume权重。
预计M0约5–12分钟，完整Q1约1–1.5小时，以实际步骤速度修正。
长任务180–300秒或预估里程碑观察，不用传输超时作为重跑理由。
外部独立审计服务额度不可用；执行器验证不标为独立审计。

配置SHA256：01cc6e8d80d8ae68df3d2d858ed9102b9faf1b46c3021bb1ca993a0dbc775f13。
元数据SHA256：656f91a81f33f43141d7ca275be7b80ef71884cb02283fb930f21d83695cb4e4。
原20个代码/协议依赖按已提交Git与远端实测字节固定，新三个源文件逐字节登记。
本地部分旧文件仅CRLF不同，校验提交blob且不改旧文件。科学配置在执行前提交并三方同步。
