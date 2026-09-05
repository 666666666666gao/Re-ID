# V20 Experiment Tracker

更新时间：2026-09-05T16:12:19.079256+08:00。状态：Q1_FAIL；独立终态审计待完成。

| ID | 内容 | 状态 | 证据/依赖 |
|---|---|---|---|
| V20-T0 | 三项远端CUDA数学单测 | DONE_PASS | evidence/trifusion_v20_t0_20260905.json |
| V20-M0 | 六模型配对、新损失梯度、容量、100步过拟合 | DONE_PASS | evidence/trifusion_v20_m0_seed42_3cea5bf.json |
| V20-M0-AUDIT | 独立M0完整性审计 | DONE_WARN_ENGINEERING_PASS | EXPERIMENT_AUDIT_V20_M0.md/json |
| V20-Q1 | 三折两端20epoch、五路完整终态 | DONE_SCIENTIFIC_FAIL | evidence/trifusion_v20_q1_seed42_3cea5bf.json |
| V20-FILES | 远端32文件SHA、6receipt一致性 | DONE_PASS | evidence/trifusion_v20_terminal_file_verification_20260905.json |
| V20-ARRAYS | 全数组/掩码/3360步/10000bootstrap实际重算 | DONE_PASS | evidence/trifusion_v20_terminal_array_audit_20260905.json |
| V20-REPORT | 三折五路、21身份、全部query变化 | DONE | results/TRIFUSION_RGBNT201_V20_CROSS_MODAL_IDENTITY_2026-09-05.md |
| V20-AUDIT | 独立完整Q1审计 | PENDING | .aris/traces/experiment-audit/2026-09-05_run07 |
| V20-D1 | 141-fit refit与30-dev | NOT_QUALIFIED_NOT_RUN | Q1固定门失败 |

三折两端各20epoch、120条epoch记录、3360优化步全部完成；运行4236.036166秒
（70.600603分钟，包含M0等整体流程）。原进程26383在16:08:40CST观测已退出，
GPU 1MiB/0%。固定Q1判定FAIL，D1拒绝晋级，dev/official/D1访问均为0。

完整141 heldout身份、3126 gallery、571合法query/21跨camera身份保留。
2555条只从query排除、仍作gallery干扰项；三折gallery/query为
1000/190、1051/179、1075/202，六端全部最终epoch20保存后strict reload，
读取六个模型的全部baseline/fused/CNN/T/M输出，无末折提前停止或结果删选。
配对初始化、完整采样序列、前8增强、绑定、baseline输出均相同；
六端203/203训练tensor梯度覆盖、overflow0、冻结state不变全部通过。

本次实际对照→跨模态身份损失的全量mAP：
baseline77.487603→77.487603，fused80.206258→79.195387（-1.010871），
CNN79.126676→78.116938（-1.009739），Transformer78.475388→73.695598
（-4.779791），Mamba77.780907→79.087275（+1.306367）。
三折fused差-1.087608/-2.539986/+0.416314。fused Rank1从83.012259降至
79.334501：9个原错query修复，30个原对query变错。全部571个query中AP改善189、
下降208、相等174；21身份全表和五路Rank1/5/10见完整结果及配套JSON。

固定五个科学条件中，aggregate>=+1、各折非负、各专家非负、bootstrap下界>0
四项失败；只有候选fused高于同checkpoint的baseline和三专家通过。
21身份聚类、10000次seed42 bootstrap的95%下界为-3.8126559810990917。
跨模态监督未带来完整泛化收益；不能以Mamba单路收益晋级，也不能据此断言
所有跨模态监督必然无效。V20封存，不扫描温度/系数/分支/epoch或另种子重训。

远端32个绑定文件（包括六个最终权重）全字节SHA校验及六receipt与summary
对象相等校验通过；下载原始summary2022853字节，SHA
23c683b92ad3551e9aa07a24470e82c47565ef54b6683e00213ce7ea0bfbf522；
日志65335字节，SHA978a9f98f8c2d38cb59b101c834c8838acab139c580f88e2612bb2585a00d50e。
本地NumPy2.5.2实际重算全部掩码、三折两端五路AP/rank聚合、增益和bootstrap，
最大绝对数值差1.3322676295501878e-15个百分点，训练损失分量检查通过。
此为JSON算术与文件SHA核验，未在本地执行模型/权重张量加载/图像或距离重算。
独立GPT-5.5 xhigh完整Q1审计待完成，M0审计不能替代Q1审计。

原执行source3cea5bfc17e214b1829c020527699d939efa221d；两端98,800,141总参数/
7,841,292可训练参数/203训练tensor，无新增推理参数。所有训练源码/配置/计划
保持执行前绑定。前两折15:38:59原始快照也保留：1440147字节，SHA
59b7bfdce0d1b0d8fe5ca351e3f7f53c16a79ab78ef5eac40bccb22557c49053。

固定计划及先前tracker永久副本保留。训练内OOF已跨版本反复开发使用，不是
独立dev或官方证据；全局目标仍未完成，可部署dev最佳仍为V8 58.4050/59.3939。
下一步先完成独立终态审计与证据归档，再固定一个有来源的新主实验假设。
目前未实现、预注册或启动V21；新优化器方向仅处于原始论文/作者源码研究。
