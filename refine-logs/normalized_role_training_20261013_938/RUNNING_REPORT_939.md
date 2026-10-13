# 恢复训练后的真实初始化/M0与原fresh50启动

**用户“继续训练”已经落实：一个201 semantic/seed42原流水线真实运行，初始化与8步M0通过后进入fresh50。尚无完整50轮/首次strict正式结果。**原停止/等待范围确认已解除，Goal仍ACTIVE_UNMET。

唯一改动保持§936/938：角色L2(h)先经既有detached BN/head分类，Triplet仍raw h；global/读取/角色/作者配方/部署不变。没有原生CNN、额外头、损失、seed/gain/margin/层数或车辆扩展。source/controller注册提交`6babb0b863448da3c8a8b07ccba8613e5d8e7471`，482项运行来源封存。

原supervisor354646/controller354648于2026-10-13T21:57:48.490853+08:00一次启动，CUDA0,1实际进程环境与命令核查一致。只26物理GPU0/1，一个原双卡分段进程；GPU2/3和25未使用，未查询/设置温度与功率，未安装或重建环境。

prepare运行一次，控制器通过完整binding与原raw semantic initializer的比较；只排除登记的架构/entry/scope/policy元数据，新state/visual/camera/参数量/cfg/采样/分段与旧配对保持。它是当前程序检查的实际结果，不是独立来源审计。

真实M0 2026-10-13T21:58:41.511972+08:00→2026-10-13T21:59:09.295914+08:00通过：8有效optimizer更新、281/281 trainable张量至少一次非零有限梯度、BN bottleneck计数8、严格重载最大差0；冻结参数不变，visual和fresh camera改变。原production optimizer构造覆盖及唯一参数所有权检查通过。M0探针SHA `0546388c53fee7bd5743c3ab69b02bbadc7556b1c9bba7d18ccc1cf07ad5c996`，仍保留至首次full严格验收及verify完成，不把它作为fresh50起点。

原fresh50 child358365于2026-10-13T21:59:12.822883+08:00独立启动。唯一M0节点观察2026-10-13T22:01:03.022582+08:00确认supervisor/controller/child均live、actual CUDA0,1，完成1/50。**该1轮是观测时刻进度，不是当前即时轮次或新的正式best成绩。**不报告中途测试数值，不据此改配方。原controller将完整50后执行首次严格重载，使用同一mAP-best的全部CMC。

预计一端full约一小时，当前唯一近完成观察为local session23306/PID43956，deadline 2026-10-13T22:54:12.822883+08:00（原full开始后55分钟），此前无远端训练进度查询。180–300秒长任务观察口径保持，不因前端session超时/缺失或campaign残留状态重启训练。下次只消费原handle/原进程/原EXIT。

计划完成后分析保存的全query修复/新增错误、身份AP与完整50轮行为；同模型G/C/F如需要另行限定。原raw semantic/global-only/完整Signal成绩保持；此时不宣称超过Signal、分类入口有效、原创模块成功或车辆可迁移。所有原best/必需依赖保留，本次权重删除0。新formal逻辑只写/覆盖mAP-best，M0是当前验收依赖；未完成前不清除。
