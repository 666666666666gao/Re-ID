# V28 R2 执行跟踪

2026-09-07，运行commit bf8de956e685311dd70631395009a2c06a2c8591。
09:02:55启动子进程143321 / wrapper143320；09:03:17确认child存活。
远端run：/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v28_joint_tokens_fp32_seed42_bf8de95。
原配置SHA43b75c87e2e7795759912b9051fae012b2cc39a23279b5d3d6bc85600fae9029，
原R2计划SHA64111a40d44e9a28ff4e9ab62670d608254480fc5c04e44f6841a9613b281e7b。

T0三项PASS：风格公式、实际CUDA合成联合模块、真实fixture梯度回归。
真实旧AMP dt0/2048，新局部FP32 dt2048/2048；scaled最大梯度3.534658077342101e-8，
两端梯度有限。无图像/优化更新的fixture测试不初始化真实fold。
M0正在运行，尚未发布M0终态或Q1检索结果。
R2数值修复没有改变原116步M0、219/203覆盖、过拟合与五科学门。

启动前校验脚本第一次生成时，通用HEAD占位替换误改git HEAD字面量而SyntaxError。
该次检查根本未执行、screen/child未创建；修复为专用占位符并先compile后执行。
模型wrapper已正确编译，未改模型/计划，真实训练只启动一次；错误JSON保留。

09:02:55磁盘实查：data15624220672B，system11129249792B；
GPU空闲24126MiB。既有24个删除路径仍不存在；12受保护模型完整SHA一致。
既有回收26736541280B，本次新删除0。预留本轮六final与数组<1GiB。

R2专用tools/verify_v28_fp32_m0.py、verify_v28_fp32_complete_terminal.py、
report_v28_fp32_complete_comparison.py已准备，尚未执行或冒充审计PASS。
原R1脚本和失败报告保持。绑定R2执行commit/config/plan及precision fixture。
完整Q1核验将检查3360更新、32,602,260距离元素、5,952,790排名位置、571query/21身份。
单seed42、重复OOF、非外部独立审计和多数据集/SOTA目标未完成边界不变。
