# role-set v1 合成数学检查

执行提交54a7d5ae879c297bedba9f9cbe2f06cccb9121b0；CPU检查于2026-09-08 18:08:47退出0，18:09:23实查wrapper33264/child33266均已结束。全部文本按SHA接收，math SHA167a8d8d873fc1d3ea4fb741266f22520b76121f43979545f3acac563940bd6e。

PASS_ROLE_SET_MATHEMATICS：合法class0/身份mask、去重、无额外提议时原标量与原梯度相同、current/history极值精确tie的原梯度保留、额外active非极值负项有非零距离导数、历史VJP与完整图参数导数精确一致、无历史情形全部通过。

这些是合成CPU向量及合成可训练矩阵的数学检查；没有真实模型前向、训练更新、图像读取或新权重，不是M0训练或检索收益。没有把合成矩阵的导数冒充CNN/Transformer/Mamba的真实参数梯度。全部原始日志/收据见evidence/msvr310_role_set_math_20260908。

下一登记是三折固定初始化真实模型梯度检查，见refine-logs/msvr310_role_set_v1/GRADIENT_CHECK_PLAN.md：每折8个来源batch，0优化器更新，原目标/新目标/重复目标以及完整历史侧导数，全部4种距离矩阵CPU复算。该检查尚未启动，完整M0和Q1仍未登记。
