# V25 experiment tracker

更新时间：2026-09-07T00:10:56.694726+08:00。COMPLETE_ENGINEERING_PASS_Q1_FAIL。
原训练112550和wrapper112548于2026-09-06 23:50:33正常退出0；完整120epoch/3360updates/215040exposures。
自动核验队列114796于23:50:53正常结束0；六数组、全部571query/21身份/五输出完整复算。

| 阶段 | 状态 | 核心事实 |
|---|---|---|
| T0 metadata | COMPLETE_PASS | 全3360批采样与曝光门通过 |
| M0 | COMPLETE_PASS | 48只读batch+116优化步，203梯度、overflow0 |
| Q1 | COMPLETE_FAIL | fused80.881569→80.420931，变化-0.460638pp |
| 原五项科学门 | 1 PASS / 4 FAIL | 仅融合高于同checkpoint全部分支通过 |
| 完整CPU核验 | COMPLETE_PASS | 32602260距离元素、5952790排序位置，误差0 |
| 全部身份/query报告 | COMPLETE_PASS | 30fold指标、105身份输出行、2855query输出行 |
| D1/dev/official/ablations | NOT_QUALIFIED_NOT_RUN | 本规则封存，不扫比例/epoch/seed |

三折融合变化+0.564526/+1.526056/-3.185387；bootstrap下界-2.306666pp。
CNN-0.636440，Transformer-2.026329，Mamba+1.835080；Signal全特征/距离/排序两端相同。
融合R1修复3、新增15；11身份改善、10下降。下降贡献中000250/000261占61.4870%。
24份原文本15638325字节全量接收并逐文件SHA核对，权重/检索tensor留在服务器。
执行97468dd；run_summary SHA aecf7353783a375e8f4226f72f1151a6a94a884ffd50ce36520e5ffdb88f9bf0。
报告：results/TRIFUSION_V25_COMPLETE_COMPARISON_2026-09-06.md与COMPLETE_FAILURE_ANALYSIS。
磁盘23:51 datafree21.094658GiB，系统10.364922GiB；清理24.90GiB结果保持。
用户后继方向：固定采样下先验证角色—模态排序责任；实例记忆有诊断前提。当前尚未运行新模型。
RGBNT100官方基线增益仍成立，三数据集整体目标未达成，执行器核验不等于外部独立审计。
