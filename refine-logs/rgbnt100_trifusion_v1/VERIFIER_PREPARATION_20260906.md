# RGBNT100 原三角色核验准备

记录时间 2026-09-06T12:54:28.792843+08:00。四个核验入口已完成静态检查，尚未在RGBNT100三角色上执行；正式模型配置未生成，M0和完整比较均未启动。
当前Signal固定30epoch基线的训练入口、配置和数值定义没有改动。

- M0文件核验在远端只读取三个完整role checkpoint、对应B0权重及原日志，核对全部baseline.signal状态、源码/配置/协议/预训练文件SHA和124步JSONL。
- M0标量核验在本地使用stdlib，逐步核算真实源身份/相机、64条记录的8×8组成、203项梯度finite标记与AMP是否更新，重新计算固定100步过拟合及标签平滑下界。
- 完整比较文件核验在远端检查三权重与15组features/distances；全部距离重算须逐位一致，完整排序与数组argsort一致，baseline须逐位等于同fold B0。
- 完整比较标量核验在本地使用JSON/stdlib/NumPy，遍历43375条query-output、50个身份、全部60epoch和真实步骤，重算mAP/Rank、身份收益、错误修复/新增和原五项条件。

本数据集采样器每epoch的实际步数会变化，已由当前Signal首折81–84步实证确认；不能照搬MSVR的13步固定推算。
评价只删除same-ID AND same-camera项，沿用真实协议字段valid_positive_count。其余身份同相机负例保留。

损失重算按真实weighted_training_loss先对三专家求和再乘共同权重，并保留FP32乘加顺序。
该算术函数已对已发表的MSVR M0四份JSON全部124行重放，124/124逐位相同，最大差0；
这只是已有标量上的核验器检查，不是RGBNT100 M0通过，也未重新执行模型、张量、图像或训练。
新模型M0与正式训练仍须各自真实终态、文件和全部标量核验。

四个入口SHA及旧JSON检查范围见evidence/trifusion_rgbnt100_original_roles_verifier_preparation_20260906.json。
后续正式配置应绑定这四个核验源码以及训练入口、模型/数据/损失依赖；不能把当前无真实B0终态的准备稿当成可执行合同。
