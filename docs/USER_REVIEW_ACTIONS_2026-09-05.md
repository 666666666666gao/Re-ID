# 用户更新复核后的执行决定（2026-09-05）

这份记录接收用户本次长篇项目复核的优先级，并区分后继研究与当前固定实验。

1. V23先完成原始三fold、两端、20epoch、3360步比较，保留所有负结果，
   原五项门保持不变；M0工程PASS不代表未知身份检索有效。
2. 后继主假设优先放在环境内身份区分、真实跨摄像头监督覆盖与
   source-only困难负例。先核对IICI原型更新/增强监督及XBM负例覆盖机制。
   V22 MCNL、旧Router、私有尾部容量、各封存版本保持原结论，不换名扫描。
3. 保留全部真实source身份；多摄像头同ID始终是正关系。
   新采样/记忆必须登记完整覆盖和刷新合同，禁止heldout/dev图库进入训练。
4. 三数据集路线为RGBNT201、MSVR310、RGBNT100；
   新跨数据集资格优先MSVR310，再RGBNT100。已有安装/协议核验不当作模型成绩，
   MSVR310仍使用自己的scene/time规则，不改成行人same-camera过滤。
5. 当前seed42和主结果优先于消融的执行约束保持；主结果成立后再安排
   角色、容量、同计算预算和鲁棒性验证。额外种子尚未启动。
6. 当前V23不接入新模块。CLIP/DINO任务适配只保留已做的来源研究，
   本次尚无新的主干实验计划、下载授权或训练。

## 融合记号澄清

modeling/trifusion/signal_preserving_v8.py:445-515先将每专家三模态残差
归一化为1536D，再作等能量拼接。对归一化后的相似度，使用r_e表示纯残差：

s_fused = 0.5 s_Signal + (s_r_CNN + s_r_Transformer + s_r_Mamba) / 6。

实际日志的cnn/transformer/mamba输出是3072D Signal加1536D残差的完整分支，
各有s_branch_e = 0.5 s_Signal + 0.5 s_r_e。因此同一模型、同一对图像：

s_fused = (s_branch_CNN + s_branch_Transformer + s_branch_Mamba) / 3。

这不等于mAP可以相加或平均；AP依赖排序。保留baseline前缀没有单调检索保证，
最终评价仍须使用完整图库，不能用训练损失、跨模态余弦或Oracle代替。
