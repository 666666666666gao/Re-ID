# 已验收权重的角色能量与方向诊断

登记日期：2026-09-29。状态：源码准备／等待审查；尚未执行GPU诊断。原15端继续完整训练，不改九份冻结runtime源、参数、权重或队列。

固定读取 accepted_667 中 RGBNT201、MSVR310 各五条件，共10份已经完整50轮／mAP-best／严格重载／CPU完整图库验收的权重。RGBNT100两端未完成，当前不纳入此诊断；不依据结果选择条件、epoch、身份或图像。

问题：局部辅助身份监督提高local-alone检索，但没有一致提高fused。测量原1536维global、区域角色修正及其学习缩放后的实际能量和方向，区分局部表示可辨识性与它对加性融合的作用。它不能单独定位失败原因。

执行 tools/diagnose_correspondence_context_energy.py。复用正式build/load/loader，模型eval、inference_mode；hook只读取backbone输出和角色证据，原forward不变，不截断、替换或调权。完整query/gallery记录每样本global范数、correction范数、abs(gain)*norm(correction)/norm(global)、global与correction余弦、global与fused余弦及L2差；summary报告各split分布和学习gain。保存remote逐样本标量与完整协议的地址绑定，local仅归档JSON。

必须核对：冻结九源SHA；权重、正式距离与回执SHA；build initializer相同；独立重载所选best；重构fused与原forward误差<1e-5；三个输出的完整距离矩阵与原已验收保存数组最大误差<1e-5。没有产生新检索成绩，不重写原official文件。任一失败保留日志和不完整目录，不能自动重试或放宽阈值。

两个数据集分别在空GPU1、2执行，各串行五权重；启动前GPU占用<500MiB、原15父队列无pending/failed及所有原M3端complete。使用现有tri_reid环境，无安装或新环境。预计每数据集约2–5分钟，只是估计；记录真实PID/命令/退出与耗时。不会抢占剩余RGBNT100训练。正常推理一次embedding，不运行分类头或教师。

这些是官方选点后的机制诊断，范数或余弦变化不等同于参数梯度、AdamW更新份额或因果证据；local路径仍能影响共享适配及context条件，不能称纯独立局部学习。其结果不用于选择倍率、失败query规则或取消剩余正式端。
