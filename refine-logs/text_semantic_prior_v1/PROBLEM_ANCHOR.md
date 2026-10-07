## Problem Anchor

- Bottom-line problem：在RGBNT201、RGBNT100、MSVR310三个数据集形成超过同协议强baseline、明确预训练资源且可经当前文献核验的SOTA方法；完整流程稳定性和机制必要性仍须证明，不能将本次小实验替代整个目标。
- Must-solve bottleneck：当前角色已经参与训练且有独立判别能力，新增修正仍未稳定改善global的未见身份排序；更大的读取容量、纯视觉重建、独立fused头、MD/repair和当前同模态几何辅助均已有闭合结果。下一项必须检验不同的身份证据学习依据，而非重命名上述控制或继续搜索gain/温度/margin/seed。
- Non-goals：不把训练配方、初始化/显存/数值修复计作创新；不宣称固定文本变换创造了输入图像没有的新信息或已经识别真实部件；不使用测试身份标签、在线gallery适配或人工Oracle。
- Constraints：只2026物理GPU0/1，现有conda、三数据集及公开CLIP；不查25/GPU2/3/温度功率，不安装。seed42、每个合格端fresh50、单一fused mAP-best与同权重CMC、完整合法query/gallery及camera/scene过滤；官方集已消费，属于开发比较。闭合旧控制不重训、不调原门槛。
- Success condition：机制对照必须同时优于匹配原角色控制与独立global，并分清新增容量与预训练先验；任何小试成功都不自动证明SOTA、多种子或三个创新点。最终完整目标保持ACTIVE_UNMET。
