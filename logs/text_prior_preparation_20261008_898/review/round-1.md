# Round 1 方法评审

**REVISE · Pilot: REVISE_BEFORE_NN · 5.90/10 · CALIBRATION: none**

冻结文本先验值得一次有界必要性检验，但当前方案必须先修订接口。它尚无论文级新意，三集 SOTA、稳定性及机制必要性的完整目标仍为 **ACTIVE_UNMET**。

| 维度 | 权重 | 分数 |
|---|---:|---:|
| Problem Fidelity | 15% | 8 |
| Method Specificity | 25% | 6 |
| Contribution Quality | 25% | 4 |
| Frontier Leverage | 15% | 7 |
| Feasibility | 10% | 6 |
| Validation Focus | 5% | 8 |
| Venue Readiness | 5% | 3 |

最小修订：

1. 在视觉输入 detach **之后**计算 ψ→T→W，再把可微 context 送入角色；既有入口会 detach 整个 context，直接在外层相加会切断适配器梯度。
2. 显式使用普通 CLIP 的 12-block 文本路径；现有 Signal 默认 Transformer wrapper 含不适用的参数传递及视觉分支。
3. 初始 context_queries=0 时 ψ/W 首次任务梯度为零。固定“C 更新后再核对 ψ/W”的活动时序，排除仅由 weight decay 造成的参数变化；保留低 LR 的解释限制。
4. 固定随机包及初始化、prefix 数值容差、严格重载和实际资源资格。BPE 已支持 12 tokens/EOT=11；输出/梯度 parity、峰值显存和可训练性尚未实测。

容量对照合理：两端同为 592,000 新可训练参数与同结构冻结 T。但对比只能识别固定预训练包相对固定随机包的条件效果，不能单独证明可读语义或独立身份信息。还须同时超过原 semantic 与独立 global，不能拿一个差值替代三集目标。

GAP：无人工精选校准样例；主要差距是可执行接口、与先例的机制区别和完整目标证据。实例 inversion 与文本区域引导已有 [DEEP](https://aihuazheng.github.io/publications/pdf/2025/2025-DEEP_Decoupled_Semantic_Prompt_Learning_Guiding_and_Embedding_for_Multi-Spectral_Object_Re-Identification.pdf) 和 [PromptSG](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_A_Pedestrian_is_Worth_One_Prompt_Towards_Language_Guidance_Person_CVPR_2024_paper.html) 先例。修好接口可让 pilot 值得做，不会使方法自动具备原创性。

Simplification Opportunities: NONE；保留单一 ψ+W，不加损失、teacher 或 matcher。Modernization Opportunities: NONE。Drift Warning: NONE，原完整目标保留。

请求 gpt-6-astra/max；实际模型与 effort UNATTESTED。独立上下文、same-family、provisional。本轮只有只读评审，未执行 NN，也不产生执行授权。
