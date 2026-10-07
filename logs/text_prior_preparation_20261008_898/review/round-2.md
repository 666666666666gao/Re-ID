# Round 2 方法评审

**REVISE · 6.40/10 · CALIBRATION: none**

**Pilot：WORTHWHILE_PENDING_ORDINARY_GATES。** 修订后的受限诊断值得继续，已不需要方法重设计。普通详细计划、source、资源、prefix 和 M0 门仍未执行且必须保留；本评审不授权直接启动 NN。

| 维度 | 权重 | 分数 |
|---|---:|---:|
| Problem Fidelity | 15% | 8 |
| Method Specificity | 25% | 8 |
| Contribution Quality | 25% | 4 |
| Frontier Leverage | 15% | 7 |
| Feasibility | 10% | 6 |
| Validation Focus | 5% | 8 |
| Venue Readiness | 5% | 3 |

R1 的四项问题已在设计层面解决：文本在视觉 detach 后接入；显式普通 CLIP block 路径；区分 C=0 首步延迟与永久断路；将结论限制为固定预训练包的条件效果。六端、ψ+W、原损失和门槛均保持。

交付 source 时仍须核对实际 d_model/n_head 签名、构造随机数与登记填值顺序、共享 ψ/W 初态，以及包含模板 buffer 的完整状态清单。这些属于实现现有合同，不要求新增模块或第三轮方法评审。

GAP：没有人工精选校准样例。只因方法接口更明确，Method Specificity 从 6 升到 8；其他轴未提高。近邻 [PromptSG](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_A_Pedestrian_is_Worth_One_Prompt_Towards_Language_Guidance_Person_CVPR_2024_paper.html) 与 [DEEP](https://aihuazheng.github.io/publications/pdf/2025/2025-DEEP_Decoupled_Semantic_Prompt_Learning_Guiding_and_Embedding_for_Multi-Spectral_Object_Re-Identification.pdf) 仍限制原创性，未执行的资格不能计为实测成功。

容量对照适合这个限定问题；正的 pretrained−random 仍须同时报告对旧 semantic、独立 global 的净收益。单 seed 开发 pilot 不能证明 SOTA、稳定性或独立新身份信息。完整目标继续 **ACTIVE_UNMET**。

Simplification Opportunities: NONE。Modernization Opportunities: NONE。Drift Warning: NONE。两轮上限不构成接受条件。

沿用同一评审者；请求 gpt-6-astra/max，实际 model/effort UNATTESTED，same-family/provisional。未执行 NN，R1 全部原始材料与失败记录保留。

