# JointSpectralDelta：具体适配源码／数学复核

2026-10-13；只核读文件和已有项目来源，未运行模型、测试、SSH或修改生产代码。复核`modeling/trifusion/joint_spectral_delta.py`的SHA256为`83b0ea3183ac063b2ff542511c44c8f764c67e335ecef81dc5f908551b145619`，私有逐字快照为`joint_spectral_delta_reviewed.py`。这是有界组件审查，不是完整训练接入验收或新颖性审核。

## 结论

**当前组件的轴、差分位置和显式梯度路径与所述适配一致；未发现组件层面的阻塞错误。**相对旧尝试，它明确改变了“跨光谱内容怎样成为可检索修正”，没有复做头入口、几何代理、位置mask或repair/keep。不过，它组合了公开LMF和项目V8差分思想，必须按项目化假设描述，不能称独有公式或预计涨分。

本次`rg`只找到组件自身的引用，尚未看到生产入口调用。下面的结论不覆盖新包装器是否实际detach了processed一侧、是否加入optimizer、M0是否活动、作者BN／分类头／Triplet是否仍保持原合同，也不覆盖最终训练效果。

## 1. 数据轴与实际运算

三项processed/reference每项为`B×3模态×16槽位×128`。44–51行stack得到`B×3角色×3模态×16槽位×128`，只对槽位轴求均值；einsum得到`B×3角色×3模态×4rank×128`。跨模态求和／乘积后，再沿rank求和得到`B×3角色×128`。**没有把跨光谱同编号槽位当真值对应，也没有跨batch乘法。**dense读出为`1536×384`，与原W保存尺寸相同，但使用方式不同。

作者[LMF实际155–167行](https://github.com/Justin1904/Low-rank-Multimodal-Fusion/blob/ea4755124d19bddf7941695215e98f16fbb1fff9/model.py#L155-L167)是分模态投影→乘积→rank组合。这里保持这个运算次序，但加入无仿射factor-LayerNorm，并不追加常数1或投影bias。

**应采用准确名称“归一化模态因子的低秩乘积”。**由于LayerNorm有输入依赖的均值、方差和epsilon，它不是严格的三线性多项式张量；也不是直接复制作者完整LMF。无bias/no1只排除了本层显式常数入口，不排除角色输入中的anchor、adapter bias、output-norm bias或模型在数据上退化到冗余信息。

## 2. Reference与梯度

13–21行只detach shared stages/context；同一角色的depth混合、role adapters、query/key/value/output投影、anchor和output norms继续可微。Reference跳过CNN、Transformer、Mamba及其桥接，保留同源semantic value的读取。对当前`GlobalTokenRoles`，其static global token也因Transformer被旁路而不进入reference，这是定义中的算子路径旁路，不是完整Signal或冻结CLIP教师。

代码执行的是`Ψ(processed)−Ψ(reference)`，不是`Ψ(processed−reference)`。令各模态归一化因子的reference为`q_m`，变化为`d_m`，乘积差展开含：

\[
\sum_m d_m\prod_{n\ne m}q_n
+\sum_{m<n}d_md_nq_{\ell}
+d_Rd_Nd_T.
\]

这使单模态变化也可以由其他模态的reference内容条件化；不能称“只保留三模态共同新信息”。对角色共享参数，任务梯度为`J_processed−J_reference`；reference不是停止梯度的目标，二者可能相消或共同漂移。对只在processed出现的CNN/T/M与桥接，梯度只来自processed。上述所有权仍须由实际包装器的真实检查证明。

若processed与reference逐字相等，两个delta条件的差分及无仿射delta-LN都为0，修正为0；**这只是组件恒等性质，正常随机初始化的真实角色输出并不与reference相等。**因此不能声称初始化完整模型必然等于global-only。最后delta-LN还会重标定非零差分，差分原范数小不代表部署修正一定小。

`joint_absolute`虽然计算reference诊断量，最终任务不使用它；共享角色／因子仍参与processed任务，但“reference有返回值”不等于它在该控制中得到任务梯度。

## 3. 与V8／V17的真实区别和重合

- **V8重合：**V8在相同冻结CLIP尾层参考下，对角色runtime减reference再读出（`signal_preserving_v8.py:397–477`）。本次继承“用匹配参考表达变化”的原则。区别是参考由共同训练的语义读取产生，差分发生在模态乘积描述子之后；不是V8冻结教师的原样复现。
- **V17重合：**V17已经计算两个角色peer的Hadamard乘积，然后MLP纠正冻结V8残差（`signal_preserving_v17.py:67–99`）。本次按RGB/NIR/TIR独立因子、每角色池化内容形成描述子，并显式扣除semantic参考；不能声称项目首次乘法或首次跨分支条件化。
- **当前角色重合：**Transformer/Mamba/Signal注意力已有非线性内容交互。本次能支持的是一个可控的描述子运算差异，不支持“旧结构只能表达unary、新结构才有联合信息”的全面结论。

## 4. 三条件能支持什么，何时需要第四条件

| 比较 | 可支持的有限效果 |
|---|---|
| joint_delta−add_delta | 匹配差分与dense读出固定时，乘积替代求和的条件效果 |
| joint_delta−joint_absolute | 乘积与dense读出固定时，扣除可微semantic参考的条件效果 |
| 新条件−旧raw semantic | 同时改变池化、factor-LN、差分、dense读出等后的系统效果；不能唯一归于高阶内容 |

**这三个条件足以回答上述两个条件效果，本轮不必为形式增加队列。**但若论文要声称“reference特别适合乘积”“两个设计存在协同”或报告二阶交互，则缺`add_absolute`，应补完整2×2。现有三项不能计算差的差。即使joint_delta同时优于另外两项，也只说明两项在该合理上下文中的必要性。

原structured readout保留3模态×4槽位组；新三项全部先池化槽位并使用dense W。**这项共变更必须写在计划与结果中。**若进一步要把“改dense本身”与整套新机制分开归因，还需同一原证据的dense unary控制；当前add_delta不是原证据直接dense，因为它也有因子、归一化和reference减法。无需现在同时上所有消融，但不能提前写相关独占贡献。

## 可执行前最小注意项

1. 真实构造中三条件共享同一原参数初始化，并以fork RNG生成因子，避免新增参数抽样改变旧camera/head/role的比较基础；三项factor数量为589824、rank权重12，共589836参数。
2. 包装器确保processed与reference均只有shared stages/context隔离，reference读出不是no_grad教师；作者raw任务、BN更新与optimizer覆盖仍须真实接入核验。组件本身不能证明它们。
3. 诊断同时保留actual/reference/delta、factor活动、同best的g/c/f和修复／破坏。LayerNorm后的更强修正、单独c提高或“差分很小”均不能替代净检索收益。

本适配可以成为一个有限、与历史接口小改不同的研究比较；当前没有真实结果、没有独特性证明，没有背景／压缩／错误对应的因果诊断，也没有任何READY训练分数。
