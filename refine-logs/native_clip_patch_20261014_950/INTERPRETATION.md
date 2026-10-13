# §950：原CLIP patch形成路径的固定best诊断

§949三端训练已经终态，本次没有重复训练、M0或旧报告。现有原角色处理后区域的局部比较最高11.577mAP，尚未建立强教师。因此检查真实原CLIP输出及辅助dense读取，先回答证据在哪里形成，再决定后续训练机制。

已核真实作者VisionTransformer：所有token经ln_post和768×512预训练投影；包装encoder直接取CLS为global。原角色另用768→128投影。源文件SHA固定，不把空间宽度差直接称唯一原因。

ClearCLIP作者commit ad68a404d55d48d27330b93554eb64a234ff717f的custom_attn使用QQ/V读取。首次API403保留，随后git固定commit读取4份原代码；不安装/导入整库、不下载额外权重或冒称分割方法复现。标准QQ数学实现用现有环境CUDA隐藏的两个CPU fixture比较作者函数，差0；这是组件证据，另由真实完整提取检查接入。

本次一个joint_delta E8 frozen backbone前向批次序列遍历完整query/gallery，同时读取三种128×512 patch来源；没有optimizer更新或新训练权重。原G计算保留，CLS与G一致、模型state前后SHA相同、grad空、完整camera评价与作者一致。Q/G为RGBNT201原836图协议，未缩库。大特征不写盘，仅内存SHA；保存四份距离和逐query结果。

# 原CLIP patch与辅助dense读取：固定权重

| 来源 | mean mAP/R1 | MaxSim mAP/R1 | ΔmAP | 修复/新增错误 |
|---|---:|---:|---:|---:|
| native_projected | 55.4165/59.9282 | 60.3528/58.7321 | +4.9363 | 91/101 |
| attention_only_qk | 72.8013/75.1196 | 71.9707/74.8804 | -0.8305 | 18/20 |
| attention_only_qq | 55.6361/55.8612 | 61.3513/62.3206 | +5.7152 | 105/51 |

One frozen best backbone query/gallery pass; original global path untouched. Native projected128 patches versus auxiliary actual-QK-output and QQ-output without final residual/FFN; the latter also bypasses the last external role-adapter snapshots. Larger512-width/128-token bags differ from prior128-width/16-slot evidence, so no unique bottleneck/processing cause is established. Borrowed dense-reading principle, not full ClearCLIP segmentation reproduction. Scores are local-only diagnostics, no global fusion coefficient, fitting, test centering, new formal method score or teacher assumption. Feature tensors are hashed in memory, not saved; inputs/weights/source and resulting distances are retained for replay.

实际编码 27.327 秒，G距离与原固定best最大差 0；原G四项 {'mAP': 74.29666357468751, 'Rank-1': 78.94736842105263, 'Rank-5': 88.27751196172248, 'Rank-10': 91.86602870813397}。原正式74.5331等成绩不变。

注意力-only两项去掉末层残差/FFN，也绕过最后外部role adapter snapshot；不能称仅QQ与QK的独立因果对照之外的所有差异都被隔离。与旧16×128槽位相比，128×512候选的数量/宽度同时增加，不证明唯一投影瓶颈。局部评分不是新正式方法成绩，未搜索G融合倍率、test统计中心化、seed或更换best。

后继训练必须依据实际原patch能力设计，并保持独立global任务和匹配对照；不能因某个局部高分就预先认定是强教师、因某个局部低分宣布所有patch无效。三数据集全面超过Signal的Goal仍ACTIVE_UNMET，先201/seed42，车辆和额外种子不提前展开。

原三端best、初始化与验证依赖保留；已退休M0只删过D精确归档的远端重复，旧verifier使用前必须恢复。只有26物理GPU0/1，无25/2/3、温度功率读取、环境安装或抢占其他项目。
