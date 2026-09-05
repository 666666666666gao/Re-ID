# V23 experiment tracker

更新时间：2026-09-05T22:14:07.866047+08:00；状态M0_PASS_COMPLETE_Q1_RUNNING。

| ID | 内容 | 状态 | 证据 |
|---|---|---|---|
| V23-SOURCE | ICPL真实入口与移植边界 | DONE_SOURCE_ONLY | docs/SPECTRAL_ADAPTER_SOURCE_NOTES_2026-09-05.md |
| V23-T0 | 五项远端CUDA合成契约 | DONE_PASS | evidence/trifusion_v23_t0_20260905.json |
| V23-M0 | 完整54前向/116优化步 | DONE_PASS | evidence/trifusion_v23_m0_seed42_9f4a10b.json |
| V23-M0-ARRAYS | 全配对、116步分量/固定门 | DONE_PASS | evidence/trifusion_v23_m0_array_verification_20260905.json |
| V23-M0-FILES | 29全文件SHA/执行Git绑定 | DONE_PASS | evidence/trifusion_v23_m0_file_verification_20260905.json |
| V23-M0-AUDIT | 独立完整M0审计 | PENDING | 准备只读原始文件请求 |
| V23-Q1 | 三fold两端20epoch完整五路检索 | RUNNING_ORIGINAL_PID44684 | 22:07仅control两epoch/58步，无完整端点 |
| V23-D1 | 141-fit refit与固定30-dev | NOT_QUALIFIED_NOT_RUN | 科学资格未产生 |

M0真实完整结束并通过：耗时240.083842516秒，
三fold两端共48次source配对前向，另6次原encoder只读对照（共54前向），
fold0两端各8不同batch更新及fresh候选固定首batch100步，共116项目优化步。
六模型初始state、source state、每端8batch增强/路径/五输出SHA精确配对；
零适配器与原encoder五输出在六个第一batch上精确相同。
所有94-source与47-heldout身份隔离；baseline和冻结尾部保持不变。

两端总参数实际100577677；控制/候选trainable7841292/9618828，
训练tensor203/239，恰差9个模态MLP/1777536参数/36tensor。
两个8步容量的非零梯度覆盖203/203和239/239，peak6090/6494MiB，overflow0；
固定100步候选239/239覆盖、冻结state不变、overflow0。
L1=0.6110473871231079，L100=0.5803323984146118，
F=0.5783829210462100，
固定excess比0.05968189909525461<=0.1，通过全部五项M0工程门。

本地JSON/NumPy重算全部116步各14原始loss分量及加权总和、全部配对和固定门，
最大分量舍入差8.9406967163085938e-08；原始log中的M0对象逐项相同。
远端29项source/config/plan/CLIP/V12/source权重全文件SHA一致，无新增模型或检索。
preregistration完整等于执行commit，所有执行前10份工件按Git blob精确复核；
latest tracker因如实更新状态而不用于当前字节等于旧prereg的声明。

22:07:16实际观测原PID44684仍在运行，已自动进入Q1且仅记录第一折control前2epoch，
58个Q1更新；完整paired fold0、端点receipt、heldout AP/Rank和科学门尚不存在。
M0快照343499字节 SHAb662bf6420f7f5dc6f92fd0d93d00b2eabfdb3aee4e70af34d6946589b587a2a；
对应log83703字节 SHA48303a40fd7ff82e4a0afa646729e4b7bf2a7f3e3b3e29f2586bcf0ea83029bf。
这是含明确非终态训练日志的source-only工程证据，不是检索改进。
独立M0审计待完成；M0通过不改变五项Q1科学门，不允许D1/dev/official。
