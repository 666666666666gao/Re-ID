# Role-set Q1：磁盘保留与终态处理准备

2026-09-08T21:05:41.528329+08:00只读普查两个指定项目目录。当前原wrapper35302和Q136320仍存在；最近训练观察2026-09-08T21:03:53.756975+08:00为3/6端完成、fold1 role_set第9/20epoch，最近epoch耗时138.567s。下次检查仍在21:25附近，预计该端约21:29完成。不是完整科学终态。

| 目录 | 空闲字节 | 空闲GiB | pt/pth/ckpt文件数 |
|---|---:|---:|---:|
| /root/autodl-tmp/trifusion-v2 | 2621513728 | 2.4415 | 304 |
| /root/trifusion-storage/artifacts | 10391728128 | 9.6781 | 30 |

空闲空间分卷报告，不相加。扩展名清单不是模型权重清单：最大文件之一是RGBNT100正式retrieval_arrays.pt（1,490,973,250B），还包含来源普查与数值诊断数组。它们属于保留证据，不能仅凭大小删除。

此次仅剩一个.resume条目：/root/autodl-tmp/trifusion-v2/artifacts/trifusion_shared_semantic_circ_urgc_v3_dev_seed42/.resume/generation-0000-epoch_boundary.pt，370928910B；目前没有独立终点可替代的证明，因此继续保留。本次删除0个文件、0字节；没有重新计入此前24个恢复权重或旧传输bundle清理量。当前输出卷高于已登记4GiB下限，无需干预训练。清单按当前文件元数据生成，没有读取模型张量或声称逐文件hash完整性检查。

终态执行入口已验证：使用现有uv离线缓存导入NumPy2.5.3/Paramiko5.0.0，并成功执行既有audit_msvr_paired_ranking_text.py的--help。旧历史实验记录中的临时Python路径已不存在；后续通过uv启动，不绑定临时路径。没有修改远端环境或训练代码。

完整六端和CPU后：严格接收所有终态文本并核对SHA；执行已在全部248个M0更新日志上检查过的source分析v2；再用现有通用排名复算器以candidate=role_set、CPU状态PASS_COMPLETE_ROLE_SET_Q1和实际终态summary SHA复算全部排名；之后进行独立终态复核。入口验证不等于这些终态工作已经执行。完整命令与范围见[终态执行入口](../evidence/role_set_disk_readiness_20260908/role_set_terminal_execution_ready_20260908.md)。

M0独立审计已封存，不能重复执行；完整三数据集Goal仍未达到。当前不读取局部科学分数，不改变原seed42、候选规则、训练预算、身份隔离或scene评价规则。
