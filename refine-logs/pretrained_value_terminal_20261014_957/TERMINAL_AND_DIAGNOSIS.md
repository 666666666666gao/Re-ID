# §41.957 两个预训练value读取端闭合及实际用途诊断

原队列已在2026-10-14 05:38:56完成，supervisor EXIT0；08:43只读收集确认两端均完成prepare、8步M0、50轮及首次严格重载，原controller/supervisor不存在。原训练没有重启。

| 条件 | best epoch | mAP | R1 | R5 | R10 |
|---|---:|---:|---:|---:|---:|
| raw_semantic | 8 | 74.3749 | 78.8278 | 88.3971 | 91.7464 |
| raw_global_only | 8 | 74.2967 | 78.9474 | 88.2775 | 91.8660 |
| semantic_reference | 8 | 74.3111 | 78.9474 | 88.1579 | 91.8660 |
| uniform_reference | 8 | 74.2974 | 78.8278 | 88.1579 | 91.8660 |

完整Signal本地参照80.3029/85.1675，初始化及系统不同，不能当作单因素对照。当前两端未超过Signal，seed42保留，尚不进入多seed或超参数搜索。

一次CPU配对报告完成，两个新端与raw semantic/global-only训练batch顺序逐字节相同；完整50轮曲线、读取/任务曲线、首位修复/新增错误和身份AP均记录在SUMMARY.json。

两个固定best各编码一次完整836 query/836 gallery，state前后相同、grad空、optimizer更新0，重新提取F距离与首次正式距离最大差均0。CPU报告14.000秒；两次NN诊断56.605/54.891秒。没有重训、改best或新正式方法成绩。

| 诊断 | semantic reference | uniform reference |
|---|---:|---:|
| own-G mAP | 74.2967 | 74.2967 |
| correction独立mAP | 39.6920 | 37.9780 |
| value每图patch均值独立mAP | 73.3979 | 73.3979 |
| query修正/G范数比均值 | 0.007757 | 0.007787 |
| query读取变化L1均值 | 1.119761 | 1.211295 |

T/M读取变化明显，不能概括为所有读取接近uniform或没有工作。semantic-reference的CNN变化较弱（L1约0.060–0.068），但这不是独有根因证据。Q/K权重范数保留初始约0.298–0.441，不支持仅据范数就归因weight decay；未测任务梯度与decay比例。

每张图内部patch偏差占value能量约2.40%/1.67%/3.07%（RGB/NIR/TIR）；这是能量统计，不是噪声或身份信息比例。两端零行和差分会在固定读取权重下抵消每图公共value。该均值自身具备73.3979的检索能力，但尚未证明它在G之外具有净互补，也未证明抵消它是当前失败的唯一原因。

下一项最小假设：保持原value、三角色、初始化、训练配方和gain不变，检验保留读取分布的公共分量，而不强制所有增量为零和。以当前uniform-reference作为完整已闭合同参数对照；不扫描倍率、不改seed、不使用test均值校准。新的读取若仍无净增量，应否定该假设，不能把活动/独立mAP当作机制成功。

历史完整Signal+R2 RGBNT201为83.4284/87.6794，使用发布Signal起点；它与当前公共CLIP起点不同，不归为当前模块从零训练的结果。已有强best继续保留。

参考调查：[QKNorm论文](https://aclanthology.org/2020.findings-emnlp.379/)与[Apache-2.0作者代码](https://github.com/CyndxAI/QKNorm)，commit c628cb5d21f1475ba95db779a175748ff9efe940。其Q/K单位化是已有机制，不能作为本项目原创。由于当前实际读取变化明显，本轮不采用“修复uniform”的移植；GitHub API限流失败记录保留，未导入/安装该仓库。

原496训练源码和best/M0依赖保持不变。诊断特征/距离已在服务器保存，唯一数据和最佳权重保留。后续消费者完成前不清理M0依赖。三数据集全面超过Signal目标ACTIVE_UNMET；仅2026物理GPU0/1，无温度/功率读取、车辆扩展或环境变化。
