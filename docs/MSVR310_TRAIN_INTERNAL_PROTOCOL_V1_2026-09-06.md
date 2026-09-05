# MSVR310 训练内部三折数据协议 v1

记录时间：2026-09-06T02:17:57.188547+08:00。状态 DATA_PROTOCOL_FROZEN_TRAINING_CONTRACT_PENDING。
完整身份、相机、scene、三模态相对路径、source 本地分类标签和 query/gallery 清单已生成。
本协议仅冻结训练标签划分；MSVR310 模型、优化器和检索运行次数仍为 0。
RGBNT201 仍为当前训练主线；本文件没有启动车辆实验或推广尚未合格的 V24。

## 数据与固定划分规则

只使用现有 bounding_box_train 的全部 155 身份、1032 条三模态记录。
以真实 scene 标签是否至少有两个取值分成 60 个可评价身份和 95 个单 scene 身份；
两组内部各按原始身份号升序，分别按 index %3 轮流分入 held-out fold0/1/2。
该规则沿用项目 build_identity_folds 的分层轮流分折方式，将可评价字段设为 scene。
划分不使用随机种子、特征或指标；后续训练 seed42 是另一个独立设置。

每折 source 是另外两折的全部身份，重新映射到连续的本地分类标签。
不额外切出另一批固定 dev 身份；当前 155 个身份通过三折各被留出一次，
将来的完整 155-ID 训练及官方评价仍需要单独固定的训练合同。
这不会把 MSVR310 的官方 591 query/1055 gallery 改写成内部协议。

| Fold | Source 身份/记录 | Source 跨 scene 身份 | Held-out 身份/完整 gallery | Query 身份/记录 | 仅作干扰的身份/记录 |
|---|---:|---:|---:|---:|---:|
| 0 | 103/672 | 40 | 52/360 | 20/210 | 32/150 |
| 1 | 103/683 | 40 | 52/349 | 20/207 | 32/142 |
| 2 | 104/709 | 40 | 51/323 | 20/183 | 31/140 |

三折合计 1032 个独立 gallery 记录、60 个 query 身份、600 条合法 query；
95 个单 scene 身份的 432 条记录只从 query 分母排除，全部仍留在对应 fold 的 gallery。

## 检索协议与训练边界

按作者 MSVR310 evaluator 的实际公式，排除 gallery 中“同身份且同 scene”的记录。
不同身份的 gallery 均保留，包括同 scene 的负例与只有单 scene 的身份。
原 camera/v 保留给 Signal 的 camera SIE；它不是该数据集的检索过滤字段。
source、held-out 的特征和距离只在各自 fold 内使用，不能跨折计算距离。
未来需要分别训练全路径 source-only Signal 与专家，不能使用见过该 fold 留出身份的冻结语义主干。

source 三元组的模态顺序固定为 vis/ni/th，文件路径都位于 bounding_box_train；
可训练本地 ID 与原始全局 ID 同时保留，不能用重新标号后恰好相同的整数跨折合并原型。
车辆输入保持作者的高×宽 128×256 与 8×16 Patch 网格；新运行入口和真实工程检查尚未完成。
原型、缓存或新损失没有在本协议中决定；V24 的 camera 原型不自动迁移为车辆 scene 原型。

## 全量标签核验

生成器和 JSON：
- tools/build_msvr310_train_oof_protocol.py；
- protocols/msvr310_train_oof_v1.json；
- evidence/trifusion_msvr310_train_oof_protocol_verification_20260906.json。

原始标签证据 SHA256 c835d20478b817a54b7710463269186af2619cab3e38850534b01f3aaee6e3c8；
协议 SHA256 4ff4c60bca3d019929add5788212c526387d93d535a2c52aa7b1c3acfd387cb4；
生成器 SHA256 57684721cac03b1be4dfa08cee1902baeb14986bd7e4d8892ff842221d8c8143。

已独立用直接枚举核对全部 1032 个可能 query 的合法正例、被排除记录及负身份干扰数，
与生成器的计数方法相互核对；所有 source/held-out 身份及记录无交集、留出集恰好覆盖全部训练记录一次。
所有记录的 filename-derived identity/camera/scene 与原始训练清单一致，原训练与官方测试身份也保持分离。

若错误使用 same-camera 过滤，内部合法 query 会从 600 增至 1032，
且 926/1032 条记录的合法正例数量改变（各折 325/310/291）。
这是实际标签证据支持的协议差异，不是模型精度或检索结果。
核验耗时 0.089693 秒，只处理本地已归档 JSON 和源码，
没有读取图片、加载权重、计算特征/距离或运行官方评价。

后续只有绑定该协议与 source-only 初始化、固定终点和真实运行回执后，才报告 MSVR310 方法效果。
这些 60 个身份提供新的跨数据集内部验证条件，但协议数量本身不能证明泛化或模型有效。
