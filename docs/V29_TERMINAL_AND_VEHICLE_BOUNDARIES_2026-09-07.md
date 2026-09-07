# V29终态接收与车辆扩展边界

更新 2026-09-07T12:59:21.075742+08:00。这是执行准备和既有配置核对，不是新的训练合同或已完成跨数据集验证。

## 当前唯一在运行的主实验

RGBNT201 V29实际代码f4c6a03e1b263aaa9e4bce71427152007018a0ca，原训练153151/wrapper153149。M0全13项通过。12:54:14第一折完整两端已写入摘要，后续fold1 control第7轮完成。全六端终态、CPU完整排名核验与最终科学判定仍待完成。

12:55:19额外核对首折存储：两端各580步、20轮、1000图库/190合法query/五输出，最终权重、整个检索数组、完整训练日志均逐字节SHA匹配。两份权重合计65,171,960B，两份数组251,852,276B；严格重载/只读评估由实际原训练回执确认。本项没有重载模型或复算部分检索成绩。磁盘可用13,877,886,976B，原PID均存活，新删除0。

## 完整终态接收与复算

原pipeline会完成所有3360训练步骤、六个终点、32,602,260距离值、5,952,790排名位置、571合法query和全部21身份的CPU核验与报告，保留原五科学门。

本地补充脚本已经准备并通过AST/F821检查，接收完整终态后再执行：

- tools/verify_v29_terminal_scalars_local.py：全部3360步七组损失重新合计、五输出AP/CMC及身份bootstrap、五科学门；另外聚合全部训练1,935,360及完整检索56,268槽位观测的保存几何统计。
- tools/summarize_v29_terminal_outcomes.py：全21身份净贡献、所有新增与修复错误、同camera关联、全训练与固定epoch20的损失和成本。只作完整描述，不调整条件。

本轮仅做语法/CLI准备，没有在不完整V29数据上执行这些终态脚本。几何统计复算是保存的全批标量复算，不冒充原始向量重放；本地不加载Torch模型、图片或特征数组。原远端核验和本地复算均不冒充外部独立审计。

## 车辆接口需要单独绑定

| 条件 | 当前RGBNT201 V29 | 已完成的车辆路径 | 后续要求 |
| --- | --- | --- | --- |
| 输入/语义网格 | 256×128 /16×8 | 128×256 /8×16 | 保留车辆纵横比和角色空间布局 |
| style stem检查 | source_style_v27.py:82硬编码768×16×8 | 实际768×8×16 | 新合同中明确网格；不能只换数据集名称 |
| Signal/角色起点 | 来源合法V12 Signal及已训练角色 | 既有车辆角色实验从各fold Signal建立fresh角色 | 新比较两端绑定同一车辆起点和训练历史 |
| 采样 | CrossCameraIdentitySampler，完整已登记曝光 | 车辆loader使用原RandomIdentitySampler | 单独重放实际batch/身份/供体支持，不能套用RGBNT201清单 |
| 评价过滤 | 同身份AND同camera | RGBNT100用camera，MSVR310用scene字段 | 复用各自已核验loader/evaluator，不将camera与scene混用 |
| 推理数值 | 当前行人原路径 | 车辆保留exact_signal_forward执行路径 | 再核对嵌入Signal与独立Signal全记录相等 |

依据：modeling/trifusion/source_style_v27.py、signal_preserving_v8_builder.py；
configs/RGBNT100/TriFusion-source-oof-v1.json；
tools/train_rgbnt100_trifusion_oof.py、train_rgbnt100_signal_oof.py、
train_msvr310_signal_oof.py、msvr310_exact_signal_inference.py。

MixStyle公式对两个空间轴一起求均值/方差，因此统计公式本身不依赖16×8方向；实际阻断在明确的shape断言，CNN/Mamba空间布局又有各自网格语义。不能把取消断言当作完成车辆适配。MSVR310 records_for同时保留camera和scene，scene_scores按scene排除同身份样本；来源扰动供体所使用的环境字段也必须单独明确，不把不同摄像头直接标成负身份。

上述是可执行接口约束，车辆新训练尚未登记或启动。V29当前源码及合同保持冻结；是否进入下一项完整车辆比较，依据完整主结果和新登记合同决定。

## 本机Signal与作者成绩的已有差异

RGBNT100官方报告已经封存：本机Signal80.712162、fused83.284770、Mamba83.440622。作者Signal表为86.3。原报告已核对作者B128/K16/workers12、三模态独立随机增强和每轮评估选优；本项目B64/K8/workers4、共享几何、固定Signal epoch30，另有已披露的Gram稳定开方定义。差异同时存在，当前没有配对证据可将5.587838pp差距归因于某一项。

完整既有依据见results/TRIFUSION_RGBNT100_OFFICIAL_COMPARISON_2026-09-06.md。本次没有新跑作者复现或重新核其网页，不把旧记录称为新实测。官方分数已消费，不依此扫描batch、增强、轮次或融合权重。三任务独立训练的同源基线增益、作者报告差距及额外预训练资源需继续分别解释；整体SOTA目标未达。
