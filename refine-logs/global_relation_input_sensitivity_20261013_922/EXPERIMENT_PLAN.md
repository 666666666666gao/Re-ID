# 真实global关系→patch：一次只读归因可行性检查

状态：源码草案；尚无实际模型调用、归因数组或性能结果。

用户再次将Goal设置为active，继续推进完整目标。当前六端训练及§919五分数诊断已经闭合，不能重启它们。这项是新问题的单best只读检查；即使通过，也不启动训练、不改mask或模型参数。原目标要求对失败结果诊断，此项不额外索取读模型授权。

## 问题与边界

§921发现角色stages是在视觉前向之后另行堆叠的，不能直接作为global的梯度取点；最后一块输出patch也不再通向CLS聚合。新的有限问题是：在原RGBNT201/uniform/E8 best的真实源训练图像上，合法global关系的梯度是否能传到第4/8/12块的真实完整输入？

这不是重跑六端区域比较、全库评分或训练M0，也不是显存/反向确定性修复。只读原best，单个B16/K2批次，FP32、eval-mode，调用共享backbone一次、批次目标VJP一次。当前角色/head/Mamba不前向；整个模型仍严格按原构造、初始化binding和best state加载，不创造新权重。

## 输入与关系

复用§918已封存且完成的输入合同，包括原465 source和90 artifact SHA、原真实退出/评价状态。新增源只包含`modeling/trifusion/global_relation_inputs_diagnostic.py`与`tools/diagnose_global_relation_inputs.py`，单独绑定SHA；原源不改。

从已登记RGBNT201训练协议选按label排序的前8个具有至少两个camera的身份；每个身份取camera排序前两个camera中路径最小的一条记录，共16。选择仅依赖合法关系和路径，不读分数；用既有eval transform，保持无擦除/随机裁剪。B16/K2是归因检查规模，不改B64/K8训练配方，也不用于推断真实训练批次覆盖率。输入顺序、label、camera和48张源图SHA记录并核对。

用全局归一化cosine构建同身份跨camera正例和异身份负例。target为margin>0的合法关系margin均值；support decision停止梯度，margin本身可微。单个批次target包含query、正例、负例作为候选时的共同梯度，不声称query单独因果贡献或类别真值saliency。没有支持时停止并保留失败，不选任意下一批次或填零mask。

## 计算与产物

在当前两卡原完整视觉图中，对块3/7/11的完整输入安装只读pre-hook，不返回替换tensor；每层捕获RGB/NIR/TIR各一次，共9个真实输入。求一个VJP；事后stacked stages也作为显式未使用输入，预期其梯度None。实际输入的None或非有限梯度不填补。

保存每个patch的signed gradient·activation和gradient norm，形状`depth, modality, batch, patch`。前者只表示**小幅缩小该patch激活时目标的一阶方向**；它不证明完全置零的影响，不是confidence/真实部件/语义互补性。记录全部9组原始统计、设备/dtype、目标支持数量、state SHA和参数grad状态。

逻辑输入总元素9×129×16×768=14,266,368，FP32约54.4MiB；9个返回梯度同规模。这不等于完整显存开销，实际峰值在两卡分别记录。没有额外9次编码/9次backward；需要一个保留正常前向图的VJP。模型初始化、原输入SHA检查可能占主要时间，预计1–3分钟。输出小于1MiB，无新checkpoint、大型距离或热图。

工程结果只判定这些真实输入有无有限敏感性，以及事后stages是否确为非祖先。state/buffer SHA必须不变，参数`.grad`全部None；不创建optimizer，不调用训练入口或官方query/gallery评价。单次实际返回0才称完成；失败保留且不自动重试。

## 执行与范围

复用2026现有`/data/gaob/Re-ID/conda-envs/tri_reid/bin/python`，现有torch2.5.1+cu121，零安装/环境重建。只允许physicalGPU0/1，同一原两卡分段。启动前实际仅检查0/1显存空闲<500MiB和至少500MiB磁盘，不读取温度/功率，不访问GPU2/3或2025。

新seal位于`refine-logs/global_relation_input_sensitivity_20261013_922/INPUT_SEAL.json`。一次调用：

```bash
CUDA_VISIBLE_DEVICES=0,1 /data/gaob/Re-ID/conda-envs/tri_reid/bin/python -B tools/diagnose_global_relation_inputs.py --seal refine-logs/global_relation_input_sensitivity_20261013_922/INPUT_SEAL.json --output-dir logs/global_relation_input_sensitivity_20261013_922
```

外部stdlib父进程记录真实命令/日志/EXIT。读原句柄至终态，不从超时重启；预计结束附近一次读取，不重复服务器NN/GPU轮询。复核先读源，终态后读实际结果；same-family/provisional，未证明的模型身份保持UNATTESTED。原六端best/依赖/Signal/距离/历史归档全部保留，只有自有传输重复ZIP可按SHA清理。

如果有有限patch敏感性，只支持后续设计可以获得真实global关系依据；不证明根据该图屏蔽后会改善检索。当前尚无证据排除候选、mask超参或新训练计划。ACoL只提供形成环节的先例，§920作者来源和边界继续适用。全三集超过Signal目标仍未达成。
