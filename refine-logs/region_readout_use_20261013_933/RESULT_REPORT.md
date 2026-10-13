# 冻结best的实际区域读出用途诊断

本次只有两个原RGBNT201/seed42/E8 best的有界诊断，0优化器更新、0新训练。相同forward采集最终CNN/Transformer/Mamba区域；仅干预最终读出，不删除上游桥接或整个角色。原1536维静态检索接口、完整合法图库、作者camera过滤及原global/权重/gain保持。

范数控制逐样本匹配原correction范数；均值控制只去掉最终槽位差异，跨组控制只交换槽位网格的组归属，不是真实图像部件错配。两侧使用相同规则。这些是固定权重的描述性控制，不是独立训练消融、新方法或SOTA。

| 原训练条件 | 读出控制 | mAP | R1 | R5 | R10 | 相对自身G ΔmAP |
|---|---|---:|---:|---:|---:|---:|
| gradient | global | 74.296664 | 78.947368 | 88.277512 | 91.866029 | +0.000000 |
| gradient | natural | 74.364459 | 79.306220 | 88.397129 | 91.866029 | +0.067796 |
| gradient | within_group | 74.364459 | 79.306220 | 88.397129 | 91.866029 | +0.067796 |
| gradient | cross_group_same_norm | 74.362498 | 79.306220 | 88.397129 | 91.866029 | +0.065835 |
| gradient | slot_mean_same_norm | 74.364093 | 79.306220 | 88.397129 | 91.866029 | +0.067429 |
| random | global | 74.296664 | 78.947368 | 88.277512 | 91.866029 | +0.000000 |
| random | natural | 74.469545 | 79.784689 | 88.157895 | 91.866029 | +0.172881 |
| random | within_group | 74.469545 | 79.784689 | 88.157895 | 91.866029 | +0.172881 |
| random | cross_group_same_norm | 74.464091 | 79.665072 | 88.157895 | 91.866029 | +0.167427 |
| random | slot_mean_same_norm | 74.462418 | 79.665072 | 88.157895 | 91.866029 | +0.165755 |

| 条件/控制 | 对照 | ΔmAP | ΔR1 | 首位修复 | 新增错误 | 身份宏AP变化 | 最差5% query平均AP变化 |
|---|---|---:|---:|---:|---:|---:|---:|
| gradient/natural | versus_global | +0.067796 | +0.358852 | 5 | 2 | +0.064679 | -2.145016 |
| gradient/natural | versus_natural | +0.000000 | +0.000000 | 0 | 0 | +0.000000 | +0.000000 |
| gradient/within_group | versus_global | +0.067796 | +0.358852 | 5 | 2 | +0.064679 | -2.145016 |
| gradient/within_group | versus_natural | +0.000000 | +0.000000 | 0 | 0 | +0.000000 | +0.000000 |
| gradient/cross_group_same_norm | versus_global | +0.065835 | +0.358852 | 5 | 2 | +0.062882 | -2.184730 |
| gradient/cross_group_same_norm | versus_natural | -0.001961 | +0.000000 | 0 | 0 | -0.001797 | -0.110105 |
| gradient/slot_mean_same_norm | versus_global | +0.067429 | +0.358852 | 5 | 2 | +0.064456 | -2.164853 |
| gradient/slot_mean_same_norm | versus_natural | -0.000366 | +0.000000 | 0 | 0 | -0.000223 | -0.059942 |
| random/natural | versus_global | +0.172881 | +0.837321 | 11 | 4 | +0.138888 | -4.553729 |
| random/natural | versus_natural | +0.000000 | +0.000000 | 0 | 0 | +0.000000 | +0.000000 |
| random/within_group | versus_global | +0.172881 | +0.837321 | 11 | 4 | +0.138888 | -4.553729 |
| random/within_group | versus_natural | +0.000000 | +0.000000 | 0 | 0 | +0.000000 | +0.000000 |
| random/cross_group_same_norm | versus_global | +0.167427 | +0.717703 | 10 | 4 | +0.131881 | -4.576001 |
| random/cross_group_same_norm | versus_natural | -0.005454 | -0.119617 | 0 | 1 | -0.007007 | -0.416963 |
| random/slot_mean_same_norm | versus_global | +0.165755 | +0.717703 | 10 | 4 | +0.131673 | -4.554640 |
| random/slot_mean_same_norm | versus_natural | -0.007126 | -0.119617 | 0 | 1 | -0.007215 | -0.304918 |

上述检索指标及其变化均为百分数或百分点；身份宏平均不替代训练种子不确定性。完整逐query多正例AP、首位分组损益、身份变化及分位数保留在RESULT.json；只取首位统计无法代替mAP。下表相对L2变化为比值、余弦无单位、最大绝对变化为原c坐标单位。

| 条件/split/控制 | c相对L2变化均值 | c与natural余弦均值 | c最大绝对变化上界 |
|---|---:|---:|---:|
| gradient/query/within_group | 5.76469787e-08 | 1 | 2.38418579e-07 |
| gradient/query/cross_group_same_norm | 0.0660991743 | 0.997772872 | 0.0707614571 |
| gradient/query/slot_mean_same_norm | 0.0373132713 | 0.999290943 | 0.0394101441 |
| gradient/gallery/within_group | 5.76469787e-08 | 1 | 2.38418579e-07 |
| gradient/gallery/cross_group_same_norm | 0.0660991743 | 0.997772872 | 0.0707614571 |
| gradient/gallery/slot_mean_same_norm | 0.0373132713 | 0.999290943 | 0.0394101441 |
| random/query/within_group | 5.80681423e-08 | 1 | 1.78813934e-07 |
| random/query/cross_group_same_norm | 0.168835893 | 0.982985437 | 0.335495681 |
| random/query/slot_mean_same_norm | 0.0919507593 | 0.995111763 | 0.187767088 |
| random/gallery/within_group | 5.80681423e-08 | 1 | 1.78813934e-07 |
| random/gallery/cross_group_same_norm | 0.168835893 | 0.982985437 | 0.335495681 |
| random/gallery/slot_mean_same_norm | 0.0919507593 | 0.995111763 | 0.187767088 |

执行一致性：
- gradient: E8固定；原g/c/h/f缓存最大差 0；原F距离最大差 0；组内置换距离最大差 7.15255737e-07；Q/G编码batch 14/14；诊断内部计时 48.635s（含构造/初始哈希/编码/保存/评分，截至结果对象生成；之后的产物哈希/末次输入封存/退出不计入）。
- random: E8固定；原g/c/h/f缓存最大差 0；原F距离最大差 0；组内置换距离最大差 7.15255737e-07；Q/G编码batch 14/14；诊断内部计时 48.736s（含构造/初始哈希/编码/保存/评分，截至结果对象生成；之后的产物哈希/末次输入封存/退出不计入）。

同一次实际forward的区域/g/c/h/f、原读出weight/gain子集及五套距离已保存并封存SHA。weight子集仅为原best诊断副本，没有产生新的训练权重。参数/buffer前后摘要相同、grad始终None、推理exclusion None；原472源码与输入在执行前后保持SHA。仅2026物理GPU0、1，一项两卡NN顺序执行；复用已有conda环境，无安装或重建，无功率温度读取。

各诊断耗时是该次共享服务器进程观测，不能作为孤立模块速度、完整训练墙钟或跨主机时间比较。原CPU/NN历史失败与正式first-strict不覆盖；本项不重复旧MaxSim/拼接，不调gain/seed/loss/margin，不恢复训练。

实际结论：组内交换全部指标保持。跨组交换的c相对L2变化均值为gradient 6.6099%、random 16.8836%，但mAP仅下降0.0019608/0.0054543点。槽位均值控制的c相对L2变化均值为3.7313%/9.1951%，mAP仅下降0.0003662/0.0071264点；gradient全部CMC不变，random的R1少一个query、R5/R10不变。因此不是控制完全没有活动，而是这些最终槽位/组变化在当前固定读出下没有形成明显的检索差距。自然结构仍有小的描述性优势，不将它改写成严格等价、统计无效或“所有槽位都无用”。不据此直接增加更复杂的槽位组织、匹配器或晚交互。

结论边界：natural超过破坏控制，只能说明当前最终组归属/槽位差异在这个冻结模型中有用途，不能证明原来存在错配、新增内容独立于global、整条角色必要、读出压缩为失分主因或跨数据集泛化。最终均值控制仍保留上游桥接已经混合的信息，并不是从模型中删除空间信息。此次证据只覆盖原RGBNT201两个seed42/E8 best，不代替跨数据集或训练流程重复。

审查范围仅为源码、保存的结果JSON和文字主张核对，same-family/provisional；实际审查模型/effort未获得运行证明（UNATTESTED）。审查者没有独立读取远端原始张量或重放NN，不能将该复核称为跨家族复现或独立数组验算。

后续研究继续以“每条新增证据在最终身份比较中确实有用”为问题；不根据破坏控制造贡献，不预设新匹配器/教师/三个模块能涨分。原gradient−random未晋级，三数据集超过完整Signal的Goal仍ACTIVE_UNMET；当前训练停止。
