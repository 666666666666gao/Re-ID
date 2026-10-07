# N1：语义引导的原生细节来源

2026-10-02。仅实施N1及其参数量匹配控制。前序干净公开CLIP六端已完成、J1 FAIL2/3、fresh审计WARN；不重跑前序以挽救分数。

## 唯一机制与控制

现有CNN读CLIP语义patch。N1保留该路径作为attention key，以同一预处理、增强后的图像构造value：共享3层3×3 stride2 CNN，3→32→64→128，每层GELU，93,248参数，无新归一化或辅助损失。CNN语义key插值到32×16（车辆16×32）并经已有key投影；query、value/output投影、桥接、Transformer、Mamba、区域读出沿用原实现。

high直接读取256×128/128×256输入；low先平均降采样2倍，使用完全相同stem，再将16×8/8×16输出双线性插值到相同512候选格。两者全部初始张量一致、候选数/attention/readout/头/1536输出一致。low插值不恢复丢失细节。high只指预处理图的较密特征，不宣称原传感器分辨率。

原CNN卷积、跨层role0投影继续形成key，保持真实梯度路径。value来自原生细节。其余角色仍读取原128语义patch。保留原static global token与context query；原sample_context忽略位置，故槽位并非已证实的语义对应。本阶段不实现N2共享/私有、N3 BIN/meta、外部文本/SAM/DINO或新排序目标。

## 同配方与既有基线

复用已完成clean_clip_joint_20261002_v1的公开ViT-B/16、新相机/头/适配器初始化；未加载任何ReID训练权重。先验证所有非stem张量与已完成roles构造逐张量一致，公共backbone/neck/classifier与原初始化回执一致。high/low所有张量逐项一致。新stem在独立RNG范围构造，不消耗训练采样/增强RNG；两端正式训练独立全新构造，M0权重不复用。

三数据集×high/low，共6个正式端，seed42。视觉FP32参数5e-6，其余3.5e-4；AdamW wd1e-4，五轮warmup+50轮cos；B64/K8、原PLAIN_V8采样和增强、CE smoothing0.1+Triplet margin0.3。不开M3/localID/统计扰动/记忆库。

每端完整50轮，最高官方fused mAP一份权重、并列取较晚轮；全部CMC随该权重；完整state保存与严格重载；真实GT与完整图库、原camera/时间段过滤、干扰身份均保留，无rerank。官方集已参与研发与选点，单seed不构成稳定性或无偏显著性。

前序六端原始结果作为同配方global-only及roles控制，既有SUMMARY SHA784e3938a0203ea043543804c6e2933b43877405d7ca624116d59e59d35538fd；不得以旧69.6415作为新固定分母。N1比旧roles多93,248参数，报告额外成本；high−low是参数与候选数量匹配的分辨率对照，仍有输入计算成本差。

## 事前判定

N1-A：high−原roles三集mAP>0、R1>=0，201及MSVR至少+0.5mAP。N1-B：high−low三集mAP>0、R1>=0。两者均通过才支持“原生更细信息提供稳定角色增量”的本阶段主张；若仅A通过，容量/来源改变与分辨率贡献仍需区分。此门不替代完整baseline/SOTA目标，也不声称各个角色必要。

六端均收齐，失败不改门、不追加种子、系数或重跑救分。报告首位修复/新增错误、身份宏平均、全50轮轨迹、参数/训练时间/峰值显存。N2最近邻MDReID/DeMo/MODAL和N3实际跨环境支持留后续，不把此次N1结果归给它们。

## 执行

fresh gpt-6-astra/max代码复核（same-family/provisional）→三数据集原roles/high/low实际构造比较→RGBNT201 high/low各8个真实batch M0→六端登记→其他4端各自M0后正式50轮→一次完整CPU报告+fresh结果审计。

最多四张经实际检查的健康空闲卡，memory<500MiB；调度轮询240秒，无自动retry，实际失败停止新增启动、保留所有失败并等待已启动端退出。预计8–12 GPU小时、四卡约3–5小时；模型/距离留远端，约5–8GiB，当前空间足够，不清理前序证据。

本计划是候选方法与固定对照；未得到新成绩，不承诺+10mAP。完整目标ACTIVE/UNMET。
