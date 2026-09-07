# Signal 公开配置与本机车辆基线的边界核对（2026-09-07）

本次仅做文本/代码核对，0模型前向、0新训练、0官方测试读取，不改变本次MSVR实例记忆Q1合同或已封存结果。远端作者源代码固定cd1b0a672d1fe642e7608731cb4899a19dda7d51；读取的车辆YAML SHA逐项等于本机基线登记合同。证据见evidence/signal_public_budget_comparison_20260907/source_binding_and_paper_record.json。

| 项目 | 作者发布YAML | 本机Signal基线合同 |
|---|---|---|
| MSVR310 batch / 每身份图像 | 64 / 4 | 64 / 8 |
| MSVR310 epoch | 50 | 固定50 |
| RGBNT100 batch / 每身份图像 | 128 / 16 | 64 / 8 |
| RGBNT100 epoch | 30 | 固定30 |
| 车辆输入高×宽 | 128×256 | 128×256 |

作者配置：[MSVR310固定提交](https://github.com/010129/Signal/blob/cd1b0a672d1fe642e7608731cb4899a19dda7d51/configs/MSVR310/Signal.yml)、[RGBNT100固定提交](https://github.com/010129/Signal/blob/cd1b0a672d1fe642e7608731cb4899a19dda7d51/configs/RGBNT100/Signal.yml)。项目对应configs/MSVR310/Signal-source-oof-v1.json、configs/RGBNT100/Signal-source-oof-v1-r2.json；已完成RGBNT100官方训练的Signal-main-v1.json继续使用30epoch、B64/K8。

[AAAI正式论文](https://ojs.aaai.org/index.php/AAAI/article/download/37674/41636) PDF第5页的实现段落同样列出MSVR310 B64/K4、RGBNT100 B128/K16，但统一写50epoch。该文字与RGBNT100发布YAML的30epoch不同；不能擅自合成为完全一致的“作者实际训练合同”，也不能单凭差异判定哪一份造成项目性能落后。此次PDF内容由解析器核对，第5页截图请求超时，不声称完成视觉渲染复核。

同一AAAI论文第6页表2的MSVR310是 **53.6 mAP / 71.9 Rank1**；[作者README模型表](https://github.com/010129/Signal/blob/cd1b0a672d1fe642e7608731cb4899a19dda7d51/README.md)则是 **53.2 / 72.4**。过去采用53.2/72.4的记录保留其“作者发布模型表”来源；以后写“论文表2”时应采用53.6/71.9并保留版本区别，不能混用或把差值当作本机训练变化。

本机Signal与TriFusion在已登记条件内的配对比较仍可解释；但本机baseline不是作者训练设置的逐项完全复现。B/K变化会同时改变每批身份数、每身份正例覆盖、更新次数和损失统计，不能只解释成显存差异。RGBNT100仍是每批8身份，但每身份图像及batch减半；MSVR310从每批16身份变为8身份。理论上的计数变化不等于已证明的泛化因果。

此前稳定Gram开方和冻结无实际更新的token selection等工程差异继续沿用已有披露。本次不新增借口、算法主张或补救式调参；后续正式资源比较应分别列作者报告、本机固定baseline及本机方法，完整列B/K、epoch、初始化、硬件、优化器/增强和checkpoint选择。当前六端Q1合同/门槛完全保持。
