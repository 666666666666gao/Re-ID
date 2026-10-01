# 视觉更新匹配对照执行表

2026-10-01 12:47准备稿。以下均未注册、未执行M0或训练；先等待原视觉起点六端完整报告与代码审查。

| Run ID | Dataset | Visual update | Readout | Priority | Status |
|---|---|---|---|---|---|
| VU01 | RGBNT201 | frozen | global_only | MUST | PREPARATION_ONLY |
| VU02 | RGBNT201 | low_lr | global_only | MUST | PREPARATION_ONLY |
| VU03 | RGBNT201 | frozen | roles | MUST | PREPARATION_ONLY |
| VU04 | RGBNT201 | low_lr | roles | MUST | PREPARATION_ONLY |
| VU05 | RGBNT100 | frozen | global_only | MUST | PREPARATION_ONLY |
| VU06 | RGBNT100 | low_lr | global_only | MUST | PREPARATION_ONLY |
| VU07 | RGBNT100 | frozen | roles | MUST | PREPARATION_ONLY |
| VU08 | RGBNT100 | low_lr | roles | MUST | PREPARATION_ONLY |
| VU09 | MSVR310 | frozen | global_only | MUST | PREPARATION_ONLY |
| VU10 | MSVR310 | low_lr | global_only | MUST | PREPARATION_ONLY |
| VU11 | MSVR310 | frozen | roles | MUST | PREPARATION_ONLY |
| VU12 | MSVR310 | low_lr | roles | MUST | PREPARATION_ONLY |

全部seed42/50轮/同一最高mAP权重；未完成不填正式分数。当前入口已编写，语法、独立审查、生产梯度与重载测试待执行。正式队列尚未实现。
