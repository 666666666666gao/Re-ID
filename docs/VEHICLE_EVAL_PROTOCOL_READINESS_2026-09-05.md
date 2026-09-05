# 车辆数据集标签与评价协议检查（2026-09-05）

本次在服务器读取全部split文件名与身份/camera/scene标签，未加载图像、权重或
特征，未训练或计算检索指标。脚本为`tools/audit_vehicle_query_protocol_labels.py`，
完整split清单和逐query正例/过滤数量见
`evidence/vehicle_query_protocol_labels_20260905.json`。

| 数据集 | train记录/身份 | gallery记录/身份 | query记录/身份 | 有效query | 保留正例范围 |
|---|---:|---:|---:|---:|---:|
| MSVR310 | 1032/155 | 1055/155 | 591/52 | 591 | 1–31 |
| RGBNT100 | 8675/50 | 8575/50 | 1715/50 | 1715 | 50–175 |

两者train/test身份集合不相交，全部query路径均属于对应gallery。
所有不同身份图库项均保留；query中未出现的图库身份不能被删掉。

MSVR310 loader `data/datasets/msvr310.py`将文件名`[6:9]`解析为scene/time标签，
`[11]`为camera。现有`utils/metrics.py::eval_func_msrv`剔除的是**同身份且同scene**
的图库项，`engine/processor.py`为MSVR310选择该评价器。
[NEXT v5 §V-A](https://arxiv.org/html/2505.20001v5)也明确说明按同身份/同时间跨度
过滤。591条query中，516条在只按同camera过滤时正例数量会改变，因此不能
直接搬用RGBNT201的同camera过滤规则后声称相同官方协议。

RGBNT100现有`data/datasets/RGBNT100.py`使用拼接的`rgbir`目录，camera从1–8
转换为0–7；通过`R1_mAP_eval`进入`utils/reid_evaluation.py::evaluate_reid`，
剔除同身份且同camera图库项。标签检查按这一规则计数，不修改评价器。

脚本SHA256：`6a936532f693832e76f479e381c7204129b87e59347423529087fa8ad7c9a772`。
MSVR310完整标签清单SHA为
`d53b191bf78624294cfdd9a4cbdef0d7eb68a2b5214ccf4e38418a26f7fa95f9`；
RGBNT100为`e5572d9a3b4e7c6a3e7b1801145d31ad1ed1904d1c3019a5efc2ed4f6899816c`。

这是数据协议准备，不是评价器数值重放或跨数据集实验完成。V19固定runner仅
实现RGBNT201身份OOF，当前不能直接用于车辆数据；通过其主实验门后，车辆
主实验仍需独立固定训练/终态选择合同，并核验真实评价调用与上述过滤一致。
RGBNT100/MSVR310当前训练次数和检索评价次数均为0。
