# MSVR310 Signal B0 终态只读核验

登记时间：2026-09-06T05:07:58.159629+08:00，正式B0仍运行，未读取完整终态。本核验不改绑定训练计划、模型、配置、loss或科学门槛。

1. 原wrapper exit0及三个50epoch最终结果存在后，执行tools/verify_msvr310_signal_terminal_files.py。
2. 在远端逐完整文件核对三个最终checkpoint、三份retrieval_arrays、独立training/receipt与summary，
   原config、7份项目输入、17份Signal源码、固定CLIP权重及M0门回执。记录核验脚本SHA及执行commit。
3. 仅读取已有特征和距离文件，不加载模型、不读图、不反传、不更新或写checkpoint。
   由保存的3072D特征重算距离，完整报告最大绝对差及逐元素相等性，不追改任何训练门槛。
4. 从原保存距离排序，导出全部600query的完整gallery索引序列，保留1032gallery/155身份；
   导出的是离散索引和标签，checkpoint、特征和距离张量不传本地。
5. 本地tools/verify_msvr310_signal_terminal_arrays.py从原真实ID/scene标签和全部排序重算逐query AP与Rank。
   核对完整query资格、真实跨scene正例数、单scene干扰身份、source/heldout分离、600query加权总指标；
   逐步核对实际B64/K8曝光、真实source索引、全部150epoch日志、20/40调度及M0独立初始化状态。
6. 记录原AMP分量重组差，不补造原未记录的dtype；不把代码算术核验等同于独立审计或新方法晋级。
7. 完整证据及执行侧报告准备后，按experiment-audit交给独立审阅者直接读文件；保留请求、原回复和限制。
   本验证不追加新的图像评估、不选择epoch、不以低mAP重训。

两份验证脚本仅本地AST检查；尚未执行终态核验。原训练执行仍bb01d60。
