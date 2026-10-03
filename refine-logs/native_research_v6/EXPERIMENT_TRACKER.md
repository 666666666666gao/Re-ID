# 逐端研究 V6 执行表

2026-10-03：用户选择第1项；四入口实现、fresh同家族源码/AST复核及五处文档同步完成。22:27:14在26物理GPU0/1启动持久新队列。旧FAIL/STOP不改判；新运行事实见以下追加。

| 数据集 | global_only | semantic | native |
|---|---|---|---|
| RGBNT201 | m0:COMPLETE/full:PENDING | m0:PENDING/full:PENDING | m0:PENDING/full:PENDING |
| MSVR310 | m0:PENDING/full:PENDING | m0:PENDING/full:PENDING | m0:PENDING/full:PENDING |
| RGBNT100 | m0:PENDING/full:PENDING | m0:PENDING/full:PENDING | m0:PENDING/full:PENDING |

实际观察时间：2026-10-03T22:30:38.855231+08:00；活动模式：train；PID：3863278。活动进程通过/proc核实，不能由PENDING表格推断停止。M0完成回执：

[
  {
    "dataset": "RGBNT201",
    "variant": "global_only",
    "status": "M0_PASS",
    "steps": 8,
    "reload_max_abs_difference": 0.0,
    "production_updates": 8,
    "author_bn_batches_tracked": {
      "bottleneck": 8
    }
  }
]

每端M0后fresh50，从同一初始化重新开始；正式只保存一份mAP-best。B128容量和剩余条件尚待真实检查。未产生完整新检索结果，Goal active/unmet。
