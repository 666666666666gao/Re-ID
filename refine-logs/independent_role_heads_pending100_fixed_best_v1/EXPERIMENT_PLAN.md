# 固定best诊断：仅续接尚未前向的RGBNT100

原三端队列在2026-10-07 03:52退出1。201/MSVR各完成一次全query/gallery诊断，RGBNT100在入口磁盘断言处停止，未进入main、构造模型或读取query/gallery。原FAILED campaign、退出回执、100失败日志和两个完成结果永久保留，不重跑两个完成端。

19份已闭合、非最佳的RGBNT100末轮临时距离缓存经过full50/首次strict回执、完整曲线、最佳/正式距离SHA及当前241依赖排除核验后退役，保留全部PTH和最佳/正式距离。末轮aggregate指标仍在完整曲线中；这些旧末轮query级数组不再可重放。清理未改原388源或241封存输入。

在新的独立输出和启动目录中只运行：

```text
/data/gaob/Re-ID/conda-envs/tri_reid/bin/python -B tools/diagnose_independent_role_heads_best.py
  --campaign logs/independent_role_heads_v1_20261006_873
  --seal refine-logs/independent_role_heads_fixed_best_diagnosis_v1/INPUT_SEAL.json
  --output-dir results/independent_role_heads_fixed_best_pending100_20261007_876
  --dataset RGBNT100 --variant semantic
```

该命令调用同一已封存入口的单端路径。部署启动器先创建新的空输出父目录；不调用三端coordinate。固定best26、1715完整query/8575完整gallery、原camera过滤、同一FP32 eval、10300记录曝光，0optimizer更新。既有评分1e-5和公式1e-6检查不变，保存三个完整距离及逐样本标量统计，不生成完整特征缓存。

启动前验证原失败及两个完成DIAGNOSIS的固定SHA、原388源/241输入、三端正式训练best和封存公开CLIP；仅26物理GPU0/1空闲，max1，已有环境，启动磁盘要求仍为2415919104B。不得降低预留、删除当前best/作者权重、调整gain/LR/margin/seed、重跑已完成NN或原六配对报告。

预计4–6分钟，首次观察在启动后5分钟，后续240秒。原进程确实终态且没有活跃的本项目NN后才发布和续接。新失败原样保留，不盲重启；所有运行期间禁止源码/Git同步。100完成后，接收这一单端并与已核验两端合并为完整描述性报告；最终三端证据含原失败和独立续接，不能称原队列一次全成功。

这是行政续接与存储清理，不是新模型、训练或算法贡献。原独立头正式0/3晋级、0/6配对结论不变；官方已选best的描述性诊断不能证明因果或多种子稳定性。主目标ACTIVE/UNMET。
