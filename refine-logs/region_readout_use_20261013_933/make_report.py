"""Render the completed frozen-readout receipts; no NN or rescoring."""
import hashlib
import json
from pathlib import Path

w=Path(__file__).resolve().parent
value=json.loads((w/'RESULT.json').read_text())
assert value['exit']['status']=='TWO_FROZEN_BEST_DIAGNOSES_COMPLETE'
assert all(row['exit_code']==0 for row in value['exit']['exits'])
metrics=('mAP','Rank-1','Rank-5','Rank-10')
lines=['# 冻结best的实际区域读出用途诊断', '',
    '本次只有两个原RGBNT201/seed42/E8 best的有界诊断，0优化器更新、0新训练。相同forward采集最终CNN/Transformer/Mamba区域；仅干预最终读出，不删除上游桥接或整个角色。原1536维静态检索接口、完整合法图库、作者camera过滤及原global/权重/gain保持。', '',
    '范数控制逐样本匹配原correction范数；均值控制只去掉最终槽位差异，跨组控制只交换槽位网格的组归属，不是真实图像部件错配。两侧使用相同规则。这些是固定权重的描述性控制，不是独立训练消融、新方法或SOTA。', '',
    '| 原训练条件 | 读出控制 | mAP | R1 | R5 | R10 | 相对自身G ΔmAP |',
    '|---|---|---:|---:|---:|---:|---:|']
rows=[]
for arm,d in value['diagnoses'].items():
    assert d['status']=='COMPLETE' and d['optimizer_updates']==0 and not d['new_training']
    assert d['model_state_before_sha256']==d['model_state_after_sha256']
    for case,s in d['scores'].items():
        delta=s['metrics']['mAP']-d['scores']['global']['metrics']['mAP']
        lines.append('| '+arm+' | '+case+' | '+' | '.join(f'{s["metrics"][k]:.6f}' for k in metrics)+f' | {delta:+.6f} |')
        rows.append(dict(arm=arm,case=case,metrics=s['metrics'],delta_mAP_vs_global=delta))
lines += ['', '| 条件/控制 | 对照 | ΔmAP | ΔR1 | 首位修复 | 新增错误 | 身份宏AP变化 | 最差5% query平均AP变化 |',
    '|---|---|---:|---:|---:|---:|---:|---:|']
for arm,d in value['diagnoses'].items():
    for case,pairs in d['comparisons'].items():
        for reference,p in pairs.items():
            lines.append(f'| {arm}/{case} | {reference} | {p["delta_metrics"]["mAP"]:+.6f} | '+
                f'{p["delta_metrics"]["Rank-1"]:+.6f} | {p["rank1_repairs"]} | {p["rank1_new_errors"]} | '+
                f'{p["identity_macro_delta_ap_points"]:+.6f} | {p["worst_5_percent_mean_delta_ap_points"]:+.6f} |')
lines += ['', '上述检索指标及其变化均为百分数或百分点；身份宏平均不替代训练种子不确定性。完整逐query多正例AP、首位分组损益、身份变化及分位数保留在RESULT.json；只取首位统计无法代替mAP。下表相对L2变化为比值、余弦无单位、最大绝对变化为原c坐标单位。', '',
    '| 条件/split/控制 | c相对L2变化均值 | c与natural余弦均值 | c最大绝对变化上界 |',
    '|---|---:|---:|---:|']
for arm,d in value['diagnoses'].items():
    for split,record in d['encoding_batches'].items():
        for case,stats in record['correction_changes'].items():
            lines.append(f'| {arm}/{split}/{case} | {stats["relative_l2_change"]["mean"]:.9g} | '+
                f'{stats["cosine_to_natural"]["mean"]:.9g} | {stats["max_abs_change"]["max"]:.9g} |')
lines += ['', '执行一致性：']
for arm,d in value['diagnoses'].items():
    feature_max=max(x for split in d['cached_feature_max_abs_errors'].values() for x in split.values())
    lines.append(f'- {arm}: E8固定；原g/c/h/f缓存最大差 {feature_max:.9g}；原F距离最大差 '+
        f'{d["original_natural_distance_max_abs_error"]:.9g}；组内置换距离最大差 {d["within_group_distance_max_abs_error"]:.9g}；'+
        f'Q/G编码batch {d["encoding_batches"]["query"]["encoding_batches"]}/{d["encoding_batches"]["gallery"]["encoding_batches"]}；'+
        f'诊断内部计时 {d["elapsed_seconds"]:.3f}s（含构造/初始哈希/编码/保存/评分，截至结果对象生成；之后的产物哈希/末次输入封存/退出不计入）。')
lines += ['', '同一次实际forward的区域/g/c/h/f、原读出weight/gain子集及五套距离已保存并封存SHA。weight子集仅为原best诊断副本，没有产生新的训练权重。参数/buffer前后摘要相同、grad始终None、推理exclusion None；原472源码与输入在执行前后保持SHA。仅2026物理GPU0、1，一项两卡NN顺序执行；复用已有conda环境，无安装或重建，无功率温度读取。', '',
    '各诊断耗时是该次共享服务器进程观测，不能作为孤立模块速度、完整训练墙钟或跨主机时间比较。原CPU/NN历史失败与正式first-strict不覆盖；本项不重复旧MaxSim/拼接，不调gain/seed/loss/margin，不恢复训练。', '',
    '结论边界：natural超过破坏控制，只能说明当前最终组归属/槽位差异在这个冻结模型中有用途，不能证明原来存在错配、新增内容独立于global、整条角色必要、读出压缩为失分主因或跨数据集泛化。若相同范数的均值控制与natural持平或更好，则最终槽位差异的必要性仍未得到支持。', '',
    '后续研究继续以“每条新增证据在最终身份比较中确实有用”为问题；不根据破坏控制造贡献，不预设新匹配器/教师/三个模块能涨分。原gradient−random未晋级，三数据集超过完整Signal的Goal仍ACTIVE_UNMET；当前训练停止。', '']
(w/'RESULT_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
(w/'TABLE_ROWS.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps({'status':'REPORT_RENDERED','rows':len(rows),
    'result_sha256':hashlib.sha256((w/'RESULT.json').read_bytes()).hexdigest()}))
