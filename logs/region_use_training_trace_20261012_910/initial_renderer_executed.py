"""Render the already computed training trace; no model or result recomputation."""
import hashlib
import json
from pathlib import Path


work = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903')
source = work / 'training_trace910/SUMMARY.json'
value = json.loads(source.read_text(encoding='utf-8'))
lines = [
    '# 已保存训练日志：困难关系与修正幅度',
    '',
    '范围为原906已完成的RGBNT201、MSVR310四端，每端seed42、50轮。这里只解析已经保存且与intake SHA一致的标量日志；没有新前向、梯度、更新、权重或服务器/GPU进度查询。RGBNT100尚不在本报告中。',
    '',
    '表中“非正global margin”是合法训练三元关系中，detached global余弦正例分数减负例分数不大于0的曝光占比；不是逐query检索错误率。重复曝光不是独立试验。aux及修正/global比均为每轮已记录batch均值的等权平均，不是同best的query/gallery几何。',
    '',
    '| 数据集/条件 | 记录位置 | epoch | 非正global margin曝光占比(%) | aux均值 | 修正/global范数比均值 | 当轮已记录fused mAP(%) |',
    '|---|---|---:|---:|---:|---:|---:|',
]
names = {'first': '首轮', 'selected': 'mAP-best', 'last': '末轮'}
for run in value['results']:
    for name, row in run['first_selected_last'].items():
        means = row['recorded_batch_mean_averages']
        lines.append(
            f"| {run['dataset']}/{run['condition']} | {names[name]} | {row['epoch']} | "
            f"{100*row['fraction_global_nonpositive_margin_among_legal_exposures']:.8f} | "
            f"{means['region_use_loss']:.8f} | "
            f"{means['actual_scaled_correction_global_ratio_mean']:.8f} | "
            f"{row['official_fused']['mAP']:.8f} |"
        )
lines += [
    '',
    '实际训练辅助项使用softplus(-sqrt(128)×(global_margin+region_margin)/2)，再按原条件的合法关系权重聚合。global任务仍单独更新；因此global_margin改变，也会改变aux数值。该计算关系说明aux下降不能单独证明区域关系变好，但这份日志没有分离两者各自贡献，不能据此判定下降的唯一原因。',
    '',
    'RGBNT201在best第8轮的合法非正global margin曝光占比已很小，末轮为0；修正范数比继续增长，而记录的fused mAP低于best。MSVR310仍存在非正margin训练关系，修正范数比接近0.005。这些是各自来源训练轨迹的描述，不是跨数据集的因果比较，也不据此放大gain、改采样、加教师、挑seed或追加模块。',
    '',
    '需要在原六端终态后，依据同best实际global、修正和最终区域的检索分解判断证据是否有新增价值。当前报告没有这些向量/分数，不能宣称读出用坏、区域更强、压缩损失、背景捷径或跨身份泛化已被证明。',
    '',
    '全部当轮fused数值来自已保存的正式训练history，不是本消费者重新评价。官方query/gallery已用于逐轮选best与研发选择，不称未使用测试或完整多种子确认。',
    '',
    '机器可读的200个epoch记录及12个输入SHA见SUMMARY.json；分析源为analyse_training_trace910.py。新鲜审计单独记录，当前文本先保存在私有工作目录，不冒称已获审计接受。',
    '',
    f"SUMMARY SHA256: `{hashlib.sha256(source.read_bytes()).hexdigest()}`",
    '',
]
output = work / 'training_trace910/REPORT.md'
assert not output.exists()
output.write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps({'status': 'PRIVATE_SAVED_TRACE_RENDERED', 'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}, ensure_ascii=True))
