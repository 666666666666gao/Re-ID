from pathlib import Path
import hashlib
import json
import statistics

b = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
packet = b / 'qualified_five_full_complete889'
r = json.loads((packet / 'REMOTE.json').read_bytes())
mapping = json.loads((packet / 'FILE_MAP.json').read_bytes())
output = b / 'qualified_five_training_text_analysis891'
assert not output.exists()
output.mkdir()
rows = []
inputs = {}
for row in r['rows']:
    run = row['run_dir'].split('/Trifusion/', 1)[1]
    paths = {name: Path(mapping[run + '/' + name]['local_file']) for name in ('training_steps.jsonl','training_batch_order.jsonl')}
    for name, p in paths.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == mapping[run + '/' + name]['sha256']
        inputs[run + '/' + name] = mapping[run + '/' + name]['sha256']
    steps = [json.loads(s) for s in paths['training_steps.jsonl'].read_text(encoding='utf-8').splitlines()]
    batches = [json.loads(s) for s in paths['training_batch_order.jsonl'].read_text(encoding='utf-8').splitlines()]
    assert len(steps) == len(batches) == row['formal_steps']
    assert [(s['epoch'],s['batch']) for s in steps] == [(s['epoch'],s['batch']) for s in batches]
    epochs = []
    for epoch in range(1,51):
        selected = [s for s in steps if s['epoch'] == epoch]
        epochs.append({'epoch':epoch,'steps':len(selected),'mean_incremental_loss':statistics.fmean(s['incremental_loss'] for s in selected),
            'mean_scaled_correction_global_ratio':statistics.fmean(s['actual_scaled_correction_global_ratio_mean'] for s in selected)})
    v = {'dataset':row['dataset'],'objective':row['variant'],'steps':len(steps),'query_exposures':sum(len(s['labels']) for s in batches),
        'best_epoch':row['best_epoch'],'at_best_epoch_training':epochs[row['best_epoch']-1],'at_epoch50_training':epochs[-1],
        'epochs':epochs,'positive_incremental_loss_steps':sum(s['incremental_loss'] > 0 for s in steps)}
    if row['variant'] == 'repair_keep':
        keys = ('eligible_queries','multiple_positive_queries','legal_positive_pairs','legal_triplets','repair_triplets','keep_triplets','repair_active_triplets','keep_active_triplets')
        v['support_totals'] = {k:sum(s[k] for s in steps) for k in keys}
        v['eligible_query_exposure_ratio'] = v['support_totals']['eligible_queries'] / v['query_exposures']
        v['multiple_positive_among_eligible_ratio'] = v['support_totals']['multiple_positive_queries'] / v['support_totals']['eligible_queries']
        v['repair_supported_steps'] = sum(s['repair_triplets'] > 0 for s in steps)
        v['keep_supported_steps'] = sum(s['keep_triplets'] > 0 for s in steps)
        v['repair_nonzero_loss_steps'] = sum(s['repair_loss'] > 0 for s in steps)
        v['keep_nonzero_loss_steps'] = sum(s['keep_loss'] > 0 for s in steps)
    rows.append(v)
value = {'status':'ALL9839_SAVED_TRAINING_STEP_TEXTS_ANALYZED','input_sha256':inputs,'rows':rows,
    'boundary':'CPU saved-text calculations only. Exposures are repeated query/relations, not unique identities. Mean loss and correction amplitude are not gradient or complementary retrieval evidence. M0 isolated qualification is not full-training per-role gradient proof. No new model forward, optimizer, SSH, rerun or threshold change.'}
(output / 'SUMMARY.json').write_bytes((json.dumps(value,indent=2)+'\n').encode('utf-8'))
lines = ['# 完整训练文本中的目标活动与关系支持','', '| 数据集 | 目标 | steps | 正增量loss步 | best轮修正/global均值 | E50均值 |','|---|---|---:|---:|---:|---:|']
lines += [f"| {v['dataset']} | {v['objective']} | {v['steps']} | {v['positive_incremental_loss_steps']} | {v['at_best_epoch_training']['mean_scaled_correction_global_ratio']:.6f} | {v['at_epoch50_training']['mean_scaled_correction_global_ratio']:.6f} |" for v in rows]
for v in rows:
    if v['objective']=='repair_keep':
        lines += ['',f"{v['dataset']}：合法query曝光占{v['eligible_query_exposure_ratio']:.6%}；其中多正例占{v['multiple_positive_among_eligible_ratio']:.6%}；repair/keep有支持批次{v['repair_supported_steps']}/{v['steps']}、{v['keep_supported_steps']}/{v['steps']}；非零loss批次{v['repair_nonzero_loss_steps']}/{v['steps']}、{v['keep_nonzero_loss_steps']}/{v['steps']}。"]
lines += ['',value['boundary']]
(output / 'REPORT.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
print(json.dumps({'status':value['status'],'rows':[{k:v for k,v in row.items() if k!='epochs'} for row in rows]},indent=2))
