from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,math

root=Path('/data/gaob/Re-ID/Trifusion');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
campaign=root/'logs/qualified_five_incremental_full_v1_20261007_888'
journal=root/'logs/qualified_five_incremental_full_launch_20261007_888'
report=root/'results/qualified_five_incremental_full_complete_20261007_888'
exit_record=json.loads((journal/'EXIT.json').read_text());assert exit_record['exit_code']==0
child=json.loads((journal/'CHILD.json').read_text());assert not (Path('/proc')/str(child['pid'])).exists()
state=json.loads((campaign/'campaign.json').read_text())
assert state['status']=='COMPLETE' and state['report_invocations']==1 and state['report_exit_code']==0
jobs={('RGBNT201','md_batch_ratio'),('RGBNT201','repair_keep'),('MSVR310','md_batch_ratio'),('MSVR310','repair_keep'),('RGBNT100','md_batch_ratio')}
assert len(state['jobs'])==5 and {(r['dataset'],r['objective']) for r in state['jobs']}==jobs
assert all(j['status']=='COMPLETE' and j['exit_code']==0 for j in state['jobs'])
source=json.loads((root/'refine-logs/incremental_role_objective_v1/QUALIFIED_FIVE_SOURCE_SCOPE.json').read_text())['source_sha256']
assert len(source)==420 and all(sha(root/n)==d for n,d in source.items())
manifest=json.loads((campaign/'manifest.json').read_text());assert manifest['source_sha256']==source
assert all(sha(Path(n))==d for n,d in manifest['initialization_sha256'].items())
controls=json.loads((root/'refine-logs/incremental_role_objective_v1/INPUT_SEAL.json').read_text())
assert len(controls['rows'])==6 and len(controls['artifact_sha256'])==45
assert all(sha(Path(n))==d for n,d in controls['artifact_sha256'].items())
initial_seal=json.loads((root/'refine-logs/incremental_role_objective_v1/INITIAL_BOUNDARY_INPUT_SEAL.json').read_text())
assert len(initial_seal['artifact_sha256'])==48 and all(sha(Path(n))==d for n,d in initial_seal['artifact_sha256'].items())
matrix=json.loads((campaign/'accepted_matrix.json').read_text());summary=json.loads((report/'SUMMARY.json').read_text())
assert matrix['accepted']==matrix['expected']==len(matrix['rows'])==summary['accepted']==5
assert summary['status']=='COMPLETE' and summary['formal_epochs']==250 and summary['formal_steps']==9839
assert len(summary['pairs'])==12
assert {(r['dataset'],r['variant']) for r in matrix['rows']}==jobs
assert len(summary['missing'])==1 and summary['missing'][0]['dataset']=='RGBNT100' and summary['missing'][0]['objective']=='repair_keep'
assert 'Missing formal endpoint: RGBNT100 repair_keep' in (report/'REPORT.md').read_text()
expected_pairs={(d,o,c) for d,o in jobs for c in ('raw_semantic','raw_global_only')}
expected_pairs|={(d,'repair_keep','md_batch_ratio') for d in ('RGBNT201','MSVR310')}
assert {(p['dataset'],p['objective'],p['control']) for p in summary['pairs']}==expected_pairs
counts={'RGBNT201':836,'MSVR310':591,'RGBNT100':1715}
assert all(len(p['paired_diagnosis']['query_changes'])==counts[p['dataset']] for p in summary['pairs'])
files={p for folder in (campaign,journal,report) for p in folder.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.log','.txt','.md','.csv','.py')}
rows=[]
for row in matrix['rows']:
    dataset,objective=row['dataset'],row['variant'];output=Path(row['run_dir'])
    assert output==root/f'trained-model/{campaign.name}_full_{objective}_{dataset}'
    training=json.loads((output/'training.json').read_text());official=json.loads((output/'official_metrics.json').read_text())
    assert training['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and official['status']=='COMPLETE'
    assert training['schema']==official['schema']==matrix['schema']=='trifusion-incremental-role-objective-v1'
    assert training['initializer']==row['initializer'] and official['condition']==training['condition']
    assert official['condition']['incremental_objective']==objective
    assert [r['epoch'] for r in training['history']]==list(range(1,51))
    best=max(training['history'],key=lambda r:(r['official_fused']['mAP'],r['epoch']))
    assert official['selected_epoch']==training['best_epoch']==row['best_epoch']==best['epoch']
    assert official['metrics']==row['metrics']
    assert all(math.isfinite(v) and abs(v-best['official_fused'][k])<1e-5 for k,v in official['metrics'].items())
    assert official['training_epochs']==50 and official['seed']==42
    assert official['independent_upstream_metrics_equal'] and not official['reranking']
    assert official['protocol_sha256']==row['initializer']['protocol_sha256']
    assert sha(output/'best_map.pth')==official['checkpoint_sha256']==row['checkpoint_sha256']
    assert sha(output/'official_distances.pt')==official['distance_sha256']==row['distance_sha256']
    assert sha(output/'best_epoch_distances.pt')==official['training_best_distance_sha256']
    steps=[json.loads(line) for line in (output/'training_steps.jsonl').read_text().splitlines()]
    batches=[json.loads(line) for line in (output/'training_batch_order.jsonl').read_text().splitlines()]
    assert len(steps)==len(batches)==sum(h['steps'] for h in training['history'])
    assert [(s['epoch'],s['batch']) for s in steps]==[(s['epoch'],s['batch']) for s in batches]
    raw=next(r for r in controls['rows'] if r['dataset']==dataset and r['variant']=='semantic')
    assert (output/'training_batch_order.jsonl').read_bytes()==(Path(raw['run_dir'])/'training_batch_order.jsonl').read_bytes()
    assert sha(output/'official_metrics.json')==row['receipt_sha256']
    files.update(p for p in output.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl','.log','.txt'))
    rows.append(dict(row,formal_steps=len(steps),last_epoch_metrics=training['history'][-1]['official_fused'],
        best_to_last_map_drop=best['official_fused']['mAP']-training['history'][-1]['official_fused']['mAP'],
        exact_full_batch_order_equal=True))
assert sum(r['formal_steps'] for r in rows)==9839
assert json.loads((root/'logs/incremental_role_objective_m0_launch_20261007_886/EXIT.json').read_text())['exit_code']==1
original_m0=json.loads((root/'logs/incremental_role_objective_m0_v1_20261007_886/campaign.json').read_text())
assert sum('acceptance' in j for j in original_m0['jobs'])==5
print(json.dumps(dict(status='FIVE_FULL50_FIRST_STRICT_AND_TWELVE_PAIRS_PHYSICAL_SHA_VERIFIED',at=datetime.now().astimezone().isoformat(),
    source_count=420,current45_and48_inputs_unchanged=True,formal_completed=5,formal_epochs=250,formal_steps=9839,
    rows=rows,pairs=[dict(dataset=p['dataset'],objective=p['objective'],control=p['control'],phase_progress=p['phase_progress'],
        delta=p['paired_diagnosis']['delta_metrics']) for p in summary['pairs']],missing=summary['missing'],
    actual_parent_exit=exit_record,free_bytes=shutil.disk_usage(root).free,
    files={str(p.relative_to(root)):dict(text=p.read_text(),sha256=sha(p)) for p in sorted(files)},
    boundary='Five explicit qualified endpoints, complete250epochs/9839updates/12all-query pairs. Failed100repair remains missing and oldM0failures retained. Consumed official single-seed development; no formula novelty, fullMDReID reproduction, fullthree-dataset repair result, SOTA or broadgoal completion claim.')))
