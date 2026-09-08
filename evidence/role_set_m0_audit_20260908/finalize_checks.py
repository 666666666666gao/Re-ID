"""Finalize independent outputs, bind logs and report arithmetic, index evidence."""
import ast
import hashlib
import json
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
names=['independent_arithmetic','independent_checkpoints','supplementary_checks','binding_and_launch_checks']
values={}
for name in names:
    value=json.loads((OUT/(name+'.stdout.txt')).read_bytes())
    for path,content in value.pop('texts',{}).items():
        target=OUT/'snapshots/remote'/path.lstrip('/')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(content,encoding='utf-8',newline='')
    (OUT/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    values[name]=value
raw=OUT/'snapshots/project/evidence/msvr310_role_set_m0_complete_20260908'
summary=json.loads((raw/'m0/summary.json').read_bytes());cpu=json.loads((raw/'m0_cpu.json').read_bytes());t0=json.loads((raw/'t0.json').read_bytes())
arithmetic=values['independent_arithmetic'];checkpoints=values['independent_checkpoints'];binding=values['binding_and_launch_checks']
events=[json.loads(line) for line in (raw/'m0.log').read_text().splitlines() if line.startswith('{')]
assert len(events)==8
for event,check in zip(events,arithmetic['endpoint_checks'],strict=True):
    training=json.loads((raw/'m0'/check['endpoint']/'training.json').read_bytes())
    assert event['optimizer_steps']==check['steps']
    for key in ('epoch','optimizer_steps','learning_rate','mean_loss','elapsed_seconds'):assert event[key]==training['history'][0][key]
logged_t0=ast.literal_eval((raw/'t0.log').read_text())
assert logged_t0=={k:v for k,v in t0.items() if k!='folds'}
logged_cpu=json.loads((raw/'m0_cpu.log').read_bytes())
assert logged_cpu=={k:v for k,v in cpu.items() if k not in ('files','training_checks')}
group_summary=json.loads((raw/'direct_group_checks.json').read_bytes())
for original,computed in zip(group_summary,arithmetic['direct_recorded_norm_checks'],strict=True):
    assert original=={k:v for k,v in computed.items() if k not in ('step','encoder_tensors')}
assert cpu['checked_training_steps']==arithmetic['m0_steps']
assert cpu['checked_memory_distance_elements']==arithmetic['matrix_elements']
assert cpu['checked_history_vjp_record_forwards']==sum(x['extra_history_vjp_record_forwards'] for x in arithmetic['endpoint_checks'])
assert cpu['checked_retrieval_distance_and_rank_elements']==0
assert len(cpu['training_checks'])==8 and len(checkpoints['cpu_receipt_file_hash_checks'])==37
remote=json.loads((OUT/'remote_inventory.json').read_bytes())
snapshot_differences=[]
for path,row in remote['files'].items():
    snap=OUT/'snapshots/remote'/path.lstrip('/')
    if snap.exists() and hashlib.sha256(snap.read_bytes()).hexdigest()!=row['sha256']:
        snapshot_differences.append(path)
assert not snapshot_differences
evidence_hashes={}
for p in (OUT/'snapshots').rglob('*'):
    if p.is_file():evidence_hashes[str(p.relative_to(OUT))]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
report=dict(status='PASS_LOCAL_REPORT_LOG_AND_MANIFEST_CHECKS',all_8_training_events_exact=True,t0_log_exact=True,m0_cpu_log_exact=True,all_6_direct_group_summaries_exact=True,reported_m0_summary_sha256=hashlib.sha256((raw/'m0/summary.json').read_bytes()).hexdigest(),reported_max_direct_relative_error=max(x['relative_l2_error'] for x in arithmetic['direct_recorded_norm_checks']),maximum_independent_loss_error=max(x['max_loss_error'] for x in arithmetic['endpoint_checks']),maximum_total_loss_ledger_error=max(x['max_total_loss_error'] for x in arithmetic['endpoint_checks']),total_encoder_fresh_record_forwards=sum(x['extra_fresh_role_record_forwards'] for x in arithmetic['endpoint_checks']),total_encoder_vjp_record_forwards=sum(x['extra_history_vjp_record_forwards'] for x in arithmetic['endpoint_checks']),total_direct_check_record_forwards=6*64,m0_distance_bytes=arithmetic['matrix_elements']*4,m0_checkpoint_bytes=sum(x['bytes'] for x in checkpoints['m0_checkpoint_checks']),m0_training_loop_seconds=sum(x['training_elapsed_seconds'] for x in arithmetic['endpoint_checks']),m0_stage_elapsed_seconds=binding['closed_stage_receipts'][1]['elapsed_seconds'],remote_snapshot_byte_mismatches=snapshot_differences,snapshot_files=len(evidence_hashes))
(OUT/'local_report_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
(OUT/'snapshot_manifest.json').write_text(json.dumps(evidence_hashes,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
