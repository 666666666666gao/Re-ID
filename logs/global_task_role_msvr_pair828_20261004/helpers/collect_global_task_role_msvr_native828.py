"""Receive original completed MSVR310 native endpoint; CPU saved-distance analysis only."""
from datetime import datetime
import json
from pathlib import Path
import paramiko

packet = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/global_task_role_msvr_native_full828')
assert not packet.exists()
code = '''
from datetime import datetime
from pathlib import Path
import json,math,shutil,torch
from tools import queue_global_task_role as panel
from tools.analyze_correspondence_distances import compare
root=Path('/data/gaob/Re-ID/Trifusion')
campaign=root/'logs/global_task_role_v1_20261004_824'
panel.configure()
panel.base.require_sources(campaign)
controls=panel.require_controls()
state=json.loads((campaign/'campaign.json').read_text())
job=next(j for j in state['jobs'] if (j['dataset'],j['variant'],j['phase'])==('MSVR310','native','full'))
assert job['status']=='COMPLETE' and job['exit_code']==0
row=panel.base.verify(campaign,'MSVR310','native')
semantic=next(j['result'] for j in state['jobs'] if (j['dataset'],j['variant'],j['phase'])==('MSVR310','semantic','full'))
run=Path(row['run_dir'])
training=json.loads((run/'training.json').read_text())
steps=[json.loads(line) for line in (run/'training_steps.jsonl').read_text().splitlines()]
assert [r['epoch'] for r in training['history']]==list(range(1,51))
assert len(steps)==sum(r['steps'] for r in training['history'])
assert all(math.isfinite(r['global_loss']) and math.isfinite(r['fused_role_loss']) for r in steps)
assert all(len(r['head_losses'])==len(r['global_head_losses'])==3 for r in steps)
assert all(abs(r['loss']-r['global_loss']-r['fused_role_loss'])<=1e-5 for r in steps)
matched=next(r for r in controls['rows'] if (r['dataset'],r['variant'])==('MSVR310','native'))
assert (run/'training_batch_order.jsonl').read_bytes()==(Path(matched['run_dir'])/'training_batch_order.jsonl').read_bytes()
combined=dict(rows=[dict(r,variant='previous_'+r['variant']) for r in controls['rows']]
 +[dict(r,variant='original_'+r['variant']) for r in controls['original_rows']]+[semantic,row])
torch.set_num_threads(1)
pairs={name:compare(combined,'MSVR310',name,'native') for name in ('previous_native','original_native','previous_global_only','semantic')}
trajectory=[]
norms=('shared_global_norm_mean','correction_norm_mean','actual_scaled_correction_norm_mean',
       'actual_scaled_correction_global_ratio_mean')
for epoch in range(1,51):
 epoch_steps=[r for r in steps if r['epoch']==epoch]
 trajectory.append(dict(epoch=epoch,steps=len(epoch_steps),
  mean_global_loss=sum(r['global_loss'] for r in epoch_steps)/len(epoch_steps),
  mean_fused_role_loss=sum(r['fused_role_loss'] for r in epoch_steps)/len(epoch_steps),
  **{n:sum(r[n] for r in epoch_steps)/len(epoch_steps) for n in norms}))
primary=pairs['previous_global_only']['delta_metrics']
textpaths=[run/'training.json',run/'official_metrics.json',campaign/'initialization/MSVR310_native.json',
 campaign/'prepare_MSVR310_native.log',campaign/'MSVR310_native_m0.log',
 campaign/'MSVR310_native_train.log',campaign/'MSVR310_native_evaluate.log',
 root/'trained-model/global_task_role_v1_20261004_824_m0_native_MSVR310/training.json']
files={str(p.relative_to(root)):p.read_text() for p in textpaths}
summary=dict(status='MSVR310_NATIVE_FORMAL50_AND_FIRST_STRICT_EVALUATION_VERIFIED',at=datetime.now().astimezone().isoformat(),
 row=row,semantic_row=semantic,formal_epochs=50,formal_steps=len(steps),source_files_verified=330,
 actual_training_batch_order_equal=True,global_and_fused_losses_verified=True,pairs=pairs,
 primary_progress=primary['mAP']>=0.5 and primary['Rank-1']>=0,training_task_and_norm_trajectory=trajectory,
 best_to_last_map_drop=row['metrics']['mAP']-training['history'][-1]['official_fused']['mAP'],
 training_and_epoch_evaluation_seconds=(datetime.fromisoformat(training['completed_at'])-datetime.fromisoformat(training['started_at'])).total_seconds(),
 campaign_state=state,disk_free_bytes=shutil.disk_usage(root).free,
 boundary='RGBNT201 and MSVR310 pairs closed; six-endpoint campaign and once-only report pending. No new model forward,optimizer,selection,retired M0 verification or report replay. Same-variant task controls match capacity/initial state; native-semantic differs159296parameters and possibly random-number consumption. Generic inherited compare capacity wording does not override actual same-variant witness. Fixed-model identity bootstrap is not training-seed uncertainty; consumed official benchmarks,not causal/novelty/SOTA proof.')
print(json.dumps(dict(summary=summary,files=files)))
'''
compile(code, 'remote_native_formal_collector.py', 'exec')
packet.mkdir()
(packet / 'remote_native_formal_collector.py').write_text(code, encoding='utf-8')
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
stdin, out, err = client.exec_command("cd /data/gaob/Re-ID/Trifusion && CUDA_VISIBLE_DEVICES='' /data/gaob/Re-ID/conda-envs/tri_reid/bin/python -B -")
stdin.write(code)
stdin.channel.shutdown_write()
out.channel.settimeout(180)
data, error = out.read(), err.read()
status = out.channel.recv_exit_status()
(packet / 'stdout.json').write_bytes(data)
(packet / 'stderr.txt').write_bytes(error)
(packet / 'EXIT.json').write_text(json.dumps(dict(exit_code=status, at=datetime.now().astimezone().isoformat())) + '\n', encoding='utf-8')
client.close()
assert status == 0, error.decode()
received = json.loads(data)
for name, value in received['files'].items():
    p = packet / 'received' / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(value, encoding='utf-8')
(packet / 'SUMMARY.json').write_text(json.dumps(received['summary'], indent=2) + '\n', encoding='utf-8')
summary = received['summary']
print(json.dumps({key: summary[key] for key in ('status','at','formal_steps','row','primary_progress','best_to_last_map_drop','training_and_epoch_evaluation_seconds','disk_free_bytes')}))
