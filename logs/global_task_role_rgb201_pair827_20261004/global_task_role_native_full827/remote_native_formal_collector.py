
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
job=next(j for j in state['jobs'] if (j['dataset'],j['variant'],j['phase'])==('RGBNT201','native','full'))
assert job['status']=='COMPLETE' and job['exit_code']==0
row=panel.base.verify(campaign,'RGBNT201','native')
semantic=next(j['result'] for j in state['jobs'] if (j['dataset'],j['variant'],j['phase'])==('RGBNT201','semantic','full'))
run=Path(row['run_dir'])
training=json.loads((run/'training.json').read_text())
steps=[json.loads(line) for line in (run/'training_steps.jsonl').read_text().splitlines()]
assert [r['epoch'] for r in training['history']]==list(range(1,51))
assert len(steps)==sum(r['steps'] for r in training['history'])
assert all(math.isfinite(r['global_loss']) and math.isfinite(r['fused_role_loss']) for r in steps)
assert all(len(r['head_losses'])==len(r['global_head_losses'])==1 for r in steps)
assert all(abs(r['loss']-r['global_loss']-r['fused_role_loss'])<=1e-5 for r in steps)
matched=next(r for r in controls['rows'] if (r['dataset'],r['variant'])==('RGBNT201','native'))
assert (run/'training_batch_order.jsonl').read_bytes()==(Path(matched['run_dir'])/'training_batch_order.jsonl').read_bytes()
combined=dict(rows=[dict(r,variant='previous_'+r['variant']) for r in controls['rows']]
 +[dict(r,variant='original_'+r['variant']) for r in controls['original_rows']]+[semantic,row])
torch.set_num_threads(1)
pairs={name:compare(combined,'RGBNT201',name,'native') for name in ('previous_native','original_native','previous_global_only','semantic')}
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
textpaths=[run/'training.json',run/'official_metrics.json',campaign/'initialization/RGBNT201_native.json',
 campaign/'prepare_RGBNT201_native.log',campaign/'RGBNT201_native_m0.log',
 campaign/'RGBNT201_native_train.log',campaign/'RGBNT201_native_evaluate.log',
 root/'trained-model/global_task_role_v1_20261004_824_m0_native_RGBNT201/training.json']
files={str(p.relative_to(root)):p.read_text() for p in textpaths}
summary=dict(status='RGBNT201_NATIVE_FORMAL50_AND_FIRST_STRICT_EVALUATION_VERIFIED',at=datetime.now().astimezone().isoformat(),
 row=row,semantic_row=semantic,formal_epochs=50,formal_steps=len(steps),source_files_verified=330,
 actual_training_batch_order_equal=True,global_and_fused_losses_verified=True,pairs=pairs,
 primary_progress=primary['mAP']>=0.5 and primary['Rank-1']>=0,training_task_and_norm_trajectory=trajectory,
 best_to_last_map_drop=row['metrics']['mAP']-training['history'][-1]['official_fused']['mAP'],
 training_and_epoch_evaluation_seconds=(datetime.fromisoformat(training['completed_at'])-datetime.fromisoformat(training['started_at'])).total_seconds(),
 campaign_state=state,disk_free_bytes=shutil.disk_usage(root).free,
 boundary='Only RGBNT201 pair closed; six-endpoint campaign and once-only report pending. No new model forward,optimizer,selection,retired M0 verification or report replay. Same-variant task controls match capacity/initial state; native-semantic differs159296parameters and possibly random-number consumption. Generic inherited compare capacity wording does not override actual same-variant witness. Fixed-model identity bootstrap is not training-seed uncertainty; consumed official benchmarks,not causal/novelty/SOTA proof.')
print(json.dumps(dict(summary=summary,files=files)))
