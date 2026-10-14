"""Compare the completed absolute-value condition with the existing controls."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import queue_absolute_pretrained_value as panel
from tools.analyze_correspondence_distances import camera_scores,compare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();assert not args.output_dir.exists()
    torch.set_num_threads(1)
    state=json.loads((args.campaign/'campaign.json').read_text())
    assert state['status']=='COMPLETE' and state['accepted']==state['expected']==1
    assert json.loads((ROOT/'logs/absolute_pretrained_value_launch958/EXIT.json').read_text())['exit_code']==0
    panel.base.SCHEMA=panel.SCHEMA
    panel.base.source_map=panel.source_map
    panel.base.output_dir=panel.previous.previous.output_dir
    arm=state['arms'][0];child=Path(arm['campaign'])
    accepted=panel.base.verify(child,'RGBNT201','semantic')
    assert dict(accepted,value_reference='uniform_reference',value_reading='absolute')==arm['result']
    assert all(step['status']=='COMPLETE' and step['exit_code']==0 for step in arm['steps'])
    old_path=ROOT/'logs/pretrained_value_training951/pretrained_value_training951_uniform_reference/ACCEPTED_RESULT.json'
    control=json.loads(old_path.read_text())
    manifest=json.loads((child/'manifest.json').read_text())
    assert panel.base.sha(old_path)==manifest['control_receipt_sha256']
    old_run=Path(control['run_dir']);new_run=Path(accepted['run_dir'])
    assert (old_run/'training_batch_order.jsonl').read_bytes()==(new_run/'training_batch_order.jsonl').read_bytes()
    original=json.loads((ROOT/'results/pretrained_value_report_20261014_957/SUMMARY.json').read_text())
    rows=[dict(row) for row in original['rows'] if row['variant'] in ('raw_semantic','raw_global_only','uniform_reference')]
    assert len(rows)==3
    rows.append(dict(accepted,variant='absolute_reading'))
    for row in rows:
        run=Path(row['run_dir'])
        assert panel.base.sha(run/'best_map.pth')==row['checkpoint_sha256']
        assert panel.base.sha(run/'official_distances.pt')==row['distance_sha256']
        assert panel.base.sha(run/'official_metrics.json')==row['receipt_sha256']
        data=torch.load(run/'official_distances.pt',map_location='cpu',weights_only=False)
        score=camera_scores(data['fused'].numpy(),data['query_ids'],data['gallery_ids'],data['query_cameras'],data['gallery_cameras'])
        assert all(abs(score['metrics'][key]-row['metrics'][key])<1e-5 for key in row['metrics'])
    pairs=[]
    for name in ('uniform_reference','raw_global_only','raw_semantic'):
        diagnosis=compare(dict(rows=rows),'RGBNT201',name,'absolute_reading')
        pairs.append(dict(control=name,candidate='absolute_reading',diagnosis=diagnosis,
                          phase_progress=diagnosis['delta_metrics']['mAP']>=.5 and diagnosis['delta_metrics']['Rank-1']>=0))
    receipt=json.loads((new_run/'training.json').read_text())
    steps=[json.loads(line) for line in (new_run/'training_steps.jsonl').read_text().splitlines()]
    assert len(steps)==sum(row['steps'] for row in receipt['history'])
    columns=('value_attention_change_abs_mean','value_region_norm_mean','correction_norm_mean',
             'actual_scaled_correction_global_ratio_mean','shared_global_norm_mean','global_loss','fused_role_loss')
    curves=[]
    for epoch in range(1,51):
        values=[row for row in steps if row['epoch']==epoch];assert values
        curves.append(dict(epoch=epoch,means={key:sum(float(row[key]) for row in values)/len(values) for key in columns}))
    summary=dict(status='COMPLETE_ABSOLUTE_VALUE_SAVED_DISTANCE_DIAGNOSIS',schema=panel.SCHEMA,
        rows=rows,pairs=pairs,batch_order_equal=True,formal_epochs=50,formal_steps=len(steps),
        history=receipt['history'],actual_reading_and_task_curves=curves,
        at=datetime.now().astimezone().isoformat(),
        boundary='One fixed seed42 absolute-value condition versus an already completed same-state/parameter/recipe differential control. Original bests/receipts/distances rehashed; the retired old M0 probe is not replayed or reverified. Full legal gallery/camera rules. This is benchmark-consumed development evidence, not unbiased SOTA or an original-mechanism proof.')
    args.output_dir.mkdir()
    panel.base.queue.write(args.output_dir/'SUMMARY.json',summary)
    lines=['# Absolute pretrained value: full terminal report','',
           '| Condition | Epoch | mAP | R1 | R5 | R10 |','|---|---:|---:|---:|---:|---:|']
    for row in rows:lines.append(f"| {row['variant']} | {row['best_epoch']} | "+' | '.join(f"{row['metrics'][key]:.4f}" for key in ('mAP','Rank-1','Rank-5','Rank-10'))+' |')
    lines+=['','| Control | Delta mAP | Delta R1 | Repairs | New errors | Advance |','|---|---:|---:|---:|---:|---|']
    for pair in pairs:
        d=pair['diagnosis'];lines.append(f"| {pair['control']} | {d['delta_metrics']['mAP']:+.4f} | {d['delta_metrics']['Rank-1']:+.4f} | {d['rank1_repairs']} | {d['rank1_new_errors']} | {pair['phase_progress']} |")
    lines+=['',summary['boundary']]
    (args.output_dir/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=summary['status'],metrics=accepted['metrics'],pairs=len(pairs))))


if __name__=='__main__':main()
