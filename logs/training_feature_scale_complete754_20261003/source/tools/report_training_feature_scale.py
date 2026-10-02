"""One CPU report for six fresh feature-scale controls, including batch parity."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import queue_training_feature_scale as panel
from tools.analyze_correspondence_distances import compare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    campaign=args.campaign.resolve()
    panel.configure()
    panel.base.require_sources(campaign)
    state=json.loads((campaign/'campaign.json').read_text())
    matrix=json.loads((campaign/'accepted_matrix.json').read_text())
    assert state['status']=='COMPLETE' and state['report_invocations']==1
    assert len(state['jobs'])==12 and all(row['status']=='COMPLETE' and row['exit_code']==0 for row in state['jobs'])
    assert matrix['accepted']==matrix['expected']==len(matrix['rows'])==6
    assert not args.output_dir.exists()
    torch.set_num_threads(1)
    rows,pairs=[],[]
    for row in matrix['rows']:
        assert panel.base.verify(campaign,row['dataset'],row['variant'])==row
        output=Path(row['run_dir'])
        training=json.loads((output/'training.json').read_text())
        steps=[json.loads(line) for line in (output/'training_steps.jsonl').read_text().splitlines()]
        batches=[json.loads(line) for line in (output/'training_batch_order.jsonl').read_text().splitlines()]
        assert len(steps)==len(batches)==sum(item['steps'] for item in training['history'])
        assert [(r['epoch'],r['batch']) for r in steps]==[(r['epoch'],r['batch']) for r in batches]
        rows.append({**row,'formal_steps':len(steps),'history':training['history'],
            'nonzero_triplet_steps':sum(item['triplet']>0 for item in steps),
            'feature_norm_range':[min(item['training_feature_norm_min'] for item in steps),max(item['training_feature_norm_max'] for item in steps)],
            'training_and_epoch_evaluation_seconds':(datetime.fromisoformat(training['completed_at'])-datetime.fromisoformat(training['started_at'])).total_seconds(),
            'peak_allocated_bytes':training['peak_allocated_bytes'],
            'final_minus_best_metrics':{name:training['history'][-1]['official_fused'][name]-row['metrics'][name] for name in row['metrics']}})
    for dataset in panel.DATASETS:
        controls={r['variant']:r for r in rows if r['dataset']==dataset}
        a,b=(Path(controls[variant]['run_dir'])/'training_batch_order.jsonl' for variant in panel.RECIPES)
        assert a.read_bytes()==b.read_bytes(),dataset
        diagnosis=compare(matrix,dataset,'normalized','raw')
        diagnosis['boundary']='Matched one-seed training feature scaling; equal capacity, initial state, actual batch order and L2 deployment. Identity bootstrap is not training-seed uncertainty.'
        delta=diagnosis['delta_metrics']
        pairs.append({'dataset':dataset,'comparison':'raw minus normalized','actual_training_batch_order_equal':True,
            'phase_progress':delta['mAP']>=0.5 and delta['Rank-1']>=0,'paired_diagnosis':diagnosis})
    report={'schema':panel.SCHEMA,'status':'COMPLETE','created_at':datetime.now().astimezone().isoformat(),
        'accepted':6,'formal_epochs':300,'formal_steps':sum(r['formal_steps'] for r in rows),'rows':rows,'pairs':pairs,
        'boundary':'One fixed current-package control; affects BN/CE and Triplet together; not whole F1 gap causality, new module or SOTA. Official benchmark consumed.'}
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# F2 training feature-scale report','','| Dataset | Variant | Epoch | mAP | R1 | Steps |','|---|---|---:|---:|---:|---:|']
    lines += [f"| {r['dataset']} | {r['variant']} | {r['best_epoch']} | {r['metrics']['mAP']:.4f} | {r['metrics']['Rank-1']:.4f} | {r['formal_steps']} |" for r in rows]
    lines += ['',report['boundary']]
    (args.output_dir/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'COMPLETE','accepted':6,'formal_steps':report['formal_steps']}),flush=True)


if __name__=='__main__':
    main()
