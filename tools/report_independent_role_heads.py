"""All-query report against three existing raw controls per dataset."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import queue_independent_role_heads as panel
from tools.analyze_correspondence_distances import compare,camera_scores,scene_scores


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    panel.configure()
    panel.base.require_sources(args.campaign)
    controls=panel.previous.require_controls()
    state=json.loads((args.campaign/'campaign.json').read_text())
    matrix=json.loads((args.campaign/'accepted_matrix.json').read_text())
    assert state['status']=='COMPLETE' and state['report_invocations']==1
    assert len(state['jobs'])==6 and all(r['status']=='COMPLETE' and r['exit_code']==0 for r in state['jobs'])
    assert matrix['accepted']==matrix['expected']==len(matrix['rows'])==3
    assert not args.output_dir.exists()
    torch.set_num_threads(1)
    combined={'rows':[dict(r,variant='raw_'+r['variant']) for r in controls['rows']]+
                       [dict(r,variant='independent_role_heads') for r in matrix['rows']]}
    rows,pairs=[],[]
    for row in matrix['rows']:
        dataset=row['dataset']
        assert panel.previous.accepted_row(args.campaign,dataset,'semantic')==row
        run=Path(row['run_dir'])
        training=json.loads((run/'training.json').read_text())
        steps=[json.loads(line) for line in (run/'training_steps.jsonl').read_text().splitlines()]
        batches=[json.loads(line) for line in (run/'training_batch_order.jsonl').read_text().splitlines()]
        assert len(steps)==len(batches)==sum(r['steps'] for r in training['history'])
        assert [(r['epoch'],r['batch']) for r in steps]==[(r['epoch'],r['batch']) for r in batches]
        rows.append(dict(row,source='independent_role_heads',formal_steps=len(steps),history=training['history'],
            complete_training_and_epoch_eval_seconds=(datetime.fromisoformat(training['completed_at'])-
                datetime.fromisoformat(training['started_at'])).total_seconds()))
        for control in ('raw_semantic','raw_global_only'):
            old=next(r for r in controls['rows'] if (r['dataset'],'raw_'+r['variant'])==(dataset,control))
            assert (run/'training_batch_order.jsonl').read_bytes()==(Path(old['run_dir'])/'training_batch_order.jsonl').read_bytes()
            result=compare(combined,dataset,control,'independent_role_heads')
            scorer=scene_scores if dataset=='MSVR310' else camera_scores
            environment='scenes' if dataset=='MSVR310' else 'cameras'
            scores={}
            for label in (control,'independent_role_heads'):
                selected=next(r for r in combined['rows'] if (r['dataset'],r['variant'])==(dataset,label))
                data=torch.load(Path(selected['run_dir'])/'official_distances.pt',map_location='cpu',weights_only=False)
                scores[label]=scorer(data['fused'].numpy(),data['query_ids'],data['gallery_ids'],
                    data['query_'+environment],data['gallery_'+environment])
            result['query_changes']=[dict(query_index=i,identity=int(identity),
                control_ap=float(scores[control]['average_precision'][i]),
                candidate_ap=float(scores['independent_role_heads']['average_precision'][i]),
                control_first_rank=int(scores[control]['first_match_rank'][i]),
                candidate_first_rank=int(scores['independent_role_heads']['first_match_rank'][i]))
                for i,identity in enumerate(data['query_ids'])]
            delta=result['delta_metrics']
            pairs.append(dict(dataset=dataset,source='independent_role_heads',control=control,
                actual_training_batch_order_equal=True,
                phase_progress=delta['mAP']>=0.5 and delta['Rank-1']>=0,paired_diagnosis=result))
    value=dict(schema=panel.SCHEMA,status='COMPLETE',created_at=datetime.now().astimezone().isoformat(),
        accepted=3,formal_epochs=150,formal_steps=sum(r['formal_steps'] for r in rows),rows=rows,pairs=pairs,
        role_head_parameters={r['dataset']:r['initializer']['role_head_parameters'] for r in rows},
        boundary='Seed42 exploratory training-head ownership comparison on consumed official benchmarks. Old raw semantic/global controls reused with full batch-order equality. Evidence, raw objectives, deployment and recipe unchanged; extra trainable head capacity and independent BN buffers disclosed. Not an original module, proof of exclusive cause, training-seed stability or SOTA. No normalized metric-stage rescue.')
    args.output_dir.mkdir(parents=True)
    panel.base.queue.write(args.output_dir/'SUMMARY.json',value)
    lines=['# 独立角色训练头对照','', '| 数据集 | best轮 | mAP | R1 | 正式步数 |',
           '|---|---:|---:|---:|---:|']
    lines += [f'| {r["dataset"]} | {r["best_epoch"]} | {r["metrics"]["mAP"]:.4f} | {r["metrics"]["Rank-1"]:.4f} | {r["formal_steps"]} |' for r in rows]
    lines += ['',value['boundary']]
    (args.output_dir/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='COMPLETE',accepted=3,pairs=len(pairs))),flush=True)


if __name__=='__main__':
    main()
