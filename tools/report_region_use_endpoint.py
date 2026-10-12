"""CPU-only diagnosis of one completed, strictly evaluated original endpoint."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_region_use_probe as panel
from tools.report_prepool_dense_correspondence import paired, descriptor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--dataset', choices=panel.previous.DATASETS, required=True)
    parser.add_argument('--comparison', choices=panel.CONDITIONS, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output_dir.exists()
    seal = json.loads(panel.SCOPE.read_text())
    panel.source_map()
    state = json.loads((args.campaign/'campaign.json').read_text())
    assert state['schema'] == panel.SCHEMA and state['input_seal_sha256'] == panel.base.sha(panel.SCOPE)
    job = next(row for row in state['jobs'] if (row['dataset'],row['comparison']) == (args.dataset,args.comparison))
    assert job['status'] == 'COMPLETE'
    assert [row['mode'] for row in job['stages']] == ['prepare','m0','train','evaluate']
    assert all(row['status'] == 'COMPLETE' and row['exit_code'] == 0 for row in job['stages'])
    run = panel.output_dir(args.campaign,'full',args.dataset,args.comparison)
    receipt = json.loads((run/'official_metrics.json').read_text())
    assert receipt == job['result'] and receipt['schema'] == panel.SCHEMA and receipt['status'] == 'COMPLETE'
    assert receipt['seed'] == 42 and receipt['training_epochs'] == 50
    assert receipt['condition']['region_use_comparison'] == args.comparison
    assert panel.base.sha(run/'best_map.pth') == receipt['checkpoint_sha256']
    assert panel.base.sha(run/'official_distances.pt') == receipt['distance_sha256']
    training = json.loads((run/'training.json').read_text())
    assert training['status'] == 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE'
    assert [row['epoch'] for row in training['history']] == list(range(1,51))
    selected = max(training['history'],key=lambda row:(row['official_fused']['mAP'],row['epoch']))
    assert receipt['selected_epoch'] == training['best_epoch'] == selected['epoch']
    assert all(abs(receipt['metrics'][name]-selected['official_fused'][name])<1e-5 for name in receipt['metrics'])
    steps = [json.loads(line) for line in (run/'training_steps.jsonl').read_text().splitlines()]
    batches = [json.loads(line) for line in (run/'training_batch_order.jsonl').read_text().splitlines()]
    assert len(steps) == len(batches) == sum(row['steps'] for row in training['history'])
    assert [(r['epoch'],r['batch']) for r in steps] == [(r['epoch'],r['batch']) for r in batches]
    control_path = ROOT/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'
    assert panel.base.sha(control_path) == seal['artifact_sha256'][str(control_path)]
    controls = json.loads(control_path.read_text())
    raw = next(row for row in controls['rows'] if (row['dataset'],row['variant']) == (args.dataset,'semantic'))
    global_only = next(row for row in controls['rows'] if (row['dataset'],row['variant']) == (args.dataset,'global_only'))
    old_order_path = Path(raw['run_dir'])/'training_batch_order.jsonl'
    assert panel.base.sha(old_order_path) == seal['artifact_sha256'][str(old_order_path)]
    old_order = old_order_path.read_bytes()
    assert (run/'training_batch_order.jsonl').read_bytes() == old_order
    traces=[]
    for epoch in range(1,51):
        rows=[row for row in steps if row['epoch']==epoch]
        assert len(rows)==training['history'][epoch-1]['steps']
        traces.append(dict(epoch=epoch,auxiliary_loss_mean=float(np.mean([row['region_use_loss'] for row in rows])),
            eligible_query_exposures=sum(row['region_use_support']['eligible_queries'] for row in rows),
            legal_triplets=sum(row['region_use_support']['legal_triplets'] for row in rows),
            source_query_exposures=len(rows)*training['initializer']['batch_size']))
    candidate=dict(dataset=args.dataset,metrics=receipt['metrics'],best_epoch=receipt['selected_epoch'],
        run_dir=str(run),distance_sha256=receipt['distance_sha256'],checkpoint_sha256=receipt['checkpoint_sha256'])
    torch.set_num_threads(1)
    pairs=[]
    for control,name in ((raw,'raw_semantic'),(global_only,'independent_global')):
        result=paired(args.dataset,descriptor(control,name),descriptor(candidate,args.comparison))
        changes=result['query_changes'];size=len(changes)
        groups=dict(repaired=[],new_error=[],first_still_correct=[],first_still_wrong=[])
        for row in changes:
            first,second=row['control_first_rank']==1,row['candidate_first_rank']==1
            group=('first_still_correct' if second else 'new_error') if first else ('repaired' if second else 'first_still_wrong')
            groups[group].append((row['candidate_ap']-row['control_ap'])*100)
        result['rank1_group_ap_contributions']={name:dict(queries=len(values),
            contribution_to_total_delta_map_points=sum(values)/size) for name,values in groups.items()}
        assert abs(sum(v['contribution_to_total_delta_map_points'] for v in result['rank1_group_ap_contributions'].values())-result['delta_metrics']['mAP'])<1e-5
        delta=np.sort([(r['candidate_ap']-r['control_ap'])*100 for r in changes])
        result['worst_five_percent_query_ap_delta_mean_points']=float(delta[:int(np.ceil(.05*size))].mean())
        pairs.append(result)
    value=dict(schema=panel.SCHEMA,status='ONE_ORIGINAL_ENDPOINT_CPU_DIAGNOSIS_COMPLETE',
        created_at=datetime.now().astimezone().isoformat(),dataset=args.dataset,comparison=args.comparison,
        candidate=candidate,metrics=receipt['metrics'],history=training['history'],auxiliary_traces=traces,
        formal_recorded_steps=len(steps),actual_raw_semantic_batch_order_equal=True,pairs=pairs,
        best_to_last_map_drop=receipt['metrics']['mAP']-training['history'][-1]['official_fused']['mAP'],
        producer_wall_seconds=(datetime.fromisoformat(job['stages'][2]['completed_at'])-datetime.fromisoformat(job['stages'][2]['started_at'])).total_seconds(),
        training_and_epoch_evaluation_interval_seconds=(datetime.fromisoformat(training['completed_at'])-datetime.fromisoformat(training['started_at'])).total_seconds(),
        training_loop_seconds=sum(row['seconds'] for row in training['history']),
        partition_peak_memory=training['partition_peak_memory'],
        inputs={str(p):panel.base.sha(p) for p in (Path(__file__),panel.SCOPE,run/'official_metrics.json',run/'training.json',run/'training_steps.jsonl',run/'training_batch_order.jsonl')},
        boundary='One seed42, original full50/strict selected-best distances, complete legal query/gallery and matched raw-semantic order. No model inference, optimizer, extra seed, test-time comparison, stronger-teacher claim or post-hoc rescue. Source support/auxiliary loss and AP tails are descriptive; no unique failure cause. Same-model global/regional decomposition and uniform-versus-ambiguity remain separate consumers, not fabricated here. Official benchmark consumed in development.')
    args.output_dir.mkdir(parents=True)
    panel.base.queue.write(args.output_dir/'SUMMARY.json',value)
    lines=['# Actual-region supervision: one original endpoint','',
        '| Dataset | Condition | Best epoch | mAP | R1 | R5 | R10 | Last mAP |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
        f'| {args.dataset} | {args.comparison} | {candidate["best_epoch"]} | '+
        ' | '.join(f'{receipt["metrics"][n]:.4f}' for n in ('mAP','Rank-1','Rank-5','Rank-10'))+
        f' | {training["history"][-1]["official_fused"]["mAP"]:.4f} |','',value['boundary']]
    for item in pairs:
        lines+=['',f'{item["candidate"]} minus {item["control"]}: mAP {item["delta_metrics"]["mAP"]:+.4f}, R1 {item["delta_metrics"]["Rank-1"]:+.4f}; first repairs/new errors {item["rank1_repairs"]}/{item["rank1_new_errors"]}.']
    (args.output_dir/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=value['status'],dataset=args.dataset,comparison=args.comparison,
                         metrics=receipt['metrics'],formal_steps=len(steps))),flush=True)


if __name__=='__main__':
    main()
