"""One CPU full-query report for the nine matched source references."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import queue_signal_selection_reference as panel
from tools.analyze_correspondence_distances import compare,camera_scores,scene_scores

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--origin-campaigns',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    panel.require_sources();panel.require_protected()
    state=json.loads((args.campaign/'campaign.json').read_text())
    assert state['status']=='COMPLETE' and state['report_invocations']==1
    assert len(state['jobs'])==18 and all(r['status']=='COMPLETE' and r['exit_code']==0 for r in state['jobs'])
    matrix=json.loads((args.campaign/'accepted_matrix.json').read_text())
    origins=json.loads(args.origin_campaigns.read_text())
    assert set(origins)=={f'{d}:{s}' for d in panel.DATASETS for s in panel.SELECTIONS}
    assert matrix['accepted']==matrix['expected']==len(matrix['rows'])==9
    assert not args.output_dir.exists()
    torch.set_num_threads(1)
    rows=[]
    for row in matrix['rows']:
        assert panel.accepted_row(Path(origins[f'{row["dataset"]}:{row["variant"]}']),row['dataset'],row['variant'])==row
        training=json.loads((Path(row['run_dir'])/'training.json').read_text())
        rows.append(dict(row,history=training['history'],selection_reference=training['selection_reference'],
            training_and_epoch_evaluation_seconds=(datetime.fromisoformat(training['completed_at'])-datetime.fromisoformat(training['started_at'])).total_seconds()))
    pairs=[]
    for dataset in panel.DATASETS:
        selected={r['variant']:r for r in rows if r['dataset']==dataset}
        batch_equal=all((Path(selected[s]['run_dir'])/'training_batch_metadata.jsonl').read_bytes()==(Path(selected['global_only']['run_dir'])/'training_batch_metadata.jsonl').read_bytes() for s in panel.SELECTIONS)
        assert batch_equal
        for candidate,control in (('masked','all_patch'),('masked','global_only'),('all_patch','global_only')):
            paired=compare(matrix,dataset,control,candidate)
            scorer=scene_scores if dataset=='MSVR310' else camera_scores
            environment='scenes' if dataset=='MSVR310' else 'cameras'
            scores={}
            for label in (control,candidate):
                data=torch.load(Path(selected[label]['run_dir'])/'official_distances.pt',map_location='cpu',weights_only=False)
                scores[label]=scorer(data['fused'].numpy(),data['query_ids'],data['gallery_ids'],data['query_'+environment],data['gallery_'+environment])
            paired['query_changes']=[dict(query_index=i,identity=int(identity),control_ap=float(scores[control]['average_precision'][i]),candidate_ap=float(scores[candidate]['average_precision'][i]),control_first_rank=int(scores[control]['first_match_rank'][i]),candidate_first_rank=int(scores[candidate]['first_match_rank'][i])) for i,identity in enumerate(data['query_ids'])]
            paired['boundary']='Single seed42, consumed official benchmark. Masked/all-patch matched3072 capacity/head; SIM/global mixes capacity, extra head, visual objectives and dimension. Metadata equality not augmentation-byte equality. Identity bootstrap not training seeds.'
            delta=paired['delta_metrics']
            pairs.append(dict(dataset=dataset,candidate=candidate,control=control,actual_batch_metadata_equal=batch_equal,phase_progress=delta['mAP']>=.5 and delta['Rank-1']>=0,paired_diagnosis=paired))
    result=dict(schema=panel.SCHEMA,status='COMPLETE',created_at=panel.stamp(),accepted=9,formal_epochs=450,
        formal_steps=sum(r['formal_steps'] for r in rows),rows=rows,pairs=pairs,
        primary_selection_progress_count=sum(p['phase_progress'] for p in pairs if p['control']=='all_patch'),
        boundary='Source-neighbor reference, not new TriFusion contribution. Full author-source RAW global/var joint objectives and3072 SIM retrieval; fixedtopk80/64/112; noAlignM, roles, shared adapters or native CNN. All primarydataset pairs published; stability/SOTA unproven.')
    args.output_dir.mkdir(parents=True)
    panel.write(args.output_dir/'SUMMARY.json',result)
    lines=['# Signal选择来源参考：完整九端','', '| Dataset | Arm | Best epoch | mAP | R1 | Steps |','|---|---|---:|---:|---:|---:|']
    lines += [f'| {r["dataset"]} | {r["variant"]} | {r["best_epoch"]} | {r["metrics"]["mAP"]:.4f} | {r["metrics"]["Rank-1"]:.4f} | {r["formal_steps"]} |' for r in rows]
    lines+=['',result['boundary']]
    (args.output_dir/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='COMPLETE',accepted=9,pairs=len(pairs))),flush=True)

if __name__=='__main__':main()
