"""Read a closed monotone-AP endpoint and its matched F1/R2 control; CPU only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean,median

ROLES=('cnn','transformer','mamba')


def distribution(values):
    values=list(values)
    assert all(math.isfinite(x) for x in values)
    if not values:return dict(count=0)
    return dict(count=len(values),minimum=min(values),median=median(values),mean=mean(values),maximum=max(values))


def paired(reference,fused):
    n=len(reference['first_match_rank']);assert len(fused['first_match_rank'])==n
    groups={k:dict(count=0,map_contribution_pp=0.) for k in ('kept_correct','repaired','new_error','still_wrong')}
    for before,after,ap0,ap1 in zip(reference['first_match_rank'],fused['first_match_rank'],reference['average_precision'],fused['average_precision'],strict=True):
        key='kept_correct' if before==after==1 else 'repaired' if before>1 and after==1 else 'new_error' if before==1 and after>1 else 'still_wrong'
        groups[key]['count']+=1;groups[key]['map_contribution_pp']+=(ap1-ap0)*100/n
    delta={k:v-reference['metrics'][k] for k,v in fused['metrics'].items()}
    assert abs(sum(g['map_contribution_pp'] for g in groups.values())-delta['mAP'])<1e-10
    return dict(metric_delta_pp=delta,rank1_repairs=groups['repaired']['count'],rank1_new_errors=groups['new_error']['count'],rank1_net_hits=groups['repaired']['count']-groups['new_error']['count'],ap_groups=groups)


def summarize_steps(rows):
    assert [r['step'] for r in rows]==list(range(1,len(rows)+1))
    assert all(r['active_fused_metric'] in ('hard_triplet','legal_monotone_ap') for r in rows)
    phases=dict(hard_triplet=[r for r in rows if r['active_fused_metric']=='hard_triplet'],
                monotone_supported=[r for r in rows if r['active_fused_metric']=='legal_monotone_ap' and r['eligible_anchors']>0],
                monotone_without_support=[r for r in rows if r['active_fused_metric']=='legal_monotone_ap' and r['eligible_anchors']==0])
    result={}
    for phase,selected in phases.items():
        roles={}
        for role in ROLES:
            values=[row['gradient_update_statistics'][role] for row in selected]
            cosine=[v['ranking_auxiliary_cosine'] for v in values if v['ranking_auxiliary_cosine'] is not None]
            history_cosine=[v['current_history_ranking_cosine'] for v in values if v['current_history_ranking_cosine'] is not None]
            direction=[v['ranking_dot_actual_parameter_delta']/(v['ranking_norm']*v['actual_parameter_delta_norm']) for v in values if v['ranking_norm']>0 and v['actual_parameter_delta_norm']>0]
            roles[role]=dict(ranking_norm=distribution(v['ranking_norm'] for v in values),auxiliary_norm=distribution(v['auxiliary_norm'] for v in values),
                auxiliary_to_ranking_norm_ratio=distribution(v['auxiliary_norm']/v['ranking_norm'] for v in values if v['ranking_norm']>0),
                zero_ranking_norm_steps=sum(v['ranking_norm']==0 for v in values),
                ranking_auxiliary_cosine=distribution(cosine),ranking_auxiliary_cosine_below_minus1e_minus6=sum(v<-1e-6 for v in cosine),
                current_history_cosine=distribution(history_cosine),
                actual_step_ranking_cosine=distribution(direction),actual_step_ranking_cosine_above_1e_minus6=sum(v>1e-6 for v in direction),
                raw_positive_ranking_dot_actual_delta_steps=sum(v['ranking_dot_actual_parameter_delta']>0 for v in values),
                rank_coefficient_at_1p6_cap_steps=sum(abs(row['role_weights'][role][0]-1.6)<1e-12 for row in selected))
        result[phase]=dict(updates=len(selected),roles=roles)
    samples=[dict(epoch=row['epoch'],step=row['step'],**row['score_statistics']) for row in rows if row['score_statistics'] is not None]
    assert len({s['epoch'] for s in samples})==len(samples)
    return dict(total_updates=len(rows),phases=result,score_samples=samples,
        score_sampling='First eligible update of each epoch only, not a whole-training score-gradient frequency.',
        interpretation='Measurements cover role encoder blocks, excluding classification heads and frozen F1. Raw norm ratios are not AdamW update shares. Positive g_rank dot actual_delta is local first-order ascent on the current training objective, not proof of finite-step loss increase or test damage. Missing/zero-norm cosines remain unavailable.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--control-receipt',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve();assert not args.output.exists()
    training=json.loads((run/'training.json').read_bytes());receipt=json.loads((run/'official_metrics.json').read_bytes())
    control=json.loads(args.control_receipt.read_bytes())
    assert receipt['schema']=='trifusion-f1-legal-monotone-ap-v1' and receipt['status']=='COMPLETE'
    assert control['schema']=='trifusion-f1-initialized-r2-v1' and control['status']=='COMPLETE'
    assert training['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE'
    assert training['training']['epochs']==receipt['training_epochs']==control['training_epochs']==50
    assert receipt['seed']==control['seed']==42
    for key in ('dataset','foundation_checkpoint_sha256','protocol_sha256','query_count','gallery_count','filter'):
        assert receipt[key]==control[key],key
    assert receipt['baseline_f1_metrics_equal'] and receipt['independent_upstream_metrics_equal']
    assert training['training']['frozen_state_unchanged'] and training['training']['missing_nonzero_gradients']==[] and training['training']['overflow_events']==0
    epochs=[json.loads(x) for x in (run/'epoch_official_metrics.jsonl').read_text().splitlines()]
    assert [e['epoch'] for e in epochs]==list(range(1,51))
    best=max(epochs,key=lambda e:(e['metrics']['mAP'],e['epoch']))
    assert best['epoch']==training['selected_epoch']==receipt['selected_epoch']
    assert best['metrics']==training['selected_metrics']
    assert all(abs(receipt['outputs']['fused']['metrics'][k]-v)<1e-5 for k,v in best['metrics'].items())
    assert sha(run/'roles_best_map.pth')==receipt['role_checkpoint_sha256']==training['checkpoint_sha256']
    assert sha(receipt['distance_arrays'])==receipt['distance_arrays_sha256']
    assert sha(control['distance_arrays'])==control['distance_arrays_sha256']
    assert all(abs(receipt['outputs']['baseline_only']['metrics'][k]-v)<1e-5 for k,v in control['outputs']['baseline_only']['metrics'].items())
    steps=[json.loads(x) for x in (run/'training_steps.jsonl').read_text().splitlines()]
    summary=summarize_steps(steps)
    assert len(steps)==training['training']['optimizer_steps'] and summary['phases']['hard_triplet']['updates']==65
    assert summary['phases']['monotone_supported']['updates']==training['training']['supported_steps']
    result=dict(status='CPU_CLOSED_F1_MONOTONE_TERMINAL_REPORT',dataset=receipt['dataset'],seed=42,selected_epoch=receipt['selected_epoch'],
        metrics=receipt['outputs']['fused']['metrics'],candidate_vs_matched_R2=paired(control['outputs']['fused'],receipt['outputs']['fused']),
        fused_vs_f1=receipt['fused_vs_f1'],fused_vs_full_signal=receipt['fused_vs_full_signal'],
        fused_vs_pure_bank=paired(receipt['outputs']['pure_bank'],receipt['outputs']['fused']),training_diagnostics=summary,
        evidence_sha256=dict(training=sha(run/'training.json'),official=sha(run/'official_metrics.json'),epoch_history=sha(run/'epoch_official_metrics.jsonl'),
          training_steps=sha(run/'training_steps.jsonl'),control_receipt=sha(args.control_receipt),role=receipt['role_checkpoint_sha256'],actual_distances=receipt['distance_arrays_sha256']),
        scope='One fixed complete best model per method after full50; diagnostics describe this candidate trajectory, not a counterfactual replay of the old control. No NN, new ranking, optimization, seed averaging or causal/originality certificate.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],dataset=result['dataset'],metrics=result['metrics'],matched_R2_delta=result['candidate_vs_matched_R2']['metric_delta_pp'])))


if __name__=='__main__':main()
