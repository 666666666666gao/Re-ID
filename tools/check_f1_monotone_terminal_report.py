"""CPU checks of report counts and interpretation; no full-endpoint verdict."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.report_f1_monotone_terminal import paired,summarize_steps


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m0-steps',type=Path,required=True)
    args=parser.parse_args()
    before=dict(first_match_rank=[1,2,1,3],average_precision=[.8,.4,.5,.2],metrics={'mAP':47.5,'Rank-1':50.})
    after=dict(first_match_rank=[2,1,1,3],average_precision=[.4,.8,.6,.25],metrics={'mAP':51.25,'Rank-1':50.})
    result=paired(before,after)
    assert result['rank1_repairs']==result['rank1_new_errors']==1 and result['rank1_net_hits']==0
    assert all(g['count']==1 for g in result['ap_groups'].values())
    assert result['metric_delta_pp']=={'mAP':3.75,'Rank-1':0.}
    rows=[json.loads(x) for x in args.m0_steps.read_text().splitlines()]
    actual=summarize_steps(rows)
    assert actual['total_updates']==8 and actual['phases']['hard_triplet']['updates']==2
    assert actual['phases']['monotone_supported']['updates']==6 and actual['phases']['monotone_without_support']['updates']==0
    assert len(actual['score_samples'])==1 and actual['score_samples'][0]['scope'].startswith('first eligible update per epoch')
    first=copy.deepcopy(rows[2]);first.update(step=1,epoch=1,eligible_anchors=1,score_statistics=None)
    for role in ('cnn','transformer','mamba'):
        first['gradient_update_statistics'][role]=dict(ranking_norm=1.,auxiliary_norm=2.,ranking_auxiliary_cosine=-.5,
            current_history_ranking_cosine=None,ranking_dot_actual_parameter_delta=.25,actual_parameter_delta_norm=.5)
    second=copy.deepcopy(first);second['step']=2
    for role in ('cnn','transformer','mamba'):
        second['gradient_update_statistics'][role].update(ranking_norm=0.,ranking_auxiliary_cosine=None,ranking_dot_actual_parameter_delta=0.)
    checked=summarize_steps([first,second])['phases']['monotone_supported']['roles']
    for role,row in checked.items():
        assert row['actual_step_ranking_cosine_above_1e_minus6']==row['zero_ranking_norm_steps']==1
        assert row['ranking_auxiliary_cosine_below_minus1e_minus6']==1
        assert row['actual_step_ranking_cosine']['count']==1 and row['actual_step_ranking_cosine']['mean']==.5
        assert row['current_history_cosine']=={'count':0}
    print(json.dumps(dict(status='CPU_REPORT_COMPONENT_CHECKS_PASS',actual_m0_updates=8,scope='Report mathematics and genuine eight-step log parsing only; full50 CLI and scientific results remain unexecuted. NN calls0, remote queries0.')))


if __name__=='__main__':main()
