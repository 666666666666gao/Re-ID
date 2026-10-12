"""CPU description of saved queries; no inference or method selection."""
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

work=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
aggregate=work/'ALL_SIX_RESULTS_912.json'
v=json.loads(aggregate.read_text(encoding='utf-8'))
pins={Path(p).resolve():h for p,h in v['inputs'].items()}
rows=[]
inputs={str(aggregate):sha(aggregate),str(Path(__file__)):sha(Path(__file__))}

def rank_bin(rank):
    assert isinstance(rank,int) and rank>=1
    return '1' if rank==1 else '2-5' if rank<=5 else '6-10' if rank<=10 else '>10'

for dataset in ('RGBNT201','RGBNT100','MSVR310'):
    sources={}
    for condition in ('uniform','ambiguity'):
        suffix='908' if dataset!='RGBNT100' else ('911' if condition=='uniform' else '912')
        branch='region_use_remaining_endpoint_20261013_912' if suffix=='912' else 'region_use_endpoint_diagnosis_20261012_907'
        path=work/('results'+suffix)/'logs'/branch/(dataset+'_'+condition)/'SUMMARY.json'
        assert sha(path)==pins[path.resolve()]
        summary=json.loads(path.read_text(encoding='utf-8'))
        assert summary['status']=='ONE_ORIGINAL_ENDPOINT_CPU_DIAGNOSIS_COMPLETE'
        assert (summary['dataset'],summary['comparison'])==(dataset,condition)
        assert [r['epoch'] for r in summary['history']]==list(range(1,51))
        pair=next(p for p in summary['pairs'] if p['control']=='independent_global')
        sources[condition]=(summary,pair)
        inputs[str(path)]=sha(path)
    uniform,ambiguity=[sources[c][1]['query_changes'] for c in ('uniform','ambiguity')]
    assert len(uniform)==len(ambiguity)>0
    groups=defaultdict(list)
    for u,a in zip(uniform,ambiguity):
        for name in ('query_index','identity','control_ap','control_first_rank'):
            assert u[name]==a[name]
        for ap in (u['control_ap'],u['candidate_ap'],a['candidate_ap']):
            assert math.isfinite(ap) and 0<=ap<=1
        item=dict(query_index=u['query_index'],identity=u['identity'],global_ap=u['control_ap'],
                  uniform_ap=u['candidate_ap'],ambiguity_ap=a['candidate_ap'],
                  global_first_rank=u['control_first_rank'],uniform_first_rank=u['candidate_first_rank'],
                  ambiguity_first_rank=a['candidate_first_rank'])
        groups[rank_bin(item['global_first_rank'])].append(item)

    def summarize(label,queries):
        count=len(queries)
        global_mean=math.fsum(q['global_ap'] for q in queries)*100/count
        result=dict(dataset=dataset,global_first_positive_rank_bin=label,queries=count,
                    identities=len({q['identity'] for q in queries}),global_mean_ap=global_mean)
        for condition in ('uniform','ambiguity'):
            ap=condition+'_ap';rank=condition+'_first_rank'
            deltas=[(q[ap]-q['global_ap'])*100 for q in queries]
            result[condition]=dict(mean_ap=math.fsum(q[ap] for q in queries)*100/count,
                within_bin_mean_delta_map_points=math.fsum(deltas)/count,
                contribution_to_all_query_delta_map_points=math.fsum(deltas)/len(uniform),
                rank1_repairs=sum(q['global_first_rank']!=1 and q[rank]==1 for q in queries),
                rank1_new_errors=sum(q['global_first_rank']==1 and q[rank]!=1 for q in queries))
        delta=[(q['ambiguity_ap']-q['uniform_ap'])*100 for q in queries]
        result['ambiguity_minus_uniform']=dict(within_bin_mean_delta_map_points=math.fsum(delta)/count,
            contribution_to_all_query_delta_map_points=math.fsum(delta)/len(uniform),
            rank1_repairs=sum(q['uniform_first_rank']!=1 and q['ambiguity_first_rank']==1 for q in queries),
            rank1_new_errors=sum(q['uniform_first_rank']==1 and q['ambiguity_first_rank']!=1 for q in queries))
        return result

    strata=[summarize(label,groups[label]) for label in ('1','2-5','6-10','>10') if label in groups]
    all_queries=[q for group in groups.values() for q in group]
    pooled=[summarize('global first correct',[q for q in all_queries if q['global_first_rank']==1]),
            summarize('global first wrong',[q for q in all_queries if q['global_first_rank']!=1])]
    for condition in ('uniform','ambiguity'):
        actual=math.fsum(s[condition]['contribution_to_all_query_delta_map_points'] for s in strata)
        assert abs(actual-sources[condition][1]['delta_metrics']['mAP'])<1e-5
        net=sum(s[condition]['rank1_repairs']-s[condition]['rank1_new_errors'] for s in strata)
        assert abs(net*100/len(uniform)-sources[condition][1]['delta_metrics']['Rank-1'])<1e-5
    contrast=next(c for c in v['ambiguity_minus_uniform'] if c['dataset']==dataset)
    actual=math.fsum(s['ambiguity_minus_uniform']['contribution_to_all_query_delta_map_points'] for s in strata)
    assert abs(actual-contrast['delta_metrics']['mAP'])<1e-5
    net=sum(s['ambiguity_minus_uniform']['rank1_repairs']-s['ambiguity_minus_uniform']['rank1_new_errors'] for s in strata)
    assert abs(net*100/len(uniform)-contrast['delta_metrics']['Rank-1'])<1e-5
    assert sum(s['queries'] for s in strata)==len(uniform)
    rows.append(dict(dataset=dataset,queries=len(uniform),strata=strata,pooled=pooled))

out=work/'FIXED_GLOBAL_DIFFICULTY_914.json'
assert not out.exists()
result=dict(status='SAVED_QUERY_FIXED_GLOBAL_DIFFICULTY_CPU_DESCRIPTION',created_at=datetime.now().astimezone().isoformat(),
            rows=rows,inputs=inputs,neural_calls=0,optimizer_updates=0,
            bin_rule='Fixed independent-global first-positive rank:1,2-5,6-10,>10; pooled1 versus>1. No threshold sweep or query selection.',
            boundary='Post-development dataset-GT diagnosis only, all original selected-best queries retained. Same identity may occur in multiple strata, so identity counts are not additive. Query-level within-bin AP means differ from contributions to overall mAP. This is not the actual train-time global margin/weight, a deployable GT-based selector, multiple-seed uncertainty, current same-model decomposition, stronger-teacher evidence, or a unique failure cause. No new NN, training or model selection.')
out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
lines=['# Fixed independent-global difficulty: saved-query CPU diagnosis','',result['boundary'],'',
       '| Dataset | Global first rank | Queries | U−global within-bin AP | A−global within-bin AP | A−U within-bin AP | A−U all-query mAP contribution | A−U repairs/new |',
       '|---|---|---:|---:|---:|---:|---:|---:|']
for row in rows:
    for s in row['strata']+row['pooled']:
        c=s['ambiguity_minus_uniform']
        lines.append(f"| {row['dataset']} | {s['global_first_positive_rank_bin']} | {s['queries']} | {s['uniform']['within_bin_mean_delta_map_points']:+.4f} | {s['ambiguity']['within_bin_mean_delta_map_points']:+.4f} | {c['within_bin_mean_delta_map_points']:+.4f} | {c['contribution_to_all_query_delta_map_points']:+.4f} | {c['rank1_repairs']}/{c['rank1_new_errors']} |")
(work/'FIXED_GLOBAL_DIFFICULTY_914.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(status=result['status'],rows=[dict(dataset=r['dataset'],pooled=r['pooled']) for r in rows])))
