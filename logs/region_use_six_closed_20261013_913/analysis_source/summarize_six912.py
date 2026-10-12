import hashlib,json,math
from pathlib import Path
from datetime import datetime
work=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs={}
for suffix in ('908','911','912'):
 folder=work/('results'+suffix)
 manifest=folder/('INTAKE_MANIFEST_'+suffix+'.json')
 value=json.loads(manifest.read_text())
 inputs[str(manifest)]=sha(manifest)
 for name,digest in value['files'].items():
  p=folder/name;assert sha(p)==digest,name
rows=[]
saved={}
for d in ('RGBNT201','MSVR310','RGBNT100'):
 for c in ('uniform','ambiguity'):
  suffix='908' if d!='RGBNT100' else ('911' if c=='uniform' else '912')
  branch='region_use_endpoint_diagnosis_20261012_907' if suffix!='912' else 'region_use_remaining_endpoint_20261013_912'
  p=work/('results'+suffix)/'logs'/branch/(d+'_'+c)/'SUMMARY.json'
  v=json.loads(p.read_text());assert v['status']=='ONE_ORIGINAL_ENDPOINT_CPU_DIAGNOSIS_COMPLETE'
  assert (v['dataset'],v['comparison'])==(d,c)
  assert [h['epoch'] for h in v['history']]==list(range(1,51))
  selected=max(v['history'],key=lambda h:(h['official_fused']['mAP'],h['epoch']))
  assert selected['epoch']==v['candidate']['best_epoch']
  assert all(abs(selected['official_fused'][k]-n)<1e-5 for k,n in v['metrics'].items())
  inputs[str(p)]=sha(p)
  saved[d,c]=v
  rows.append(dict(dataset=d,comparison=c,best_epoch=v['candidate']['best_epoch'],
   metrics=v['metrics'],last_metrics=v['history'][-1]['official_fused'],
   formal_recorded_steps=v['formal_recorded_steps'],candidate=v['candidate'],
   best_to_last_map_drop=v['best_to_last_map_drop'],
   controls=[{k:p[k] for k in ('control','delta_metrics','rank1_repairs','rank1_new_errors',
     'rank1_group_ap_contributions','worst_five_percent_query_ap_delta_mean_points')} for p in v['pairs']]))
contrasts=[]
for d in ('RGBNT201','MSVR310','RGBNT100'):
 first,second=saved[d,'uniform'],saved[d,'ambiguity']
 u,a=[next(p for p in s['pairs'] if p['control']=='raw_semantic') for s in (first,second)]
 q1,q2=u['query_changes'],a['query_changes']
 assert len(q1)==len(q2) and len(q1)>0
 groups={k:[] for k in ('repaired','new_error','first_still_correct','first_still_wrong')}
 query_changes=[]
 for left,right in zip(q1,q2):
  for name in ('query_index','identity','control_ap','control_first_rank'):
   assert left[name]==right[name]
  before,after=left['candidate_first_rank']==1,right['candidate_first_rank']==1
  group=('first_still_correct' if after else 'new_error') if before else ('repaired' if after else 'first_still_wrong')
  delta=(right['candidate_ap']-left['candidate_ap'])*100
  groups[group].append(delta)
  query_changes.append(dict(query_index=left['query_index'],identity=left['identity'],
   uniform_ap=left['candidate_ap'],ambiguity_ap=right['candidate_ap'],delta_ap_points=delta,
   uniform_first_rank=left['candidate_first_rank'],ambiguity_first_rank=right['candidate_first_rank']))
 count=len(query_changes)
 metric_delta={k:second['metrics'][k]-first['metrics'][k] for k in first['metrics']}
 assert abs(math.fsum(q['delta_ap_points'] for q in query_changes)/count-metric_delta['mAP'])<1e-5
 repairs,new=len(groups['repaired']),len(groups['new_error'])
 assert abs((repairs-new)*100/count-metric_delta['Rank-1'])<1e-5
 ordered=sorted(q['delta_ap_points'] for q in query_changes)
 contrasts.append(dict(dataset=d,control='uniform',candidate='ambiguity',queries=count,
  delta_metrics=metric_delta,rank1_repairs=repairs,rank1_new_errors=new,
  group_ap_contributions={k:dict(queries=len(values),contribution_to_total_delta_map_points=math.fsum(values)/count) for k,values in groups.items()},
  worst_five_percent_query_ap_delta_mean_points=math.fsum(ordered[:math.ceil(.05*count)])/math.ceil(.05*count),
  query_changes=query_changes))
out=work/'ALL_SIX_RESULTS_912.json'
assert not out.exists()
result=dict(status='SIX_ORIGINAL_FULL50_FIRST_STRICT_RESULTS_COLLECTED',
 created_at=datetime.now().astimezone().isoformat(),rows=rows,ambiguity_minus_uniform=contrasts,
 total_formal_updates=sum(r['formal_recorded_steps'] for r in rows),total_formal_epochs=300,
 inputs=inputs,neural_calls=0,optimizer_updates=0,
 boundary='Six original accepted seed42 endpoints; fifth/sixth straddled the preserved original storage failure. Saved-query CPU contrasts use each original mAP-best with its own CMC. All six tried results retained. No model/teacher promotion, no new inference/training or coefficient/seed sweep; same-model five-score diagnosis remains unexecuted.')
out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=result['status'],formal_updates=result['total_formal_updates'],
 rows=[{k:r[k] for k in ('dataset','comparison','best_epoch','metrics','last_metrics')} for r in rows],
 contrasts=[{k:c[k] for k in ('dataset','delta_metrics','rank1_repairs','rank1_new_errors')} for c in contrasts])))
