"""Independent text/CSV arithmetic checks over all received records. Stdlib only."""
import collections,csv,hashlib,io,json,math
from pathlib import Path
from inspect_inputs import OUT,REPO,record
INTAKE=Path('C:/Users/gb/.codex_tmp/history_gradient_q1_complete_20260908')
PROCESS=Path('C:/Users/gb/.codex_tmp/history_gradient_q1_terminal_processing_20260908')
def get(path):return json.loads(record(path))
def mean(v):return math.fsum(v)/len(v)
def q(v,p):
 a=sorted(v);t=(len(a)-1)*p;i=math.floor(t);j=math.ceil(t);return a[i]+(a[j]-a[i])*(t-i)
def moment(vals):
 v=[float(x) for x in vals];assert all(math.isfinite(x) for x in v)
 if not v:return {'count':0}
 return dict(count=len(v),mean=mean(v),minimum=min(v),p05=q(v,.05),median=q(v,.5),p95=q(v,.95),maximum=max(v))
checked_numeric=0
def compare(actual,expected,path='root'):
 global checked_numeric
 if isinstance(expected,dict):
  assert set(expected)<=set(actual),(path,'keys')
  for k,v in expected.items():compare(actual[k],v,path+'.'+str(k))
 elif isinstance(expected,list):
  assert len(actual)==len(expected),(path,'length')
  for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,path+f'[{i}]')
 elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
  assert abs(actual-expected)<=1e-10*max(1,abs(expected)),(path,actual,expected);checked_numeric+=1
 else: assert actual==expected,(path,actual,expected)
def aggregate(rows,saved):
 hist=[r for r in rows if r['memory']];roles={}
 for role in ('cnn','transformer','mamba'):
  val={}
  for key in ('total_vs_history','total_vs_both'):
   group=[r['roles'][role][key] for r in hist];cos=[g['cosine'] for g in group if g['cosine'] is not None]
   val[key]=dict(cosine=moment(cos),undefined_cosines=sum(g['cosine'] is None for g in group),negative_cosines=sum(c<0 for c in cos),**{name:moment([g[name] for g in group]) for name in ('first_norm','second_norm','difference_norm')})
  val['history_to_current_total_norm_ratio']=moment([r['roles'][role]['total_vs_history']['second_norm']/r['roles'][role]['total_vs_history']['first_norm'] for r in hist if r['roles'][role]['total_vs_history']['first_norm']>0])
  val['actually_applied_difference_norm']=moment([r['applied_gradients'][role]['difference_norm'] for r in hist]);val['updates_with_applied_difference']=sum(r['applied_gradients'][role]['difference_norm']>0 for r in hist);roles[role]=val
 ages=collections.Counter(m['age'] for r in rows for m in r['memory']);available=sum(len({m['stored_step'] for m in r['memory']}) for r in rows);selected=sum(len(r['history_vjp_groups']) for r in rows)
 keys=('memory_positive_pairs','memory_negative_pairs','memory_cross_scene_positive_pairs','memory_negative_violations_against_batch_hard_positive','harder_positive_anchors','harder_negative_anchors','current_wrong_order_anchors','expanded_wrong_order_anchors','expanded_hinge_positive_anchors')
 return dict(updates=len(rows),historical_updates=len(hist),historical_current_anchor_exposures=64*len(hist),history_anchor_count=0,candidate_record_exposures=sum(ages.values()),ages={str(k):v for k,v in sorted(ages.items())},available_history_group_exposures=available,selected_vjp_group_exposures=selected,zero_upstream_group_skip_exposures=available-selected,fresh_role_record_forwards=sum(r['fresh_role_record_forwards'] for r in rows),history_vjp_record_forwards=sum(r['history_vjp_record_forwards'] for r in rows),mining_exposures={k:sum(r['statistics'][k] for r in rows) for k in keys},current_triplet=moment([r['statistics']['current_triplet'] for r in rows]),expanded_triplet=moment([r['statistics']['expanded_triplet'] for r in rows]),total_loss=moment([r['loss'] for r in saved]),all14_components={k:moment([r['components'][k] for r in saved]) for k in saved[0]['components']},roles=roles)

replay=get(OUT/'remote_replay.stdout.json');analysis=get(PROCESS/'training_text_analysis.json');rankanalysis=get(PROCESS/'rankings/ranking_replay.json');summary=get(INTAKE/'q1/summary.json');cpu=get(INTAKE/'q1_cpu.json');pipeline=get(INTAKE/'pipeline.json');state=get(PROCESS/'state.json');plan=get(Path('C:/Users/gb/.codex_tmp/history_gradient_terminal_watcher_plan_20260908.json'))
assert state['status']=='COMPLETE_LOCAL_TEXT_AND_RANKING_VERIFIED_AWAITING_INDEPENDENT_AUDIT'
assert state['terminal_summary_sha256']==pipeline['terminal_summary_sha256']==cpu['summary_sha256']==hashlib.sha256(record(INTAKE/'q1/summary.json')).hexdigest()
assert state['training_analysis_sha256']==hashlib.sha256(record(PROCESS/'training_text_analysis.json')).hexdigest()
assert state['ranking_analysis_sha256']==hashlib.sha256(record(PROCESS/'rankings/ranking_replay.json')).hexdigest()
assert state['plan_sha256']==hashlib.sha256(record(Path('C:/Users/gb/.codex_tmp/history_gradient_terminal_watcher_plan_20260908.json'))).hexdigest()
for p,sha in plan['script_sha256'].items():assert hashlib.sha256(record(p)).hexdigest()==sha
assert all(s['exit_code']==0 for s in state['stages']) and [s['name'] for s in state['stages'][-3:]]==['intake','training_text','ranking_text']
epochs=[];ends=[];training_map={}
for f in summary['folds']:
 for end,item in f['endpoints'].items():
  directory=INTAKE/'q1'/f'fold_{f["fold"]}_{end}';tr=get(directory/'training.json');assert item==get(directory/'receipt.json') and tr==item['training']
  rows=[json.loads(x) for x in record(directory/'memory_steps.jsonl').splitlines()]
  total=aggregate(rows,tr['steps']);actual=next(x for x in analysis['endpoints'] if x['fold']==f['fold'] and x['endpoint']==end);compare(actual,total)
  compare(actual,dict(peak_allocated_mib=tr['peak_allocated_mib'],training_seconds=sum(x['elapsed_seconds'] for x in tr['history'])))
  ends.append(dict(fold=f['fold'],endpoint=end,**total));training_map[f['fold'],end]=tr
  for ep in range(1,21):
   rr=[r for r,t in zip(rows,tr['steps']) if t['epoch']==ep];tt=[t for t in tr['steps'] if t['epoch']==ep];total=aggregate(rr,tt)
   actual=next(x for x in analysis['all_epochs'] if x['fold']==f['fold'] and x['endpoint']==end and x['epoch']==ep);compare(actual,total);epochs.append(dict(fold=f['fold'],endpoint=end,epoch=ep,**total))
assert len(epochs)==len(analysis['all_epochs'])==120 and len(ends)==6
assert sum(e['updates'] for e in ends)==analysis['total_updates']==1560
logevents=[json.loads(line) for line in record(INTAKE/'q1.log').decode().splitlines() if line.startswith('{"event": "history_gradient_epoch"')]
assert len(logevents)==120
for event in logevents:
 tr=training_map[event['fold'],event['endpoint']];epoch=tr['history'][event['epoch']-1];compare(event,epoch)
cpulog=[json.loads(line) for line in record(INTAKE/'q1_cpu.log').decode().splitlines() if line.startswith('{')]
assert len(cpulog)==1;compare(cpu,cpulog[0])

queries=[]
for o in ('baseline_only','fused','cnn','transformer','mamba'):
 for a,b in zip(replay['query_records']['control'][o],replay['query_records']['history_gradient'][o]):
  key=('fold','record_index','identity','scene','positive_count');assert all(a[k]==b[k] for k in key)
  r={k:a[k] for k in key};r.update(output=o,delta_ap_pp=(b['ap']-a['ap'])*100,rank1_repaired=a['first_match_rank']>1 and b['first_match_rank']==1,rank1_new_error=a['first_match_rank']==1 and b['first_match_rank']>1)
  for tag,val in [('control',a),('candidate',b)]:r.update({tag+'_'+k:v for k,v in val.items() if k not in key})
  queries.append(r)
for filename,expected in [('all3000_query_output_changes.csv',queries),('all300_identity_output_changes.csv',replay['identity_records'])]:
 raw=record(PROCESS/'rankings'/filename);actual=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))));assert len(actual)==len(expected)
 for i,(a,b) in enumerate(zip(actual,expected)):
  assert set(a)==set(b)
  for k,v in b.items():
   parsed=a[k]=='True' if isinstance(v,bool) else float(a[k]) if isinstance(v,(int,float)) else a[k]
   compare(parsed,v,f'{filename}:{i+2}:{k}')
 assert rankanalysis['csv_sha256'][filename]==hashlib.sha256(raw).hexdigest()
 with (OUT/('independent_'+filename)).open('w',newline='',encoding='utf-8') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(expected[0]));writer.writeheader();writer.writerows(expected)
compare(rankanalysis['paired_changes'],replay['paired']['changes']);compare(rankanalysis['paired_gains'],replay['paired']['gains']);compare(rankanalysis['paired_fold_gains'],replay['paired']['fold_fused_gains']);compare(rankanalysis['paired_lower_bound'],replay['paired']['bootstrap_lower_pp']);compare(rankanalysis['paired_gates'],replay['paired']['checks'])
for end,item in replay['endpoints'].items():
 compare(rankanalysis['endpoints'][end],dict(metrics=item['metrics'],gains_over_signal=item['gains'],fold_gains=item['fold_fused_gains'],lower_bound=item['bootstrap_lower_pp'],gates=item['checks']))
assert rankanalysis['query_count']==600 and rankanalysis['checked_rank_positions']==2069520
assert rankanalysis['scientific_status']==summary['status']==replay['scientific_status']=='Q1_FAIL'
(OUT/'independent_training_aggregates.json').write_text(json.dumps(dict(endpoints=ends,epochs=epochs),indent=2)+'\n')
result=dict(status='PASS_INDEPENDENT_ALL_LOCAL_TEXT_CSV_VALIDATION',training_updates=1560,endpoint_aggregates=6,epoch_aggregates=120,numeric_fields_compared=checked_numeric,query_output_csv_rows=3000,identity_output_csv_rows=300,epoch_log_events=120,cpu_log_records=1,terminal_state_and_watcher_bindings_valid=True,all_manifest_files_read=True)
(OUT/'local_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
