from pathlib import Path
from collections import Counter
import json,hashlib,math,csv
import numpy as np
OUT=Path(__file__).parent;D=Path(r'C:\Users\gb\.codex_tmp\smooth_ap_q1_complete_20260909')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_bytes())
def write(n,v):
 with (OUT/n).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2)
iv=load(D/'remote_terminal_inventory.json');count=0;size=0
for r in iv['files']:
 p=D/r['path'];assert p.is_file() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];count+=1;size+=p.stat().st_size
summary=load(D/'q1/summary.json');c=summary['comparison'];metrics=load(OUT/'independent_per_query_metrics.json')
arr={e:[] for e in ['control','smooth_ap']};last={e:[] for e in arr};warmup=[]
for fi in range(3):
 for e in arr:
  p=D/'q1'/f'fold_{fi}_{e}';tr=load(p/'training.json');aud=[json.loads(l) for l in (p/'memory_steps.jsonl').read_text().splitlines()]
  for h in tr['history']:
   ep=h['epoch'];lr=.00035*(ep/5 if ep<=5 else .5*(1+math.cos(math.pi*(ep-6)/15)));assert abs(lr-h['learning_rate'])<1e-18
  arr[e].extend(tr['steps']);last[e].extend([(a,t) for a,t in zip(aud,tr['steps']) if t['step']>=196])
 a=load(D/'q1'/f'fold_{fi}_control'/'training.json')['steps'];b=load(D/'q1'/f'fold_{fi}_smooth_ap'/'training.json')['steps'];warmup.append(dict(fold=fi,max_total_difference=max(abs(x['loss']-y['loss']) for x,y in zip(a[:65],b[:65]))))
last_summary={e:dict(batch_hard=float(np.mean([a['original_triplet'] for a,t in last[e]])),expanded_hard=float(np.mean([a['relation_objective']['hard_loss'] for a,t in last[e]])),smooth_ap_loss=float(np.mean([a['relation_objective']['smooth_ap_loss'] for a,t in last[e]])),active_total=float(np.mean([t['loss'] for a,t in last[e]]))) for e in arr}
delta=np.array([r['gains_mAP']['fused'] for r in c['paired_per_identity']]);id_change=dict(improved=int(sum(delta>0)),declined=int(sum(delta<0)),unchanged=int(sum(delta==0)))
rows={e:[r for r in metrics if r['endpoint']==e and r['output']=='fused'] for e in arr};changes=Counter()
for a,b in zip(rows['control'],rows['smooth_ap']):
 assert (a['fold'],a['record_index'])==(b['fold'],b['record_index'])
 dif=b['average_precision']-a['average_precision'];changes['improved' if dif>0 else 'declined' if dif<0 else 'unchanged']+=1
 changes['rank1_repaired']+=(a['first_match_rank']>1 and b['first_match_rank']==1);changes['rank1_new_errors']+=(a['first_match_rank']==1 and b['first_match_rank']>1)
txt=load(OUT/'independent_text_results.json');by={}
for e in arr:
 by[e]=dict(fit_seconds=sum(r['fit_elapsed_seconds'] for r in txt['training'] if r['endpoint']==e),fresh_forwards=sum(r['extra_fresh_forwards'] for r in txt['training'] if r['endpoint']==e),vjp_forwards=sum(r['extra_vjp_forwards'] for r in txt['training'] if r['endpoint']==e))
der=load(Path(r'C:\Users\gb\.codex_tmp\smooth_ap_q1_positive_derivative_complete_20260909\result.json'));quant={}
for e in arr:
 dd=[r['phases']['last65'] for r in der['endpoints'] if r['endpoint'].endswith('_'+e)]
 quant[e]={cat:{k:sum(r[cat]['smooth_distance_derivative'][k] for r in dd) for k in ['count','positive','negative','zero','absolute_sum']} for cat in ['all_positive','cross_positive','inverted_positive','inverted_nonmax','inverted_nonmax_cross']}
 for vals in quant[e].values():assert vals['count']==vals['positive']+vals['negative']+vals['zero']
result=dict(status='PASS',scope='No M0 computation; full 57-file intake identity check only; Q1 learning rates, terminal changes and source-phase arithmetic. Executor Float32 derivative aggregation is explicitly labeled.',intake_files=count,intake_bytes=size,last65=last_summary,warmup=warmup,paired_fused_query_changes=dict(changes),paired_fused_identity_changes=id_change,resources=by,fit_time_increase_percent=(by['smooth_ap']['fit_seconds']/by['control']['fit_seconds']-1)*100,executor_float32_derivative_last65_aggregation=quant,independent_float64_census_file='independent_remote_statistics_results.json')
write('additional_text_results.json',result);print(json.dumps(result,indent=2))
