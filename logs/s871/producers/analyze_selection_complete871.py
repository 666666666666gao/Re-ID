from pathlib import Path
from datetime import datetime
import csv,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft');source=base/'signal_selection_complete871/received/results/signal_selection_reference_complete_20261006_871/SUMMARY.json'
packet=base/'selection_complete_analysis871';assert not packet.exists();packet.mkdir()
r=json.loads(source.read_bytes());assert r['accepted']==9 and r['formal_epochs']==450 and r['formal_steps']==19452
campaign=json.loads((base/'signal_selection_complete871/received/logs/signal_selection_reference_v1_20261006_870/campaign.json').read_bytes())
rows=[];epochs=[]
for x in r['rows']:
 assert [h['epoch'] for h in x['history']]==list(range(1,51))
 assert sum(h['steps'] for h in x['history'])==x['formal_steps']
 last=x['history'][-1]['official_fused'];cli=None
 if x['dataset']=='RGBNT100' and x['variant']!='global_only':
  job=next(j for j in campaign['jobs'] if (j['dataset'],j['selection'],j['phase'])==(x['dataset'],x['variant'],'full'))
  s=job['steps'][0];assert s['mode']=='train' and s['exit_code']==0
  cli=(datetime.fromisoformat(s['completed_at'])-datetime.fromisoformat(s['started_at'])).total_seconds()
 rows.append(dict(dataset=x['dataset'],arm=x['variant'],best_epoch=x['best_epoch'],**x['metrics'],last_map=last['mAP'],last_R1=last['Rank-1'],best_to_last_map=x['metrics']['mAP']-last['mAP'],formal_steps=x['formal_steps'],loss_loop_seconds=sum(h['seconds'] for h in x['history']),training_receipt_interval_seconds=x['training_and_epoch_evaluation_seconds'],train_cli_seconds=cli))
 epochs.extend(dict(dataset=x['dataset'],arm=x['variant'],epoch=h['epoch'],steps=h['steps'],mean_loss=h['mean_loss'],loss_loop_seconds=h['seconds'],**h['official_fused']) for h in x['history'])
for name,data in (('RESULTS.csv',rows),('EPOCHS.csv',epochs)):
 with (packet/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
fig,axes=plt.subplots(3,2,figsize=(11,10),constrained_layout=True)
colors=dict(global_only='#274b76',masked='#cc5b32',all_patch='#39856e')
for i,d in enumerate(('RGBNT201','RGBNT100','MSVR310')):
 for j,metric in enumerate(('mAP','Rank-1')):
  ax=axes[i,j]
  for x in r['rows']:
   if x['dataset']!=d:continue
   ax.plot([h['epoch'] for h in x['history']],[h['official_fused'][metric] for h in x['history']],label=x['variant'],color=colors[x['variant']],linewidth=1.4)
   best=x['history'][x['best_epoch']-1];ax.scatter(best['epoch'],best['official_fused'][metric],color=colors[x['variant']],s=32,zorder=4)
  ax.set(title=f'{d} — {metric}',xlabel='Epoch',ylabel='Percentage points',xlim=(1,50));ax.grid(alpha=.2)
  if i==0 and j==0:ax.legend(fontsize=8)
fig.suptitle('Nine complete 50-epoch runs — markers follow each mAP-best checkpoint',fontsize=13)
for suffix in ('png','pdf','svg'):fig.savefig(packet/f'TRAJECTORIES.{suffix}',dpi=160)
plt.close(fig)
primary=[]
for p in r['pairs']:
 if p['control']!='all_patch':continue
 q=p['paired_diagnosis'];primary.append(dict(dataset=p['dataset'],advance=p['phase_progress'],delta=q['delta_metrics'],repairs=q['rank1_repairs'],new_errors=q['rank1_new_errors'],query_ap_improved=q['query_ap_improved'],query_ap_worsened=q['query_ap_worsened'],identity_macro_delta=q['identity_macro_mean_delta_ap_points'],identity_bootstrap95=q['identity_bootstrap_95_percentile_interval']))
facts=dict(status='CLOSED_NINE_RUN_LOCAL_ANALYSIS',at=datetime.now().astimezone().isoformat(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),formal_epochs=450,formal_steps=19452,rows=rows,primary=primary,body_report_exit=0,original870_exit=1,boundary='Read-only summary/trajectory analysis; no new score/NN/eval/report replay. mAP-best markers apply to R1 too. Individual epochs and identity bootstrap are not training seeds. CLI fields only available for two newRGBNT100 runs; other time fields explicit loop vs receipt intervals. Shared-server sequential cost is not a causal performance benchmark. Primary advancement0/3, no scientific success/SOTA.')
(packet/'FACTS.json').write_text(json.dumps(facts,indent=2)+'\n',encoding='utf-8')
(packet/'NOTE.md').write_text(facts['boundary']+'\n',encoding='utf-8')
print(json.dumps(dict(status=facts['status'],primary=primary,RGBNT100=[x for x in rows if x['dataset']=='RGBNT100']),indent=2))
