"""Analyze the three closed MSVR310 references from original text receipts only."""
from pathlib import Path
from datetime import datetime
import csv,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
packet=base/'selection_msvr_analysis868';assert not packet.exists()
observation_path=base/'signal_selection_rgb100_start868/OBSERVATION.json'
observed=json.loads(observation_path.read_bytes())
inputs={
 'global_only':('selection_counter_failure867_r2','text_files'),
 'masked':('selection_masked_msvr_complete868_r2','files'),
 'all_patch':('selection_allpatch_msvr_complete868','files'),
}
source_sha={str(observation_path):hashlib.sha256(observation_path.read_bytes()).hexdigest()}
rows=[];epochs=[]
for arm,(name,key) in inputs.items():
 intake=base/name;remote_path=intake/'REMOTE.json';remote=json.loads(remote_path.read_bytes())
 source_sha[str(remote_path)]=hashlib.sha256(remote_path.read_bytes()).hexdigest()
 job=next(j for j in observed['campaign']['jobs'] if (j['dataset'],j['selection'],j['phase'])==('MSVR310',arm,'full'))
 assert job['status']=='COMPLETE' and job['exit_code']==0 and all(s['exit_code']==0 for s in job['steps'])
 row=job['result'];prefix=row['run_dir'].removeprefix('/data/gaob/Re-ID/Trifusion/')
 folder=intake/'received'/prefix
 for file in ('training.json','official_metrics.json'):
  p=folder/file;info=remote[key][prefix+'/'+file]
  assert p.stat().st_size==info['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256']
  source_sha[str(p)]=info['sha256']
 training=json.loads((folder/'training.json').read_bytes());receipt=json.loads((folder/'official_metrics.json').read_bytes())
 history=training['history'];assert [h['epoch'] for h in history]==list(range(1,51))
 assert training['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and receipt['status']=='COMPLETE'
 best=max(history,key=lambda h:(h['official_fused']['mAP'],h['epoch']))
 assert best['epoch']==row['best_epoch']==receipt['selected_epoch']
 assert sum(h['steps'] for h in history)==row['formal_steps']==706
 assert all(abs(best['official_fused'][k]-v)<1e-5 for k,v in row['metrics'].items())
 assert hashlib.sha256((folder/'official_metrics.json').read_bytes()).hexdigest()==row['receipt_sha256']
 assert receipt['checkpoint_sha256']==row['checkpoint_sha256'] and receipt['distance_sha256']==row['distance_sha256']
 times={s['mode']:(datetime.fromisoformat(s['completed_at'])-datetime.fromisoformat(s['started_at'])).total_seconds() for s in job['steps']}
 rows.append(dict(arm=arm,feature_width=row['initializer']['feature_width'],trainable_parameters=row['initializer']['trainable_parameters'],best_epoch=row['best_epoch'],formal_steps=706,best_metrics=row['metrics'],last_metrics=history[-1]['official_fused'],best_to_last_map_drop=row['metrics']['mAP']-history[-1]['official_fused']['mAP'],training_loop_seconds=sum(h['seconds'] for h in history),loop_with_epoch_evaluation_save_seconds=(datetime.fromisoformat(training['completed_at'])-datetime.fromisoformat(training['started_at'])).total_seconds(),full_train_cli_seconds=times['train'],first_strict_cli_seconds=times['evaluate'],checkpoint_sha256=row['checkpoint_sha256'],receipt_sha256=row['receipt_sha256']))
 for h in history:epochs.append(dict(arm=arm,epoch=h['epoch'],steps=h['steps'],mean_loss=h['mean_loss'],training_loop_seconds=h['seconds'],**h['official_fused']))
assert len(rows)==3 and len(epochs)==150
arms={r['arm']:r for r in rows};a,b=arms['masked'],arms['all_patch']
assert a['feature_width']==b['feature_width']==3072 and a['trainable_parameters']==b['trainable_parameters']
pair=dict(candidate='masked',control='all_patch',delta_metrics={k:a['best_metrics'][k]-b['best_metrics'][k] for k in a['best_metrics']},train_cli_seconds_added=a['full_train_cli_seconds']-b['full_train_cli_seconds'],train_cli_ratio=a['full_train_cli_seconds']/b['full_train_cli_seconds'],training_loop_ratio=a['training_loop_seconds']/b['training_loop_seconds'],loop_epoch_evaluation_save_ratio=a['loop_with_epoch_evaluation_save_seconds']/b['loop_with_epoch_evaluation_save_seconds'])
pair['advance_gate']=pair['delta_metrics']['mAP']>=.5 and pair['delta_metrics']['Rank-1']>=0
packet.mkdir()
with (packet/'EPOCHS.csv').open('w',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(epochs[0]));w.writeheader();w.writerows(epochs)
flat=[dict(arm=r['arm'],feature_width=r['feature_width'],trainable_parameters=r['trainable_parameters'],best_epoch=r['best_epoch'],**r['best_metrics'],last_mAP=r['last_metrics']['mAP'],best_to_last_mAP_drop=r['best_to_last_map_drop'],formal_steps=r['formal_steps'],training_loop_seconds=r['training_loop_seconds'],loop_epoch_eval_save_seconds=r['loop_with_epoch_evaluation_save_seconds'],train_cli_seconds=r['full_train_cli_seconds'],first_strict_cli_seconds=r['first_strict_cli_seconds']) for r in rows]
with (packet/'RESULTS.csv').open('w',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(10.2,4.0),layout='constrained')
colors={'global_only':'#176B61','masked':'#B96116','all_patch':'#3567A8'}
labels={'global_only':'Global only (1536D)','masked':'SIM masked (3072D)','all_patch':'SIM all patch (3072D)'}
for ax,metric in zip(axes,('mAP','Rank-1')):
 for r in rows:
  selected=[h for h in epochs if h['arm']==r['arm']]
  ax.plot([h['epoch'] for h in selected],[h[metric] for h in selected],label=labels[r['arm']],color=colors[r['arm']],linewidth=1.7)
  ax.scatter(r['best_epoch'],r['best_metrics'][metric],color=colors[r['arm']],s=38,zorder=3,edgecolors='white',linewidths=.6)
 ax.set(xlabel='Epoch',ylabel=metric+' (%)',xlim=(1,50));ax.grid(axis='y',alpha=.18)
axes[0].legend(loc='lower right',fontsize=8.4,frameon=False)
fig.suptitle("MSVR310: completed 50-epoch source references, seed 42\nDots follow each arm's mAP-best checkpoint",fontsize=11)
for extension in ('png','pdf','svg'):fig.savefig(packet/f'TRAJECTORIES.{extension}',dpi=220)
plt.close(fig)
facts=dict(status='CLOSED_MSVR310_TRAJECTORIES_AND_COSTS',created_at=datetime.now().astimezone().isoformat(),rows=rows,primary_pair=pair,source_sha256=source_sha,boundary='Threeclosed50epoch/706update arms,150epochrows. Originalfirststrict andsame mAP-best CMC only. No NN/SSH/scoring/report replay. Finalregisteredall9queryCPUreport pending. SIMpair matchedhead/capacity/3072D; globalcomparison mixeshead/capacity/width/objectives. Time single sequentialsharedserverrun, not isolatedrepeatedbenchmark; excludesinitial/M0/oldfailcost. No newseed, significance/equivalence/stability/SOTA orthree-datasetcompletionclaim.')
(packet/'FACTS.json').write_text(json.dumps(facts,indent=2)+'\n',encoding='utf-8')
note=['# MSVR310：完整三端曲线与成本','', '| 条件 | 最佳轮次 | mAP | R1 | R5 | R10 | 末轮mAP | best−末轮 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
note += [f'| {r["arm"]} | {r["best_epoch"]} | {r["best_metrics"]["mAP"]:.4f} | {r["best_metrics"]["Rank-1"]:.4f} | {r["best_metrics"]["Rank-5"]:.4f} | {r["best_metrics"]["Rank-10"]:.4f} | {r["last_metrics"]["mAP"]:.4f} | {r["best_to_last_map_drop"]:.4f} |' for r in rows]
note += ['',f'匹配的masked−all-patch为{pair["delta_metrics"]["mAP"]:+.4f} mAP、{pair["delta_metrics"]["Rank-1"]:+.4f} R1，本轮未达到原推进条件。两条SIM路径相对plain-global的改善同时包含交互层、分类头、维度和目标变化，不能计成选择机制的独立增量。', '',f'masked/all-patch训练CLI时间比为{pair["train_cli_ratio"]:.4f}，差{pair["train_cli_seconds_added"]:.2f}秒。该次顺序运行中masked耗时更长；这不是重复隔离测速或一般硬件效率结论。表中时间分别记录loss-loop、loop与逐轮评价保存、完整训练CLI、首次严格重载。初始化、M0及历史失败成本另列。', '', '三条完整轨迹共150轮，来自三个模型各一次训练，不是150个独立种子。两个SIM的mAP-best位于第47、48轮，best到末轮变化不足0.1 mAP；当前记录不呈现此前RGBNT100 joint-L2那种大幅后期退化，不能用旧退化概括这两个新端。', '', '保留作者mask实现：未选token置零但不移除，模态内与模态间mask取并集，top-k不是最终保留总数。当前未记录真实mask覆盖率，不能宣称MSVR实际全选或据此定位负结果的唯一原因。', '',facts['boundary']]
(packet/'NOTE.md').write_text('\n'.join(note)+'\n',encoding='utf-8')
(packet/'SOURCE.py').write_bytes(Path(__file__).read_bytes())
outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in packet.iterdir() if p.is_file()}
(packet/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',outputs_sha256=outputs,epochs=150,arms=3,local_models_or_arrays=0,public_hot_sync=0),indent=2)+'\n')
print(json.dumps(dict(status=facts['status'],pair=pair,rows=flat,output=str(packet)),indent=2))
