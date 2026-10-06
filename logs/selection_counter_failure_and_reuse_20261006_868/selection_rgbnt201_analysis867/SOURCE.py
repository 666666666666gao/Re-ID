"""Closed RGBNT201 trajectory/cost analysis; no model, query scoring or SSH."""
from pathlib import Path
from datetime import datetime
import csv,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
intake=base/'selection_rgbnt201867'
packet=base/'selection_rgbnt201_analysis867'
assert not packet.exists()
remote_path=intake/'REMOTE.json'
remote=json.loads(remote_path.read_bytes())
assert json.loads((intake/'COMPLETE.json').read_bytes())['status']=='COMPLETE'
observation_path=base/'signal_selection_msvr_start_followup867/OBSERVATION.json'
observation=json.loads(observation_path.read_bytes())
source_sha={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (remote_path,observation_path)}
rows=[];epochs=[]
for row in remote['rows']:
    selection=row['variant']
    prefix=row['run_dir'].removeprefix('/data/gaob/Re-ID/Trifusion/')
    local=intake/'received'/prefix
    for name in ('training.json','official_metrics.json'):
        p=local/name;info=remote['text_files'][prefix+'/'+name]
        assert p.stat().st_size==info['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256']
        source_sha[str(p)]=info['sha256']
    training=json.loads((local/'training.json').read_bytes())
    receipt=json.loads((local/'official_metrics.json').read_bytes())
    history=training['history']
    assert [r['epoch'] for r in history]==list(range(1,51))
    best=max(history,key=lambda r:(r['official_fused']['mAP'],r['epoch']))
    assert best['epoch']==row['best_epoch']==receipt['selected_epoch']
    assert all(abs(best['official_fused'][k]-v)<1e-5 for k,v in row['metrics'].items())
    job=next(j for j in observation['campaign']['jobs'] if (j['dataset'],j['selection'],j['phase'])==('RGBNT201',selection,'full'))
    assert job['status']=='COMPLETE' and job['result']==row
    times={s['mode']:(datetime.fromisoformat(s['completed_at'])-datetime.fromisoformat(s['started_at'])).total_seconds() for s in job['steps']}
    rows.append(dict(arm=selection,feature_width=row['initializer']['feature_width'],
        trainable_parameters=row['initializer']['trainable_parameters'],best_epoch=row['best_epoch'],
        formal_steps=row['formal_steps'],best_metrics=row['metrics'],last_metrics=history[-1]['official_fused'],
        best_to_last_map_drop=row['metrics']['mAP']-history[-1]['official_fused']['mAP'],
        training_loop_seconds=sum(r['seconds'] for r in history),
        loop_with_epoch_evaluation_save_seconds=(datetime.fromisoformat(training['completed_at'])-datetime.fromisoformat(training['started_at'])).total_seconds(),
        full_train_cli_seconds=times['train'],first_strict_cli_seconds=times['evaluate'],
        checkpoint_sha256=row['checkpoint_sha256'],receipt_sha256=row['receipt_sha256']))
    for h in history:
        epochs.append(dict(arm=selection,epoch=h['epoch'],steps=h['steps'],mean_loss=h['mean_loss'],
            training_loop_seconds=h['seconds'],**h['official_fused']))
assert len(epochs)==150
packet.mkdir()
with (packet/'EPOCHS.csv').open('w',encoding='utf-8',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(epochs[0]));writer.writeheader();writer.writerows(epochs)
with (packet/'RESULTS.csv').open('w',encoding='utf-8',newline='') as stream:
    flat=[dict(arm=r['arm'],feature_width=r['feature_width'],trainable_parameters=r['trainable_parameters'],
        best_epoch=r['best_epoch'],**r['best_metrics'],last_mAP=r['last_metrics']['mAP'],
        best_to_last_mAP_drop=r['best_to_last_map_drop'],formal_steps=r['formal_steps'],
        training_loop_seconds=r['training_loop_seconds'],loop_epoch_eval_save_seconds=r['loop_with_epoch_evaluation_save_seconds'],
        train_cli_seconds=r['full_train_cli_seconds'],first_strict_cli_seconds=r['first_strict_cli_seconds']) for r in rows]
    writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(10.2,4.0),layout='constrained')
colors={'global_only':'#176B61','masked':'#B96116','all_patch':'#3567A8'}
labels={'global_only':'Global only (1536D)','masked':'SIM masked (3072D)','all_patch':'SIM all patch (3072D)'}
for ax,metric in zip(axes,('mAP','Rank-1')):
    for r in rows:
        selected=[e for e in epochs if e['arm']==r['arm']]
        x=[e['epoch'] for e in selected];y=[e[metric] for e in selected]
        ax.plot(x,y,label=labels[r['arm']],color=colors[r['arm']],linewidth=1.7)
        ax.scatter(r['best_epoch'],r['best_metrics'][metric],color=colors[r['arm']],s=38,zorder=3,edgecolors='white',linewidths=.6)
    ax.set(xlabel='Epoch',ylabel=metric+' (%)',xlim=(1,50))
    ax.grid(axis='y',alpha=.18)
axes[0].legend(loc='lower right',fontsize=8.4,frameon=False)
fig.suptitle('RGBNT201: completed 50-epoch source references, seed 42\nDots follow each arm\'s mAP-best checkpoint',fontsize=11)
fig.savefig(packet/'TRAJECTORIES.png',dpi=220)
fig.savefig(packet/'TRAJECTORIES.pdf')
fig.savefig(packet/'TRAJECTORIES.svg')
plt.close(fig)
data=dict(status='CLOSED_RGBNT201_TRAJECTORIES_AND_COSTS',created_at=datetime.now().astimezone().isoformat(),rows=rows,
    pairs=remote['pairs'],source_sha256=source_sha,
    boundary='All3closed50-epoch records,150epochs. No NN/SSH/scoring/report rerun; not newseed/stability/significance/SOTA. mAP-bestCMC unchanged. Mainmask-allpatch matched3072; globalcomparison mixescapacity/head/dimension/objectives. Loss totals have different headcounts. Timefields distinguishloop,loop+epoch-eval/save,trainCLI andfirststrict; no M0/prepare/oldfailedwork orinferencebenchmark included.')
(packet/'FACTS.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
lines=['# RGBNT201：已闭合三端轨迹与成本','',
    '| Arm | Best epoch | mAP | R1 | R5 | R10 | Last mAP | Best–last |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
lines += [f'| {r["arm"]} | {r["best_epoch"]} | {r["best_metrics"]["mAP"]:.4f} | {r["best_metrics"]["Rank-1"]:.4f} | {r["best_metrics"]["Rank-5"]:.4f} | {r["best_metrics"]["Rank-10"]:.4f} | {r["last_metrics"]["mAP"]:.4f} | {r["best_to_last_map_drop"]:.4f} |' for r in rows]
lines += ['',
    '1. 主要同容量配对masked−all_patch为+0.2370mAP、−1.1962R1，未达原推进门。两SIM臂mAP均低于无SIM global；后者还混有头/维度/容量/训练目标差，不是选择专属效应。',
    '2. best到末轮分别回落1.6558、0.8598、2.6447mAP；记录提示后期回落，尚未定位唯一原因。图中两类指标的点都沿用mAP-best，未另选R1峰值。',
    '3. 所有曲线是同一训练内的50轮，不是50个种子。无置信区间或跨数据集成功主张。时间分项来自实际日志，未混算首次构造、M0、旧失败和推理成本。',
    '', '后续：继续既定MSVR310/RGBNT100六端与唯一九端全query报告；不据本三端调整topk、LR、seed或推进门。',
    '', data['boundary']]
(packet/'NOTE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(packet/'SOURCE.py').write_bytes(Path(__file__).read_bytes())
outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in packet.iterdir() if p.is_file()}
(packet/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',outputs_sha256=outputs,epochs=150,arms=3,local_models_or_arrays=0,public_hot_sync=0),indent=2)+'\n')
print(json.dumps(dict(status=data['status'],rows=[dict(arm=r['arm'],best_to_last_map_drop=r['best_to_last_map_drop'],train_cli_seconds=r['full_train_cli_seconds']) for r in rows],output=str(packet)),indent=2))
