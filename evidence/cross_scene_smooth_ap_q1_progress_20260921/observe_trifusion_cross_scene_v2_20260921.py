from pathlib import Path
import json,datetime,subprocess,shutil
root=Path('/root/trifusion-storage/artifacts/msvr310_cross_scene_smooth_ap_v1_seed42_d35864d')
result=dict(observed_at=datetime.datetime.now().astimezone().isoformat(),root=str(root),
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True),
            free_bytes=shutil.disk_usage(root.parent).free)
pipeline=root/'pipeline.json'
if pipeline.exists():
    data=json.loads(pipeline.read_bytes());result['pipeline']=data
    pids=[data['wrapper_pid']]+[s['original_pid'] for s in data['stages']]
    result['original_processes']={str(pid):(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode() if (Path('/proc')/str(pid)/'cmdline').exists() else None for pid in pids}
for stage in ('t0','m0_cpu','q1_cpu'):
    file=root/(stage+'.json')
    if file.exists():
        obj=json.loads(file.read_bytes());result[stage]={k:v for k,v in obj.items() if k in ('status','checked_training_steps','full_source_batches','elapsed_seconds','summary_sha256')}
for mode in ('m0','q1'):
    file=root/mode/'summary.json'
    if file.exists():
        obj=json.loads(file.read_bytes());ends=[]
        for fold in obj['folds']:
            for name,row in fold['endpoints'].items():
                tr=row['training']
                ends.append(dict(fold=fold['fold'],endpoint=name,updates=tr['optimizer_steps'],epochs=tr['epochs'],fit_epoch_seconds=sum(r['elapsed_seconds'] for r in tr['history']),overflow=tr['overflow_events'],nonzero=tr['nonzero_gradient_tensors'],checks=row['engineering_checks'],endpoint_receipt_present=(root/mode/f"fold_{fold['fold']}_{name}"/'receipt.json').exists(),checkpoint_recorded='checkpoint_sha256' in row,retrieval_recorded='retrieval' in row))
        result[mode]=dict(status=obj['status'],endpoints=ends,overfit={k:dict(checks=v['checks'],gate=v['gate']) for k,v in obj['overfit'].items()})
    log=root/(mode+'.log')
    if log.exists():result[mode+'_log_tail']=log.read_text()[-5000:]
for name in ('t0.log','m0_cpu.log','q1_cpu.log'):
    file=root/name
    if file.exists():result[name]=file.read_text()[-1800:]
outer=root.with_name(root.name+'_wrapper.log')
if outer.exists():result['wrapper_log_tail']=outer.read_text()[-1800:]
print(json.dumps(result))
