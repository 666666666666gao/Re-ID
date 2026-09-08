import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import pathlib,json,hashlib,subprocess,datetime
R=pathlib.Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
RUN=pathlib.Path('/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739')
def h(p):
    d=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):d.update(b)
    return d.hexdigest()
def j(p):return json.loads(pathlib.Path(p).read_text())
names=['TriFusion-role-set-paired-v1.json','TriFusion-history-gradient-paired-v1.json','TriFusion-fresh-coordinate-paired-v1.json','TriFusion-instance-memory-paired-v1.json','TriFusion-source-style-paired-v1-r2.json','Signal-source-oof-v1.json']
configs={str(R/'configs/MSVR310'/n):j(R/'configs/MSVR310'/n) for n in names}
style=configs[str(R/'configs/MSVR310/TriFusion-source-style-paired-v1-r2.json')]
base=configs[str(R/'configs/MSVR310/Signal-source-oof-v1.json')]
signal=pathlib.Path(base['signal_source'])
pins=[]
for cp,c in configs.items():
    for key in ['project_file_sha256','project_source_file_sha256','fixed_file_sha256','signal_source_file_sha256']:
        for name,expected in c.get(key,{}).items():
            p=pathlib.Path(name) if key=='fixed_file_sha256' else (signal if key=='signal_source_file_sha256' else R)/name
            actual=h(p);pins.append(dict(config=cp,field=key,path=str(p),bytes=p.stat().st_size,expected=expected,sha256=actual,match=expected==actual))
protocol=R/base['protocol'];meta=R/style['SOURCE_METADATA']['PATH'];baseline=pathlib.Path(style['BASELINE']['SUMMARY'])
extra={str(protocol):base['protocol_sha256'],str(meta):style['SOURCE_METADATA']['SHA256'],str(baseline):style['BASELINE']['SUMMARY_SHA256'],base['clip_weight']:base['clip_weight_sha256']}
b=j(baseline)
for f in b['folds']:
    extra[f['checkpoint']]=f['checkpoint_sha256'];extra[str(pathlib.Path(f['checkpoint']).parent/'retrieval_arrays.pt')]=f['retrieval']['retrieval_arrays_sha256']
for path,expected in extra.items():
    p=pathlib.Path(path);actual=h(p);pins.append(dict(field='runtime_input',path=str(p),bytes=p.stat().st_size,expected=expected,sha256=actual,match=expected==actual))
texts={str(p):p.read_text() for p in [protocol,meta,*[pathlib.Path(p) for p in configs]]}
for name in ['data/datasets/msvr310.py','data/datasets/sampler.py','data/datasets/bases.py','data/datasets/make_dataloader.py','utils/metrics.py','configs/MSVR310/Signal.yml']:
    p=signal/name;texts[str(p)]=p.read_text()
commits=[j(RUN/'pipeline.json')['code_commit'],j(RUN/'q1/summary.json')['project_commit'],subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()]
version=[]
source_names=sorted({pathlib.Path(p['path']).relative_to(R).as_posix() for p in pins if pathlib.Path(p['path']).is_relative_to(R) and pathlib.Path(p['path']).suffix in ('.py','.json','.yml')})
for commit in dict.fromkeys(commits):
    rows=[]
    for name in source_names:
        proc=subprocess.run(['git','-C',str(R),'show',commit+':'+name],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        actual=hashlib.sha256(proc.stdout).hexdigest() if proc.returncode==0 else None
        rows.append(dict(path=name,returncode=proc.returncode,commit_sha256=actual,current_sha256=h(R/name),equal=actual==h(R/name),stderr=proc.stderr.decode()))
    version.append(dict(commit=commit,files=rows))
baseline_scope=[]
pr=j(protocol)
for f,pf in zip(b['folds'],pr['folds'],strict=True):
    tr=f['training'];steps=tr['steps']
    sampled=[i for row in steps for i in row['sampled_record_indices']]
    assert set(sampled)<=set(pf['source_record_indices'])
    baseline_scope.append(dict(fold=f['fold'],checkpoint=f['checkpoint'],source_ids=pf['source_ids'],heldout_ids=pf['heldout_ids'],training_keys=list(tr),step_count=len(steps),all_sampled_source_only=True,sampled_unique_records=len(set(sampled)),training_initial_state=tr.get('initial_state_sha256'),training_final_state=tr['final_state_sha256']))
result=dict(status='PRIMARY_TEXT_HASH_INTAKE',time=datetime.datetime.now().astimezone().isoformat(),host=subprocess.check_output(['hostname'],text=True).strip(),user=subprocess.check_output(['whoami'],text=True).strip(),cwd=str(pathlib.Path.cwd()),pins=pins,texts=texts,git_source_versions=version,baseline_scope=baseline_scope,baseline_summary_top={k:v for k,v in b.items() if k not in ('folds','comparison','aggregate')},run_files=[dict(path=str(p),bytes=p.stat().st_size) for p in RUN.rglob('*') if p.is_file()],model_forwards=0,optimizer_updates=0,all_writes='stdout only')
print(json.dumps(result,ensure_ascii=False,allow_nan=False))
