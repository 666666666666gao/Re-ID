"""Read scientific bindings and source labels; no imports of model execution code."""
from pathlib import Path
import hashlib
import json
import re

repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def read(name): return json.loads((repo/name).read_bytes())
spec=read('configs/MSVR310/TriFusion-smooth-ap-objective-gradients-v1.json')
base=read('configs/MSVR310/Signal-source-oof-v1.json')
style=read('configs/MSVR310/TriFusion-source-style-paired-v1.json')
protocol=read(spec['protocol'])
pending=[repo/'configs/MSVR310/TriFusion-smooth-ap-objective-gradients-v1.json']
bindings=[];files={};processed=set()
while pending:
    path=pending.pop()
    if path in processed: continue
    processed.add(path)
    obj=json.loads(path.read_bytes())
    for key in ('project_file_sha256','project_source_file_sha256','fixed_file_sha256','project_files','fixed_files'):
        for name,digest in obj.get(key,{}).items():
            p=Path(name) if name.startswith('/') else repo/name
            actual=sha(p)
            bindings.append(dict(contract=str(path),field=key,path=str(p),expected=digest,actual=actual,passed=actual==digest))
            assert actual==digest,(str(path),str(p),actual,digest)
            files[str(p)]=dict(path=str(p),bytes=p.stat().st_size,sha256=actual)
            if p.suffix=='.json' and p.is_relative_to(repo/'configs'):pending.append(p)
    # Base configuration bindings are explicit non-map fields.
    for key,hkey in [('base_config','base_config_sha256'),('protocol','protocol_sha256'),('q1_summary','q1_summary_sha256')]:
        if hkey in obj:
            p=Path(obj[key]) if obj[key].startswith('/') else repo/obj[key]
            actual=sha(p); assert actual==obj[hkey]
            bindings.append(dict(contract=str(path),field=hkey,path=str(p),expected=obj[hkey],actual=actual,passed=True))
            files[str(p)]=dict(path=str(p),bytes=p.stat().st_size,sha256=actual)
            if p.is_relative_to(repo/'configs'):pending.append(p)
    if 'BASELINE' in obj:
        for key in ('CONFIG','SUMMARY'):
            p=Path(obj['BASELINE'][key]) if obj['BASELINE'][key].startswith('/') else repo/obj['BASELINE'][key]
            actual=sha(p);assert actual==obj['BASELINE'][key+'_SHA256']
            bindings.append(dict(contract=str(path),field='BASELINE.'+key,path=str(p),expected=obj['BASELINE'][key+'_SHA256'],actual=actual,passed=True))
            files[str(p)]=dict(path=str(p),bytes=p.stat().st_size,sha256=actual)
            if p.is_relative_to(repo/'configs'):pending.append(p)
    if 'SOURCE_METADATA' in obj:
        p=repo/obj['SOURCE_METADATA']['PATH'];actual=sha(p);assert actual==obj['SOURCE_METADATA']['SHA256']
        files[str(p)]=dict(path=str(p),bytes=p.stat().st_size,sha256=actual)
for name,digest in base['signal_source_file_sha256'].items():
    p=Path(base['signal_source'])/name;actual=sha(p);assert actual==digest
    bindings.append(dict(contract='Signal-source-oof-v1.json',field='signal_source_file_sha256',path=str(p),expected=digest,actual=actual,passed=True))
    files[str(p)]=dict(path=str(p),bytes=p.stat().st_size,sha256=actual)
for p in processed:
    files[str(p)]=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
for name in ('tools/build_msvr310_train_oof_protocol.py','tools/audit_vehicle_query_protocol_labels.py','evidence/msvr310_dataset_install_20260905.json','evidence/vehicle_query_protocol_labels_20260905.json','tools/probe_msvr_history_candidate_gradients.py','tools/msvr_freshness_probe.py','tools/msvr_instance_memory.py','tools/msvr_role_set_relations.py'):
    p=repo/name
    files[str(p)]=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
dataset=Path(base['dataset_root'])
labels=[]
for p in sorted((dataset/'bounding_box_train').glob('*/vis/*.jpg')):
    m=re.fullmatch(r'(\d+)_s(\d+)_v(\d+)_(\d+)\.jpg',p.name);assert m,p
    pid,scene,cam,num=map(int,m.groups());assert int(p.parent.parent.name)==pid
    paths=[p.parent.parent/mod/p.name for mod in ('vis','ni','th')]
    assert all(q.is_file() for q in paths)
    labels.append(dict(index=len(labels),identity=pid,camera=cam,scene=scene,paths=[q.relative_to(dataset).as_posix() for q in paths]))
assert labels==protocol['records']
ids=sorted({r['identity'] for r in labels})
multi=[i for i in ids if len({r['scene'] for r in labels if r['identity']==i})>=2]
single=[i for i in ids if i not in multi]
folds=[]
for number,fold in enumerate(protocol['folds']):
    held=sorted(multi[number::3]+single[number::3]);source=sorted(set(ids)-set(held))
    assert fold['heldout_ids']==held and fold['source_ids']==source
    si=[r['index'] for r in labels if r['identity'] in source]
    gi=[r['index'] for r in labels if r['identity'] in held]
    assert fold['source_record_indices']==si and fold['gallery_record_indices']==gi
    assert fold['source_label_map']=={str(p):j for j,p in enumerate(source)}
    assert not set(si)&set(gi) and set(si)|set(gi)==set(range(len(labels)))
    folds.append(dict(fold=number,source_ids=len(source),heldout_ids=len(held),source_records=len(si),heldout_records=len(gi)))
print(json.dumps(dict(status='PASS_STATIC_BINDINGS_AND_DATASET_LABELS',bindings=bindings,files=list(files.values()),
    protocol_sha256=sha(repo/spec['protocol']),dataset_root=str(dataset),dataset_records=len(labels),
    dataset_identities=len(ids),multi_scene_identities=len(multi),single_scene_identities=len(single),
    folds=folds,dataset_labels=labels,official_query_gallery_reads=0,model_forwards=0,optimizer_updates=0)))
