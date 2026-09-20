from pathlib import Path
import json,hashlib,ast,datetime
out=Path(__file__).parent;repo=Path(r'C:\Users\gb\.trifusion_github_publish_22c3bee')
protocol=json.loads((out/'snapshots/repo/protocols/msvr310_train_oof_v1.json').read_bytes())
added=[]
for name in ('tools/build_msvr310_train_oof_protocol.py','tools/audit_vehicle_query_protocol_labels.py','evidence/vehicle_query_protocol_labels_20260905.json','evidence/msvr310_dataset_install_20260905.json'):
    data=(repo/name).read_bytes();p=out/'snapshots/ground_truth'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    added.append(dict(path=name,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),lines=len(data.splitlines())))
assert added[0]['sha256']==protocol['builder_sha256']
label=(repo/'evidence/vehicle_query_protocol_labels_20260905.json').read_bytes()
assert hashlib.sha256(label).hexdigest()==protocol['label_evidence_sha256']
dataset=next(d for d in json.loads(label)['datasets'] if d['dataset']=='MSVR310')
source=dataset['record_manifest']['bounding_box_train']
assert len(source)==len(protocol['records'])==1032
for i,(a,b) in enumerate(zip(source,protocol['records'])):
    assert b['index']==i
    assert (a['identity'],a['camera'],a['scene'])==(b['identity'],b['camera'],b['scene'])
    assert 'bounding_box_train/'+a['path']==b['paths'][0]
ids=sorted({r['identity'] for r in protocol['records']})
scenes={i:{r['scene'] for r in protocol['records'] if r['identity']==i} for i in ids}
partition=[set() for _ in range(3)]
for group in ([i for i in ids if len(scenes[i])>1],[i for i in ids if len(scenes[i])==1]):
    for n,i in enumerate(group):partition[n%3].add(i)
for f,held in zip(protocol['folds'],partition):
    assert f['heldout_ids']==sorted(held)
    assert f['source_ids']==sorted(set(ids)-held)
    assert f['source_label_map']=={str(i):n for n,i in enumerate(f['source_ids'])}
newline=[]
for p in sorted((out/'remote_artifacts/source_dependencies/modeling').rglob('*.py')):
    rel=p.relative_to(out/'remote_artifacts/source_dependencies').as_posix()
    remote=p.read_bytes();local=(out/'snapshots/dependencies'/rel).read_bytes()
    assert local.replace(b'\r\n',b'\n')==remote,rel
    assert ast.dump(ast.parse(local))==ast.dump(ast.parse(remote)),rel
    newline.append(dict(path=rel,only_crlf_vs_lf=True,ast_equal=True,local_sha256=hashlib.sha256(local).hexdigest(),remote_sha256=hashlib.sha256(remote).hexdigest()))
assert len(newline)==5
result=dict(status='PASS_DATASET_LABEL_AND_SOURCE_BINDINGS',generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_records=1032,source_identities=155,cross_scene_identities=60,single_scene_identities=95,all_source_labels_match_archived_dataset_manifest=True,source_partition_label_only_round_robin_matches=True,label_sha256=protocol['label_evidence_sha256'],builder_sha256=protocol['builder_sha256'],added_input_inventory=added,newline_only_files=newline,image_reads=0,model_forwards=0)
(out/'provenance_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('added_input_inventory','newline_only_files')},indent=2))
with (out/'initial_environment_errors.md').open('a',encoding='utf-8') as f:
    f.write('\n5. Two initial reads expected trifusion/aligned_data.py at the repository root; the actual module is modeling/trifusion/aligned_data.py, resolved from the builder sys.path. No source change.\n6. One rg command supplied wildcard strings as literal Windows paths and returned os error 123; repeated as rg with directory and -g scope, locating build_msvr310_train_oof_protocol.py.\n7. R1 independent checker exited 1 on NumPy int64 summary serialization before arrays; full traceback, scripts and terminal receipt preserved in remote_artifacts/failed_attempt01. R2 only casts three summary counts to int, with exact change receipt.\n')
