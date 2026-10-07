from pathlib import Path
import hashlib,json
root=Path('/data/gaob/Re-ID/Trifusion')
journal=root/'logs/qualified_five_incremental_full_launch_20261007_888'
campaign=root/'logs/qualified_five_incremental_full_v1_20261007_888'
assert json.loads((journal/'EXIT.json').read_text())['exit_code']==0
assert json.loads((campaign/'campaign.json').read_text())['status']=='COMPLETE'
for p in Path('/proc').iterdir():
    if p.name.isdigit() and (p/'cmdline').is_file():
        words=(p/'cmdline').read_bytes().split(bytes([0]))
        assert not any(w.startswith(b'/data/gaob/Re-ID/Trifusion/tools/') for w in words),words
scope_path=root/'refine-logs/incremental_role_objective_v1/QUALIFIED_FIVE_SOURCE_SCOPE.json'
scope=json.loads(scope_path.read_text())['source_sha256']
assert len(scope)==420
files={}
for name,digest in scope.items():
    p=root/name
    data=p.read_bytes()
    assert hashlib.sha256(data).hexdigest()==digest,name
    files[name]={'text':data.decode('utf-8'),'sha256':digest,'bytes':len(data)}
extras=['refine-logs/incremental_role_objective_v1/QUALIFIED_FIVE_SOURCE_SCOPE.json',
        'refine-logs/incremental_role_objective_v1/INPUT_SEAL.json',
        'refine-logs/incremental_role_objective_v1/INITIAL_BOUNDARY_INPUT_SEAL.json']
extras += ['logs/training_feature_scale_protocols_20261002/'+d+'.json' for d in ('RGBNT201','RGBNT100','MSVR310')]
for name in extras:
    data=(root/name).read_bytes()
    files[name]={'text':data.decode('utf-8'),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
print(json.dumps({'status':'EXACT_FROZEN_420_SOURCES_AND_SIX_PROTOCOL_SEAL_TEXTS','source_count':420,'files':files}))
