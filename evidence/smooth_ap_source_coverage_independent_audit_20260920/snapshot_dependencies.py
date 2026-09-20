from pathlib import Path
import json,hashlib,ast
out=Path(__file__).parent
repo=Path(r'C:\Users\gb\.trifusion_github_publish_22c3bee')
pending=['configs/MSVR310/TriFusion-smooth-ap-source-coverage-v1.json']
configs={};pins=[];extra=set()
while pending:
    name=pending.pop(0)
    if name in configs:continue
    data=json.loads((repo/name).read_bytes());configs[name]=data
    for field in ('training_config','previous_config','coordinate_config','memory_config','base_config'):
        if field in data:pending.append(data[field])
    if 'BASELINE' in data:pending.append(data['BASELINE']['CONFIG'])
    for key in ('project_file_sha256','project_source_file_sha256'):
        for path,digest in data.get(key,{}).items():pins.append(dict(config=name,path=path,expected_sha256=digest))
    for key in ('protocol',):
        if key in data:extra.add(data[key])
extra.update('''tools/train_msvr_role_set.py
tools/train_msvr_history_gradient.py
tools/train_msvr_fresh_coordinate.py
tools/train_msvr_instance_memory.py
tools/train_msvr310_source_style.py
tools/train_msvr310_signal_oof.py
tools/train_msvr310_trifusion_oof.py
tools/build_v12_complete_path_oof_targets.py
tools/run_signal_baseline_dev.py
tools/run_signal_preserving_v5.py
modeling/trifusion/aligned_data.py
modeling/trifusion/signal_preserving_v8.py
modeling/trifusion/signal_preserving_v8_builder.py
modeling/trifusion/signal_preserving_v7.py
modeling/trifusion/signal_preserving_v6.py
modeling/trifusion/signal_preserving_v5.py
modeling/trifusion/experts/mamba.py
modeling/trifusion/state.py
modeling/trifusion/builder.py'''.splitlines())
files=sorted(set(configs)|{p['path'] for p in pins}|extra)
inventory=[]
for name in files:
    p=repo/name;data=p.read_bytes();target=out/'snapshots/dependencies'/name
    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    inventory.append(dict(path=name,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),lines=len(data.splitlines())))
(out/'dependency_inventory.json').write_text(json.dumps(dict(config_chain=list(configs),files=inventory,pins=pins),indent=2),encoding='utf-8')
selected={
 'tools/train_msvr_role_set.py':['context'],
 'tools/train_msvr_history_gradient.py':['context'],
 'tools/train_msvr_fresh_coordinate.py':['context'],
 'tools/train_msvr_instance_memory.py':['context','reload_model'],
 'tools/train_msvr310_source_style.py':['context','pixels'],
 'tools/train_msvr310_signal_oof.py':['configure','records_for','clean_triplet','loader_for','new_model','scene_scores'],
 'tools/train_msvr310_trifusion_oof.py':['build_model','output_mapping'],
 'tools/run_signal_baseline_dev.py':['_configure_signal_source'],
 'tools/build_v12_complete_path_oof_targets.py':['_build_signal_teacher','_build_v8_experts'],
 'tools/run_signal_preserving_v5.py':['_set_seed','_module_state_sha256','_training_batch'],
}
parts=[]
for name,names in selected.items():
    text=(repo/name).read_text(encoding='utf-8');lines=text.splitlines()
    tree=ast.parse(text)
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:
            parts.append('\n'+name+'\n'+'\n'.join(f'{i+1}:{lines[i]}' for i in range(node.lineno-1,node.end_lineno)))
(out/'loader_context_source_extracts.txt').write_text('\n'.join(parts),encoding='utf-8')
payload='''from pathlib import Path
import json,hashlib,subprocess,datetime
repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
pins=PINS
files=FILES
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def emit(kind,**v):print(json.dumps(dict(kind=kind,**v)),flush=True)
digests={name:sha(repo/name) for name in files}
emit('source_hashes',files=[dict(path=name,sha256=digests[name],bytes=(repo/name).stat().st_size) for name in files])
checks=[dict(**p,actual_sha256=digests[p['path']],matches=digests[p['path']]==p['expected_sha256']) for p in pins]
emit('recursive_config_binding',checks=checks,all_match=all(p['matches'] for p in checks))
base=json.loads((repo/'configs/MSVR310/Signal-source-oof-v1.json').read_bytes());signal=Path(base['signal_source'])
checks=[dict(path=name,actual_sha256=sha(signal/name),expected_sha256=digest) for name,digest in base['signal_source_file_sha256'].items()]
emit('signal_source_binding',head=subprocess.check_output(['git','-C',str(signal),'rev-parse','HEAD'],text=True).strip(),expected_head=base['signal_commit'],diff_sha256=hashlib.sha256(subprocess.check_output(['git','-C',str(signal),'diff','--binary'])).hexdigest(),expected_diff_sha256=base['signal_diff_sha256'],checks=checks)
for name in ('data/datasets/make_dataloader.py','data/datasets/sampler.py','data/datasets/msvr310.py','utils/metrics.py','modeling/__init__.py','modeling/make_model.py','modeling/meta_arch.py'):
 p=signal/name;emit('signal_source_text',path=str(p),sha256=sha(p),text=p.read_text())
spec=json.loads((repo/'configs/MSVR310/TriFusion-smooth-ap-source-coverage-v1.json').read_bytes())
q=json.loads(Path(spec['q1_summary']).read_bytes())
for f in q['folds']:
 for endpoint in ('control','smooth_ap'):
  row=f['endpoints'][endpoint];p=Path(row['checkpoint']);actual=sha(p)
  emit('checkpoint_identity',fold=f['fold'],endpoint=endpoint,path=str(p),bytes=p.stat().st_size,sha256=actual,expected_sha256=row['checkpoint_sha256'],matches=actual==row['checkpoint_sha256'],final_state_sha256=row['training']['final_state_sha256'])
emit('complete',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),model_forwards=0,optimizer_updates=0,image_reads=0)
'''.replace('PINS',repr(pins)).replace('FILES',repr(files))
(out/'remote_source_binding.py').write_text(payload,encoding='utf-8')
print(json.dumps(dict(config_chain=list(configs),files=len(files),pins=len(pins))))
