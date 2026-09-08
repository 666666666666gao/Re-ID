"""Independent stdlib-only static preexecution audit; never imports project code."""
import ast
from collections import Counter, OrderedDict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

R = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
A = Path(__file__).resolve().parent
inputs = {}

def data(name):
    path = R / name
    value = path.read_bytes()
    inputs[name] = {'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}
    return value

def js(name):
    return json.loads(data(name))

primary = [
    'refine-logs/msvr310_history_gradient_v1/EXPERIMENT_PLAN.md',
    'refine-logs/msvr310_history_gradient_v1/EXPERIMENT_TRACKER.md',
    'configs/MSVR310/TriFusion-history-gradient-paired-v1.json',
    'tools/train_msvr_history_gradient.py',
    'tools/verify_msvr_history_gradient.py',
    'tools/check_msvr_history_gradient.py',
    'tools/run_msvr_history_gradient.py',
]
read_sources = [
    'AGENTS.md', 'tools/probe_msvr_history_candidate_gradients.py',
    'tools/msvr_freshness_probe.py', 'tools/msvr_instance_memory.py',
    'tools/train_msvr_fresh_coordinate.py', 'tools/train_msvr_instance_memory.py',
    'tools/train_msvr310_source_style.py', 'tools/train_msvr310_trifusion_oof.py',
    'tools/train_msvr310_signal_oof.py', 'tools/verify_msvr310_source_style.py',
    'tools/run_signal_preserving_v5.py', 'tools/run_signal_baseline_dev.py',
    'tools/build_v12_complete_path_oof_targets.py', 'tools/train_signal_preserving_v17.py',
    'tools/msvr310_exact_signal_inference.py', 'tools/build_msvr310_train_oof_protocol.py',
    'modeling/trifusion/signal_preserving_v8.py',
    'modeling/trifusion/signal_preserving_v8_builder.py',
    'modeling/trifusion/criterion.py', 'modeling/trifusion/experts/mamba.py',
    'modeling/trifusion/experts/semantic_residual.py', 'modeling/trifusion/signal_preserving_v13.py',
    'modeling/trifusion/aligned_data.py', 'modeling/trifusion/state.py',
    'data/datasets/msvr310.py',
    'results/MSVR310_HISTORY_CANDIDATE_GRADIENT_SOURCE_2026-09-08.md',
    'refine-logs/msvr310_history_candidate_gradient_v1/EXPERIMENT_AUDIT_SOURCE.md',
]
config_names = [
    primary[2], 'configs/MSVR310/TriFusion-fresh-coordinate-paired-v1.json',
    'configs/MSVR310/TriFusion-instance-memory-paired-v1.json',
    'configs/MSVR310/TriFusion-source-style-paired-v1-r2.json',
    'configs/MSVR310/Signal-source-oof-v1.json',
]
for name in primary + read_sources:
    data(name)
snapshots = A / 'inputs'
snapshots.mkdir(exist_ok=True)
for name in primary:
    destination = snapshots / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    value = data(name)
    if destination.exists():
        assert destination.read_bytes() == value, ('snapshot input changed', name)
    else:
        destination.write_bytes(value)

syntax = {}
trees = {}
for name in primary + read_sources:
    if name.endswith('.py'):
        source = data(name).decode('utf-8-sig')
        node = ast.parse(source, filename=name, feature_version=(3, 10))
        compile(node, name, 'exec')
        trees[name] = node
        syntax[name] = {'python_3_10_grammar': 'PASS', 'compile': 'PASS', 'lines': len(source.splitlines())}

binding_checks = []
remote_binding_count = 0
for name in config_names:
    config = js(name)
    for key in ('project_file_sha256', 'project_source_file_sha256'):
        for relative, expected in config.get(key, {}).items():
            b = data(relative)
            digest = hashlib.sha256(b).hexdigest()
            row = {'config': name, 'path': relative, 'expected': expected, 'actual': digest}
            if digest == expected:
                row['status'] = 'EXACT_LOCAL_BYTES'
            elif hashlib.sha256(b.replace(b'\r\n', b'\n')).hexdigest() == expected:
                blob = subprocess.check_output(['git', '-C', str(R), 'show', 'HEAD:' + relative])
                row.update(status='INHERITED_CRLF_VS_BOUND_LF', git_head_blob_sha256=hashlib.sha256(blob).hexdigest())
                assert row['git_head_blob_sha256'] == expected, row
            else:
                row['status'] = 'MISMATCH'
            binding_checks.append(row)
    remote_binding_count += len(config.get('fixed_file_sha256', {}))

top, coordinate, memory_config, base, signal_config = [js(name) for name in config_names]
assert top['memory'] == coordinate['memory'] == memory_config['memory'] == {
    'capacity':512,'maximum_age':8,'warmup_steps':65,'capacity_warmup_steps':2,'overfit_warmup_steps':2}
assert top['seed'] == coordinate['seed'] == memory_config['seed'] == base['EXPERIMENT']['SEED'] == 42
assert top['training_epochs'] == coordinate['training_epochs'] == memory_config['training_epochs'] == base['OPTIMIZATION']['MAX_EPOCHS'] == 20
assert top['minimum_free_bytes'] == 3 * 1024**3
assert inputs[memory_config['base_config']]['sha256'] == memory_config['base_config_sha256']
assert inputs[base['BASELINE']['CONFIG']]['sha256'] == base['BASELINE']['CONFIG_SHA256']
assert top['official_test_access'] is False and top['heldout_access_only_after_m0'] is True
assert top['new_trainable_parameters'] == 0 and top['whole_goal_achieved'] is False
assert base['MODEL']['ARCHITECTURE'] == 'signal_preserving_collaborative_v8_expert_formation'
assert base['OPTIMIZATION']['NEW_MODULE_LR'] == .00035 and base['OPTIMIZATION']['WEIGHT_DECAY'] == .0001
assert base['OPTIMIZATION']['AMP_INIT_SCALE'] == 256 and base['OPTIMIZATION']['WARMUP_EPOCHS'] == 5
assert base['LOSS'] == {'ID_FUSED':.25,'TRIPLET_FUSED':1.,'ID_BRANCH':1/12,'TRIPLET_BRANCH':.25,'ID_RESIDUAL':1/12,'TRIPLET_RESIDUAL':.25,'TRIPLET_MARGIN':.3,'LABEL_SMOOTHING':.1}

protocol = js(signal_config['protocol'])
assert inputs[signal_config['protocol']]['sha256'] == signal_config['protocol_sha256'] == base['DATA']['PROTOCOL_SHA256']
labels = js('evidence/vehicle_query_protocol_labels_20260905.json')
assert inputs['evidence/vehicle_query_protocol_labels_20260905.json']['sha256'] == protocol['label_evidence_sha256']
rows = next(d for d in labels['datasets'] if d['dataset']=='MSVR310')['record_manifest']['bounding_box_train']
assert len(rows) == len(protocol['records']) == 1032
identities = sorted({r['identity'] for r in rows})
scenes = {i: sorted({r['scene'] for r in rows if r['identity']==i}) for i in identities}
assert len(identities) == 155
eligible = [i for i in identities if len(scenes[i])>1]
single = [i for i in identities if len(scenes[i])==1]
assert len(eligible)==60 and len(single)==95
for i, (raw, record) in enumerate(zip(rows, protocol['records'], strict=True)):
    p=Path(raw['path']); n=p.name
    assert int(n[:4])==raw['identity']==record['identity']
    assert int(n[6:9])==raw['scene']==record['scene']
    assert int(n[11])==raw['camera']==record['camera']
    assert record == {'index':i,'identity':raw['identity'],'camera':raw['camera'],'scene':raw['scene'],
                      'paths':[(Path('bounding_box_train')/p.parts[0]/m/n).as_posix() for m in ('vis','ni','th')]}
assert protocol['identity_scene_membership']=={str(i):scenes[i] for i in identities}

metadata = js(base['SOURCE_METADATA']['PATH'])
assert inputs[base['SOURCE_METADATA']['PATH']]['sha256'] == base['SOURCE_METADATA']['SHA256']
protocol_checks=[]
queue_checks=[]
all_gallery=[]
all_queries=[]
for f, md in zip(protocol['folds'], metadata['folds'], strict=True):
    fi=f['fold']; assert fi==md['fold']
    held=sorted(eligible[fi::3]+single[fi::3]); source=sorted(set(identities)-set(held))
    assert f['heldout_ids']==held and f['source_ids']==source and set(held).isdisjoint(source)
    assert f['source_label_map']=={str(i):j for j,i in enumerate(source)}
    sr=[r['index'] for r in protocol['records'] if r['identity'] in source]
    ga=[r['index'] for r in protocol['records'] if r['identity'] in held]
    assert sr==f['source_record_indices'] and ga==f['gallery_record_indices']
    gallery=[protocol['records'][i] for i in ga]
    qr=[]; excluded=[]
    for gp, record in enumerate(gallery):
        removed=sum(r['identity']==record['identity'] and r['scene']==record['scene'] for r in gallery)
        positive=sum(r['identity']==record['identity'] and r['scene']!=record['scene'] for r in gallery)
        if not positive:
            excluded.append(record['index']);continue
        qr.append(dict(record_index=record['index'], gallery_position=gp, identity=record['identity'],
                       scene=record['scene'], valid_positives=positive, removed_same_identity_same_scene=removed,
                       retained_gallery=len(gallery)-removed,
                       negative_identity_distractors=sum(r['identity']!=record['identity'] for r in gallery)))
    assert qr==f['query_rows'] and excluded==f['excluded_query_record_indices']
    all_gallery+=ga;all_queries += [q['record_index'] for q in qr]
    assert len(md['batches'])==260
    sampled=set()
    for b in md['batches']:
        inds=b['record_indices'];sampled.update(inds)
        assert len(inds)==64 and set(inds)<=set(sr)
        assert sorted(Counter(protocol['records'][i]['identity'] for i in inds).values())==[8]*8
    assert sampled==set(sr)
    protocol_checks.append(dict(fold=fi,source_records=len(sr),source_identities=len(source),
                                gallery_records=len(ga),queries=len(qr),query_identities=len({q['identity'] for q in qr}),
                                excluded_queries_retained=len(excluded),label_zero_identity=source[0],batches=260))
    for mode, warmup, batches in [('comparison',65,md['batches']),('capacity',2,md['batches'][:8]),('overfit',2,[md['batches'][0]]*100)]:
        cache=OrderedDict();groups_by_step={};counts=[];first=None;max_queue=0;dup=current_excluded=expired=evictions=0
        for step,b in enumerate(batches):
            inds=b['record_indices'];dup+=len(inds)-len(set(inds))
            for i in [i for i, v in cache.items() if step-v[0]>8]:
                expired+=1;del cache[i]
            available=[(i,v) for i,v in cache.items() if i not in inds]
            current_excluded+=len(cache)-len(available)
            assert len({i for i,_ in available})==len(available)
            groups={v[0] for i,v in available}
            counts.append(len(available))
            if available and first is None:
                first=dict(zero_based_step=step,recorded_step=step+1,history_groups=len(groups),history_records=len(available))
            for i,(stored, pos) in available:
                assert 1<=step-stored<=8
                stored_indices=groups_by_step[stored]
                assert pos==max(j for j,x in enumerate(stored_indices) if x==i)
            if step>=warmup:
                groups_by_step[step]=inds
                for pos,i in enumerate(inds):
                    cache.pop(i,None);cache[i]=(step,pos)
                while len(cache)>512:
                    evictions+=1;cache.popitem(last=False)
            max_queue=max(max_queue,len(cache))
        if mode=='comparison':assert sum(counts[:66])==0 and sum(c>0 for c in counts)==194
        if mode=='capacity':assert first['zero_based_step']==3 and first['history_groups']==1
        if mode=='overfit':assert not any(counts)
        queue_checks.append(dict(fold=fi,mode=mode,steps=len(batches),first_history=first,historical_candidate_exposures=sum(counts),
                                  history_batches=sum(c>0 for c in counts),max_history=max(counts),max_post_update_queue=max_queue,
                                  duplicate_batch_exposures=dup,current_record_history_exclusions=current_excluded,
                                  age_expirations=expired,capacity_evictions=evictions))
assert sorted(all_gallery)==list(range(1032)) and len(all_queries)==len(set(all_queries))==600

train_tree=trees['tools/train_msvr_history_gradient.py']
fit=next(n for n in train_tree.body if isinstance(n,ast.FunctionDef) and n.name=='fit')
calls=[(n.lineno,ast.unparse(n.func)) for n in ast.walk(fit) if isinstance(n,ast.Call)]
assert sum(name=='torch.optim.AdamW' for _,name in calls)==1
assert sum(name=='scaler.step' for _,name in calls)==1
assert sum(name=='scaler.unscale_' for _,name in calls)==1
assert sum(name=='scaler.update' for _,name in calls)==1
assert sum(name=='scaler.scale(loss).backward' for _,name in calls)==1
assert not any(name in ('optimizer.step','loss.backward') for _,name in calls)
step_line=next(ln for ln,name in calls if name=='scaler.step')
unscale_line=next(ln for ln,name in calls if name=='scaler.unscale_')
backward_line=next(ln for ln,name in calls if name=='scaler.scale(loss).backward')
add_line=next(ln for ln,name in calls if name=='p.grad.add_')
assert backward_line < add_line < unscale_line < step_line
endpoints=[n for n in ast.walk(fit) if isinstance(n,ast.If) and "endpoint == 'history_gradient'" in ast.unparse(n.test)]
assert len(endpoints)==1
train_source=data('tools/train_msvr_history_gradient.py').decode()
assert "if endpoint=='history_gradient':" in train_source
assert "selected = [(n,p) for n,p in model.named_parameters() if p.requires_grad and n.startswith('encoder.')]" in train_source
assert "assert not any(p.requires_grad for p in model.fusion.parameters())" in train_source
wrapper=trees['tools/run_msvr_history_gradient.py']
modules=sorted({n.value for n in ast.walk(wrapper) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('tools.')})
assert modules==['tools.check_msvr_history_gradient','tools.train_msvr_history_gradient','tools.verify_msvr_history_gradient']

fixed_source = 'evidence/msvr310_history_gradient_complete_source_20260908/source/'
source=js(fixed_source+'summary.json'); cpu=js(fixed_source+'cpu_verification.json')
assert inputs[fixed_source+'summary.json']['sha256']==top['fixed_file_sha256'][top['source_summary']]
assert inputs[fixed_source+'cpu_verification.json']['sha256']==top['fixed_file_sha256'][top['source_cpu']]
assert source['status']=='PASS_COMPLETE_FIXED_STATE_PROBE'
assert cpu['summary_sha256']==inputs[fixed_source+'summary.json']['sha256']
audit=js(top['source_audit'])
assert audit['verdict'] in ('PASS','WARN') and audit['deterministic_checks_status']=='pass'

result=dict(status='PASS_STATIC_CHECKS_WITH_INHERITED_NEWLINE_LIMIT',generated_at=datetime.now().astimezone().isoformat(),
            reviewer='gpt-6-astra',reasoning_effort='max',review_independence='same-family',acceptance_status='provisional',
            interpreter=sys.version,project_commit=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip(),
            syntax=syntax,config_binding_counts=dict(total=len(binding_checks),**dict(Counter(r['status'] for r in binding_checks))),
            config_bindings=binding_checks,remote_fixed_bindings_not_live_rehashed=remote_binding_count,
            protocol_checks=protocol_checks,queue_checks=queue_checks,
            gradient_ast_order=dict(backward=backward_line,history_add=add_line,unscale=unscale_line,step=step_line),
            endpoint_mutation_branches=1,stage_modules=modules,
            source_prerequisite_local_mirror_exact=True,
            assertions_executed_are_static_or_stdlib_metadata_only=True,
            project_module_imports=0,model_forwards=0,model_backwards=0,optimizer_updates=0,remote_commands=0)
assert not any(row['status']=='MISMATCH' for row in binding_checks)
for name, before in list(inputs.items()):
    assert hashlib.sha256((R/name).read_bytes()).hexdigest()==before['sha256'], ('input changed during check',name)
inputs['AUDITOR/independent_static_checks.py']={'bytes':Path(__file__).stat().st_size,'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(A/'audited_input_hashes.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(A/'independent_static_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('config_bindings','syntax')},ensure_ascii=False,indent=2))
