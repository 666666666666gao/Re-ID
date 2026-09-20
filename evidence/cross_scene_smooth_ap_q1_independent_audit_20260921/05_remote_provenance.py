"""Read-only immutable source, label and initialization provenance; no training."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import subprocess
import datetime
import torch

ROOT=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
RUN=Path('/root/trifusion-storage/artifacts/msvr310_cross_scene_smooth_ap_v1_seed42_d35864d')
read=lambda p:json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def state_sha(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

bindings=[];configs={};path=ROOT/'configs/MSVR310/TriFusion-cross-scene-smooth-ap-paired-v1.json'
while True:
    c=read(path);configs[str(path)]=c
    for key in ('project_file_sha256','project_source_file_sha256','fixed_file_sha256'):
        for name,value in c.get(key,{}).items():
            p=ROOT/name
            bindings.append(dict(owner=str(path),key=key,path=str(p),expected=value,actual=sha(p)))
    nxt=c.get('previous_config') or c.get('coordinate_config') or c.get('memory_config') or c.get('base_config')
    if not nxt:break
    path=ROOT/nxt
base=c;signal_path=ROOT/base['BASELINE']['CONFIG'];signal=read(signal_path)
for p,value in ((signal_path,base['BASELINE']['CONFIG_SHA256']),
                (Path(base['BASELINE']['SUMMARY']),base['BASELINE']['SUMMARY_SHA256']),
                (ROOT/base['SOURCE_METADATA']['PATH'],base['SOURCE_METADATA']['SHA256']),
                (ROOT/signal['protocol'],signal['protocol_sha256']),
                (Path(signal['clip_weight']),signal['clip_weight_sha256'])):
    bindings.append(dict(owner='resolved_initialization',path=str(p),expected=value,actual=sha(p)))
for key,parent in (('project_source_file_sha256',ROOT),('signal_source_file_sha256',Path(signal['signal_source']))):
    for name,value in signal[key].items():
        p=parent/name;bindings.append(dict(owner=str(signal_path),key=key,path=str(p),expected=value,actual=sha(p)))
support=read(ROOT/'evidence/smooth_ap_cross_scene_support_20260909/smooth_ap_cross_scene_support_20260909.json')
for name,value in support['input_bindings'].items():
    bindings.append(dict(owner='source_support',path=str(ROOT/name),expected=value,actual=sha(ROOT/name)))
assert all(x['expected']==x['actual'] for x in bindings)
protocol=read(ROOT/signal['protocol']);rows=protocol['records']
labels=read(ROOT/'evidence/vehicle_query_protocol_labels_20260905.json')
train=next(d for d in labels['datasets'] if d['dataset']=='MSVR310')['record_manifest']['bounding_box_train']
assert sha(ROOT/'evidence/vehicle_query_protocol_labels_20260905.json')==protocol['label_evidence_sha256']
assert len(rows)==len(train)==1032
dataset=Path(signal['dataset_root'])
for r,t in zip(rows,train):
    n=Path(r['paths'][0]).name
    assert (r['identity'],r['camera'],r['scene'])==(int(n[:4]),int(n[11]),int(n[6:9]))
    assert all(r[k]==t[k] for k in ('identity','camera','scene'))
    assert r['paths'][0]=='bounding_box_train/'+t['path']
    assert all((dataset/p).is_file() for p in r['paths'])
expected_names={p for r in rows for p in r['paths']}
actual_names={p.relative_to(dataset).as_posix() for p in (dataset/'bounding_box_train').glob('*/*/*.jpg')}
assert expected_names==actual_names and len(actual_names)==3096
ids=sorted({r['identity'] for r in rows})
scenes={i:{r['scene'] for r in rows if r['identity']==i} for i in ids}
groups=[[i for i in ids if len(scenes[i])>1],[i for i in ids if len(scenes[i])==1]]
baseline=read(base['BASELINE']['SUMMARY']);q1=read(RUN/'q1/summary.json')
protocol_results=[];checkpoint_results=[];baseline_steps=0
for f,b0,qr in zip(protocol['folds'],baseline['folds'],q1['folds']):
    k=f['fold'];held=sorted([i for g in groups for n,i in enumerate(g) if n%3==k])
    assert held==f['heldout_ids'];assert sorted(set(ids)-set(held))==f['source_ids']
    src=[r['index'] for r in rows if r['identity'] in f['source_ids']]
    gal=[r['index'] for r in rows if r['identity'] in held]
    assert src==f['source_record_indices'] and gal==f['gallery_record_indices']
    legal=[];excluded=[]
    for pos,idx in enumerate(gal):
        r=rows[idx];positives=sum(rows[j]['identity']==r['identity'] and rows[j]['scene']!=r['scene'] for j in gal)
        removed=sum(rows[j]['identity']==r['identity'] and rows[j]['scene']==r['scene'] for j in gal)
        if positives:
            legal.append(dict(record_index=idx,gallery_position=pos,identity=r['identity'],scene=r['scene'],
                              valid_positives=positives,removed_same_identity_same_scene=removed,
                              retained_gallery=len(gal)-removed,
                              negative_identity_distractors=sum(rows[j]['identity']!=r['identity'] for j in gal)))
        else:excluded.append(idx)
    assert legal==f['query_rows'] and excluded==f['excluded_query_record_indices']
    bt=b0['training']
    assert bt['epochs']==50 and bt['optimizer_steps']==650
    for step in bt['steps']:assert set(step['sampled_record_indices'])<=set(src)
    baseline_steps+=len(bt['steps'])
    bp=Path(b0['checkpoint']);assert sha(bp)==b0['checkpoint_sha256']
    payload=torch.load(bp,map_location='cpu',weights_only=True)
    assert payload['source_ids']==f['source_ids'] and payload['heldout_ids']==held and payload['fold']==k
    original=payload['model_state_dict'];assert state_sha(original)==bt['final_state_sha256']
    protocol_results.append(dict(fold=k,source_ids=len(f['source_ids']),heldout_ids=len(held),source_records=len(src),
                                 gallery_records=len(gal),legal_queries=len(legal),excluded_queries_retained=len(excluded),
                                 gallery_only_identities=sum(len(scenes[i])==1 for i in held),baseline_steps=len(bt['steps'])))
    for end,item in qr['endpoints'].items():
        p=Path(item['checkpoint']);ck=torch.load(p,map_location='cpu',weights_only=True)
        assert sha(p)==item['checkpoint_sha256'];assert ck['binding']==item['initialization']
        assert ck['config_sha256']==q1['config_sha256'] and ck['fold']==k
        assert ck['source_ids']==f['source_ids'] and ck['heldout_ids']==held
        assert ck['binding']['signal_checkpoint_sha256']==b0['checkpoint_sha256']
        assert ck['binding']['signal_state_sha256']==state_sha(original)
        assert not ck['binding']['role_weights_loaded'] and ck['binding']['role_initialization_seed']==42
        full={name:original[alias] for name,alias in ck['baseline_aliases'].items()}
        assert not(set(full)&set(ck['role_state_dict']))
        full.update(ck['role_state_dict'])
        final=state_sha(full);tr=item['training']
        assert final==tr['final_state_sha256']==item['strict_reload_state_sha256']
        neck_biases=['fused_neck.bias']+[g+'.'+e+'.bias' for g in ('branch_necks','residual_necks') for e in ('cnn','transformer','mamba')]
        frozen={n:v for n,v in full.items() if n.startswith('baseline.') or n in neck_biases}
        assert all(torch.count_nonzero(full[n])==0 for n in neck_biases)
        assert state_sha(frozen)==tr['frozen_state_after_sha256']==tr['frozen_state_before_sha256']
        assert tr['signal_state_before_sha256']==tr['signal_state_after_sha256']==state_sha(original)
        checkpoint_results.append(dict(fold=k,endpoint=end,checkpoint=str(p),sha256=sha(p),
            baseline_alias_tensors=len(ck['baseline_aliases']),role_state_tensors=len(ck['role_state_dict']),
            full_state_tensors=len(full),full_state_sha256=final,frozen_state_sha256=state_sha(frozen),
            trainable_tensors=len(ck['binding']['trainable_names']),
            baseline_checkpoint_sha256=b0['checkpoint_sha256']))
source=Path(signal['signal_source'])
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==signal['signal_commit']
assert hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=source)).hexdigest()==signal['signal_diff_sha256']
code={str(source/n):(source/n).read_text() for n in ('data/datasets/msvr310.py','utils/metrics.py','data/datasets/sampler.py')}
execution='d35864d6411591e05c8ac3e5164ebae48063ad99'
execution_matches=[]
for p in sorted({b['path'] for b in bindings if b['path'].startswith(str(ROOT)+'/')}):
    rel=Path(p).relative_to(ROOT).as_posix()
    blob=subprocess.check_output(['git','show',execution+':'+rel],cwd=ROOT)
    execution_matches.append(dict(path=p,current_sha256=sha(p),execution_blob_sha256=hashlib.sha256(blob).hexdigest()))
assert all(r['current_sha256']==r['execution_blob_sha256'] for r in execution_matches)
print(json.dumps(dict(status='PASS_COMPLETE_SOURCE_AND_INITIALIZATION_PROVENANCE',
    observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),bindings=bindings,
    execution_source_matches=execution_matches,protocol=protocol_results,baseline_training_steps_checked=baseline_steps,
    dataset_training_filename_count=len(actual_names),checkpoint_results=checkpoint_results,source_texts=code,
    source_support_status=support['status'],
    limits=['Original parameter-gradient trajectories and initial random role states are not regenerated.',
            'Image existence and filenames are checked; image bytes and official-test paths are not read.'],
    model_forwards=0,optimizer_updates=0,official_test_reads=0)))
