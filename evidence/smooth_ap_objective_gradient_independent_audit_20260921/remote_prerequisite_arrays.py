"""Independent CPU checks of frozen checkpoint identity and retained preflight vectors."""
from pathlib import Path
import hashlib,json,math
import numpy as np
import torch
torch.set_num_threads(4)
repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
qroot=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_v1_seed42_2e947a4/q1')
pre=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_objective_gradients_preflight_seed42_a2dec7f')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def js(p):return json.loads(p.read_bytes())
q=js(qroot/'summary.json');s=js(pre/'summary.json');v=js(pre/'preflight_verification.json')
spec=js(repo/'configs/MSVR310/TriFusion-smooth-ap-objective-gradients-v1.json')
base=js(repo/'configs/MSVR310/TriFusion-source-style-paired-v1.json')
b=js(Path(base['BASELINE']['SUMMARY']))
protocol=js(repo/spec['protocol'])
assert sha(qroot/'summary.json')==spec['q1_summary_sha256']
assert s['status']=='PASS_PREFLIGHT' and q['status']=='Q1_FAIL'
assert sha(pre/'summary.json')=='bc5aa928dc98f0694008d044753718c87877ad14fddc797b37acf8e6abcf98d0'
assert sha(pre/'preflight_verification.json')=='93fd2174c5084e565db65e8a0f09208132836c6a770c4b92726facf73e2d398a'
out=[];bindings=[];max_error=0.;tensors=0
def record(p,expected=None):
    actual=sha(p)
    if expected is not None:assert actual==expected,str(p)
    bindings.append(dict(path=str(p),bytes=p.stat().st_size,sha256=actual))
def stats(x,y):
    # Use NumPy dot products of persisted float32 vectors in independent float64 accumulation.
    a=[z.numpy().astype(np.float64).ravel() for z in x];c=[z.numpy().astype(np.float64).ravel() for z in y]
    xx=math.fsum(float(np.dot(z,z)) for z in a);yy=math.fsum(float(np.dot(z,z)) for z in c)
    dot=math.fsum(float(np.dot(z,w)) for z,w in zip(a,c))
    dd=math.fsum(float(np.dot(z-w,z-w)) for z,w in zip(a,c))
    return dict(first_norm=math.sqrt(xx),second_norm=math.sqrt(yy),difference_norm=math.sqrt(dd),cosine=dot/math.sqrt(xx*yy) if xx and yy else None)
for f in range(3):
    bp=Path(b['folds'][f]['checkpoint']);record(bp,b['folds'][f]['checkpoint_sha256'])
    baseline=torch.load(bp,map_location='cpu',weights_only=True)['model_state_dict']
    for end in ('control','smooth_ap'):
        original=q['folds'][f]['endpoints'][end];p=Path(original['checkpoint']);record(p,original['checkpoint_sha256'])
        payload=torch.load(p,map_location='cpu',weights_only=True)
        assert payload['fold']==f and payload['source_ids']==protocol['folds'][f]['source_ids'] and payload['heldout_ids']==protocol['folds'][f]['heldout_ids']
        assert payload['binding']==original['initialization'] and payload['config_sha256']==sha(repo/spec['training_config'])
        state={n:baseline[key] for n,key in payload['baseline_aliases'].items()};state.update(payload['role_state_dict'])
        h=hashlib.sha256()
        for name,tensor in sorted(state.items()):h.update(name.encode());h.update(tensor.contiguous().numpy().tobytes())
        assert h.hexdigest()==original['training']['final_state_sha256']
        cond=next(c for c in s['conditions'] if c['directory']==f'fold_{f}_{end}')
        assert cond['model_state_sha256']==h.hexdigest()
        names=[n for n in original['initialization']['trainable_names'] if n.startswith('encoder.')]
        assert cond['parameters']==names and len(names)==189
        directory=pre/cond['directory'];tpath=directory/'first_history_gradients.pt'
        record(tpath,cond['files'][tpath.name]['sha256'])
        record(directory/'steps.jsonl',cond['files']['steps.jsonl']['sha256'])
        vectors=torch.load(tpath,map_location='cpu',weights_only=True)
        assert vectors['names']==names and vectors['scale']==256.0
        rows=[json.loads(line) for line in (directory/'steps.jsonl').read_text().splitlines()]
        row=rows[vectors['step']-1]
        assert vectors['step']==cond['direct_history_proof']['step']==4
        for role in ('cnn','transformer','mamba'):
            ix=[i for i,n in enumerate(names) if n.startswith('encoder.'+role+'_')]
            for kind,ykey in [('fused_vs_other','other'),('fused_vs_full','full')]:
                actual=stats([vectors['fused'][i] for i in ix],[vectors[ykey][i] for i in ix])
                for key,value in actual.items():
                    expected=row['roles'][role][kind][key]
                    if value is None:assert expected is None
                    else:
                        error=abs(value-expected);max_error=max(max_error,error)
                        assert error<=1e-10*max(1,abs(expected)),(f,end,role,kind,key,error)
        for key in ('fused','other','full'):
            assert len(vectors[key])==189
            for name,t in zip(names,vectors[key]):assert tuple(t.shape)==tuple(state[name].shape) and bool(torch.isfinite(t).all())
            tensors+=len(vectors[key])
        sums=[x+y for x,y in zip(vectors['fused'],vectors['other'])]
        dec=stats(sums,vectors['full'])
        for key,value in dec.items():
            expected=row['full_decomposition'][key]
            if value is None:assert expected is None
            else:assert abs(value-expected)<=1e-10*max(1,abs(expected))
        out.append(dict(fold=f,endpoint=end,checkpoint_state_tensors=len(state),encoder_tensors=len(names),
                        all_trainable_tensors=len(original['initialization']['trainable_names']),
                        state_sha256=h.hexdigest(),vector_step=vectors['step'],preflight_vector_tensors=567))
        del payload,state,vectors,sums
    del baseline
print(json.dumps(dict(status='PASS_FROZEN_CHECKPOINTS_AND_SAVED_PREFLIGHT_VECTORS',conditions=out,
    checked_vector_tensors=tensors,max_scalar_absolute_error=max_error,files=bindings,
    new_model_forwards=0,new_backward_passes=0,new_optimizer_updates=0,
    limitation='Recomputed statistics from saved preflight vectors, not their generation; no full-source vectors exist to reconstruct.')))
