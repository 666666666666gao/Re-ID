"""Independently recompute identity diagnosis and its descriptive bootstrap."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(2)
ROOT=Path('/data/gaob/Re-ID/Trifusion')
OUT=ROOT/'.codex_tmp/integrity678_identity_checks.json'
assert not OUT.exists()
matrix=json.loads((ROOT/'logs/patch_memory_roles_recovery_20261001/accepted_matrix.json').read_text())
result={'status':'COMPLETE','no_neural_forward':True,'gpu_use':False,'rows':[]}
for ds in ('RGBNT201','RGBNT100','MSVR310'):
    aps={}
    for variant in ('local_memory','full_memory'):
        row=next(r for r in matrix['rows'] if r['dataset']==ds and r['variant']==variant)
        a=torch.load(Path(row['run_dir'])/'official_distances.pt',map_location='cpu',weights_only=False)
        field='scenes' if ds=='MSVR310' else 'cameras'
        qid,gid=a['query_ids'],a['gallery_ids'];qe,ge=a['query_'+field],a['gallery_'+field]
        values=[]
        for i,order in enumerate(np.argsort(a['fused'].numpy(),axis=1)):
            valid=order[~((gid[order]==qid[i])&(ge[order]==qe[i]))]
            ranks=np.flatnonzero(gid[valid]==qid[i])+1
            values.append(math.fsum((np.arange(1,len(ranks)+1)/ranks).tolist())/len(ranks))
        aps[variant]=np.array(values)
    delta=aps['full_memory']-aps['local_memory']
    changes=[{'identity':int(pid),'queries':int((qid==pid).sum()),'mean_delta_ap_points':float(delta[qid==pid].mean()*100)} for pid in np.unique(qid)]
    means=np.array([r['mean_delta_ap_points'] for r in changes])
    interval=np.percentile(np.random.default_rng(42).choice(means,size=(2000,len(means)),replace=True).mean(axis=1),[2.5,97.5]).tolist()
    p=ROOT/(f'logs/patch_memory_pair_diagnosis_20261001/{ds}.json' if ds!='RGBNT100' else 'logs/patch_memory_100_diagnosis_20261001/cpu/RGBNT100.json')
    prior=json.loads(p.read_text())
    assert len(changes)==len(prior['identity_changes'])
    assert all(x['identity']==y['identity'] and x['queries']==y['queries'] and abs(x['mean_delta_ap_points']-y['mean_delta_ap_points'])<1e-10 for x,y in zip(changes,prior['identity_changes']))
    assert abs(float(means.mean())-prior['identity_macro_mean_delta_ap_points'])<1e-10
    assert max(abs(x-y) for x,y in zip(interval,prior['identity_bootstrap_95_percentile_interval']))<1e-10
    assert int((means>1e-6).sum())==prior['identity_ap_improved'] and int((means<-1e-6).sum())==prior['identity_ap_worsened']
    result['rows'].append({'dataset':ds,'status':'PASS','identities':len(changes),'identity_macro_delta_ap_pp':float(means.mean()),'descriptive_bootstrap_95_interval':interval,
                           'source':str(p),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'boundary':'Fixed-model identity resampling, not uncertainty across training seeds or untouched test populations.'})
assert not torch.cuda.is_initialized()
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
