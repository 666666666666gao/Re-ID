"""Metadata/protocol reconstruction and CPU-only edge algebra; no images/models."""
import ast, hashlib, importlib.metadata, importlib.util, json, math, sys, time, subprocess
from pathlib import Path
from collections import Counter
import numpy as np
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(2)
R=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
RUN=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_v1_seed42_2e947a4')
out={'generated_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'files':{},'texts':{},'dataset':{},'protocol':{},'synthetic':{}}
def read(p):
    data=Path(p).read_bytes();out['files'][str(p)]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    if Path(p).suffix in ('.py','.json','.yml'):out['texts'][str(p)]=data.decode('utf-8')
    return data
def j(p):return json.loads(read(p))
protocol=j(R/'protocols/msvr310_train_oof_v1.json');rows=protocol['records']
evidence=j(R/'evidence/vehicle_query_protocol_labels_20260905.json')
assert out['files'][str(R/'evidence/vehicle_query_protocol_labels_20260905.json')]['sha256']==protocol['label_evidence_sha256']
install=j(R/'evidence/msvr310_dataset_install_20260905.json')
labels=next(x for x in evidence['datasets'] if x['dataset']=='MSVR310')['record_manifest']['bounding_box_train']
expected=[]
for idx,item in enumerate(labels):
    name=Path(item['path']).name;parts=Path(item['path']).parts
    assert parts[1]=='vis' and (int(name[:4]),int(name[11]),int(name[6:9]))==(item['identity'],item['camera'],item['scene'])
    expected.append({'index':idx,'identity':item['identity'],'camera':item['camera'],'scene':item['scene'],'paths':[f'bounding_box_train/{parts[0]}/{m}/{name}' for m in ('vis','ni','th')]})
assert expected==rows and len(rows)==1032
root=Path(install['root']);train=root/'bounding_box_train'
actual=sorted(p.relative_to(root).as_posix() for p in train.rglob('*.jpg'))
expected_paths=sorted(p for r in rows for p in r['paths'])
assert actual==expected_paths and len(actual)==3096
assert all((root/p).is_file() and (root/p).stat().st_size>0 for p in actual)
ids=sorted({r['identity'] for r in rows});eligible=[i for i in ids if len({r['scene'] for r in rows if r['identity']==i})>1]
ineligible=[i for i in ids if i not in eligible]
held=[set() for _ in range(3)]
for group in (eligible,ineligible):
    for idx,identity in enumerate(group):held[idx%3].add(identity)
qtotal=0;foldchecks=[]
for f in protocol['folds']:
    fi=f['fold'];assert sorted(held[fi])==f['heldout_ids']
    assert [i for i in ids if i not in held[fi]]==f['source_ids']
    gallery=[r for r in rows if r['identity'] in held[fi]]
    assert [r['index'] for r in gallery]==f['gallery_record_indices']
    queries=[];excluded=[]
    for pos,r in enumerate(gallery):
        positives=sum(t['identity']==r['identity'] and t['scene']!=r['scene'] for t in gallery)
        removed=sum(t['identity']==r['identity'] and t['scene']==r['scene'] for t in gallery)
        if positives:
            queries.append(dict(record_index=r['index'],gallery_position=pos,identity=r['identity'],scene=r['scene'],valid_positives=positives,removed_same_identity_same_scene=removed,retained_gallery=len(gallery)-removed,negative_identity_distractors=sum(t['identity']!=r['identity'] for t in gallery)))
        else:excluded.append(r['index'])
    assert queries==f['query_rows'] and excluded==f['excluded_query_record_indices']
    qtotal+=len(queries);foldchecks.append({'fold':fi,'gallery':len(gallery),'eligible_queries':len(queries),'gallery_only_distractor_records':len(excluded)})
assert len(ids)==155 and len(eligible)==60 and qtotal==600
out['dataset']={'source_filenames_metadata_matched':3096,'triplets':1032,'identities':155,'record_manifest_and_protocol_exact':True,'image_content_bytes_read':0,'original_archive_rehashed':False,'install_receipt_sha256':out['files'][str(R/'evidence/msvr310_dataset_install_20260905.json')]['sha256']}
out['protocol']={'round_robin_split_reconstructed':True,'source_label_complete_path_contract':True,'folds':foldchecks,'eligible_queries':qtotal,'eligible_identities':len(eligible),'scene_filter':'same identity AND same scene'}

# Read every local code dependency mirrored by the current audited import path.
todo=[R/'tools/train_msvr_smooth_ap.py',R/'tools/check_msvr_smooth_ap.py',R/'tools/verify_msvr_smooth_ap.py',R/'tools/run_msvr_smooth_ap.py']
seen=set();signal=Path('/root/autodl-tmp/trifusion-v2/comparators/Signal-cd1b0a6')
todo += [signal/'modeling/make_model.py',signal/'modeling/make_model_clipreid.py',signal/'modeling/clip/clip.py',signal/'modeling/clip/model.py',signal/'modeling/AddModule/useA.py',signal/'modeling/AddModule/useB.py']
while todo:
    path=todo.pop()
    if path in seen or not path.is_file():continue
    seen.add(path);data=read(path)
    tree=ast.parse(data)
    for n in ast.walk(tree):
        candidates=[]
        if isinstance(n,ast.Import):candidates=[a.name for a in n.names]
        elif isinstance(n,ast.ImportFrom):
            if n.level:
                basepath=path.parent
                for _ in range(n.level-1):basepath=basepath.parent
                prefix=basepath/(n.module.replace('.','/') if n.module else '')
                for p in [prefix,*[prefix/a.name for a in n.names]]:
                    todo += [Path(str(p)+'.py'),p/'__init__.py']
            elif n.module:candidates=[n.module,*[n.module+'.'+a.name for a in n.names]]
        for name in candidates:
            for basepath in (R,R/'modeling',signal):
                p=basepath/name.replace('.','/');todo += [Path(str(p)+'.py'),p/'__init__.py']
out['transitive_code_files']=len(seen)
out['packages']={p:importlib.metadata.version(p) for p in ('torch','numpy','torchvision','mamba-ssm','causal-conv1d','einops','timm')}
out['package_sources']={}
site=Path('/root/miniconda3/envs/tri_reid/lib/python3.10/site-packages')
for rel in ('mamba_ssm/modules/mamba_simple.py','mamba_ssm/ops/selective_scan_interface.py','causal_conv1d/causal_conv1d_interface.py'):
    p=site/rel;read(p);out['package_sources'][rel]=out['files'][str(p)]
out['current_head']=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()
out['files_changed_since_execution_commit']=subprocess.check_output(['git','-C',str(R),'diff','--name-only','2e947a4325144e37fed638105ac954e7e54b5fe5','HEAD'],text=True).splitlines()

ms=importlib.util.spec_from_file_location('smooth_ap_subject',R/'tools/msvr_smooth_ap.py');m=importlib.util.module_from_spec(ms);ms.loader.exec_module(m)
d=torch.tensor([[0,.94,.99,1.02,.98,1.04],[.94,0,1.03,.97,1.01,.99],[.99,1.03,0,.96,1.04,.98],[1.02,.97,.96,0,.99,1.01]],dtype=torch.float64)
ids=[0,0,1,1];mids=[0,1]
def pure_ap(a,labels):
    score=1-a*a/2;ap=[]
    for i in range(len(a)):
        pos=[k for k in range(a.shape[1]) if labels[k]==labels[i] and k!=i]
        terms=[]
        for p in pos:
            sig={k:1/(1+math.exp(-(score[i,k]-score[i,p])/.01)) for k in range(a.shape[1]) if k not in (i,p)}
            terms.append((1+sum(v for k,v in sig.items() if k in pos))/(1+sum(sig.values())))
        ap.append(sum(terms)/len(terms))
    return np.asarray(ap)
checks=[]
for count in (0,2):
    a=d[:,:4+count].clone().requires_grad_();loss,ap=m.smooth_ap_from_distances(a[:,:4],a[:,4:],ids,mids[:count]);g=torch.autograd.grad(loss,a)[0].numpy();arr=a.detach().numpy()
    ref=pure_ap(arr,ids+mids[:count]);errors=[]
    for i in range(4):
        for k in range(4+count):
            plus=arr.copy();minus=arr.copy();plus[i,k]+=1e-6;minus[i,k]-=1e-6
            fd=-(pure_ap(plus,ids+mids[:count]).mean()-pure_ap(minus,ids+mids[:count]).mean())/2e-6
            errors.append(abs(fd-g[i,k]))
    assert max(errors)<1e-6 and np.max(np.abs(ref-ap.detach().numpy()))<1e-12
    perm=[2,3,0,1];hp=list(reversed(range(count)))
    perm_loss,_=m.smooth_ap_from_distances(a[:,:4][perm][:,perm],a[:,4:][perm][:,hp],[ids[i] for i in perm],[mids[i] for i in hp])
    assert abs(float(loss.detach()-perm_loss.detach()))<1e-12
    checks.append({'history_columns':count,'finite_difference_entries':len(errors),'max_finite_difference_error':max(errors),'permutation_exact_to_1e12':True})
tie=torch.ones(4,6,dtype=torch.float64);tie[:,:4].fill_diagonal_(0);tie.requires_grad_()
h,s,ap=m.paired_objectives(tie[:,:4],tie[:,4:],ids,mids)
gh=torch.autograd.grad(h,tie)[0].numpy();wanted=np.zeros((4,6))
for i in range(4):
    current_p=next(k for k in range(4) if k!=i and ids[k]==ids[i]);current_n=next(k for k in range(4) if ids[k]!=ids[i]);history_p=4+mids.index(ids[i]);history_n=4+mids.index(1-ids[i])
    wanted[i,[current_p,history_p]]=1/8;wanted[i,[current_n,history_n]]=-1/8
assert np.array_equal(gh,wanted) and torch.equal(ap,torch.full((4,),.5,dtype=torch.float64))
gen=torch.Generator().manual_seed(42);unit=torch.nn.functional.normalize(torch.randn(12,19,generator=gen,dtype=torch.float64),dim=1)
cosine_error=float((1-torch.cdist(unit,unit).square()/2-unit@unit.T).abs().max());assert cosine_error<1e-12
out['synthetic']={'finite_difference':checks,'current_history_tie_first_within_group_equal_split_between_groups_exact':True,'tie_ap':.5,'unit_euclidean_cosine_max_error':cosine_error,'dataset_metric_claim':False}
assert not torch.cuda.is_initialized()
out['status']='PASS_METADATA_PROVENANCE_SYNTHETIC_EDGE_ALGEBRA'
print(json.dumps(out))
