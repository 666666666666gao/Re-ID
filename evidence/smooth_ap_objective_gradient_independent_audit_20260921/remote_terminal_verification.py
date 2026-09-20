"""Independent NumPy/stdlib verification of all terminal source scalar rows.

No project verifier or analyzer is imported. No models, images or full source
parameter gradients are reconstructed. All writes are in the caller's local
audit directory through captured stdout/stderr.
"""
from pathlib import Path
from collections import OrderedDict,Counter
import csv,hashlib,json,math
import numpy as np

repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
root=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_objective_gradients_source_seed42_c6fbfb4_20260920')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def js(path):return json.loads(path.read_bytes())
inputs={}
def bind(path,expected=None):
    digest=sha(path)
    if expected is not None:assert digest==expected,(str(path),digest,expected)
    inputs[str(path)]=dict(path=str(path),bytes=path.stat().st_size,sha256=digest)
    return digest
pipelines={}
for suffix in ('_pipeline.json','_verification_pipeline.json','_analysis_pipeline.json'):
    path=Path(str(root)+suffix);p=js(path);bind(path)
    assert p['status']=='COMPLETE' and p['exit_code']==0,(suffix,p)
    pipelines[suffix]=p
summary=js(root/'summary.json');verification=js(root/'source_verification.json');analysis=js(root/'analysis/analysis.json')
summary_hash=bind(root/'summary.json');verification_hash=bind(root/'source_verification.json')
assert summary['status']=='COMPLETE_SOURCE_OBJECTIVE_GRADIENTS' and summary['mode']=='source'
assert verification['status']=='PASS_COMPLETE_SOURCE_OBJECTIVE_GRADIENT_LEDGER'
assert analysis['status']=='COMPLETE_VERIFIED_SOURCE_GRADIENT_ANALYSIS'
assert verification['summary_sha256']==analysis['summary_sha256']==summary_hash
assert analysis['verification_sha256']==verification_hash
assert verification['steps']==1560 and verification['role_rows']==analysis['role_rows']==4680 and analysis['groups']==72
specpath=repo/'configs/MSVR310/TriFusion-smooth-ap-objective-gradients-v1.json';spec=js(specpath)
bind(specpath,summary['contract_sha256'])
for name,digest in spec['project_file_sha256'].items():bind(repo/name,digest)
qpath=Path(spec['q1_summary']);bind(qpath,spec['q1_summary_sha256']);q1=js(qpath)
assert q1['status']=='Q1_FAIL' and q1['optimizer_steps']==1560
protocol=js(repo/spec['protocol'])
config=js(repo/'configs/MSVR310/TriFusion-source-style-paired-v1.json')
metadata=js(repo/config['SOURCE_METADATA']['PATH']);bind(repo/config['SOURCE_METADATA']['PATH'],config['SOURCE_METADATA']['SHA256'])
weights=config['LOSS'];roles=('cnn','transformer','mamba');ends=('control','smooth_ap')
expected=[f'fold_{f}_{end}' for f in range(3) for end in ends]
assert [x['directory'] for x in summary['conditions']]==expected
component_weights={'id_fused':weights['ID_FUSED'],'triplet_fused':weights['TRIPLET_FUSED']}
for role in roles:
    component_weights.update({f'id_{role}':weights['ID_BRANCH'],f'triplet_{role}':weights['TRIPLET_BRANCH'],
                             f'id_residual_{role}':weights['ID_RESIDUAL'],f'triplet_residual_{role}':weights['TRIPLET_RESIDUAL']})
assert len(component_weights)==14 and weights['TRIPLET_FUSED']==1.0
max_loss_error=0.;max_norm_identity_error=0.;max_denominator_error=0.;steps=0;values_count=0
max_current_decomposition=0.;max_full_decomposition=0.;max_direct_error=0.
max_repeat_relative={'current':0.,'history':0.};max_symmetry_error=0.;distance_min=math.inf;distance_max=-math.inf
conditions=[];all_rows=[];paired={};costs=[]
def check_comparison(c):
    global max_norm_identity_error
    a,b,d=[c[k] for k in ('first_norm','second_norm','difference_norm')]
    assert all(math.isfinite(z) and z>=0 for z in (a,b,d))
    co=c['cosine']
    if a==0 or b==0:
        assert co is None and math.isclose(d,abs(a-b),rel_tol=1e-12,abs_tol=1e-12)
    else:
        assert math.isfinite(co) and abs(co)<=1+1e-10
        error=abs(d*d-(a*a+b*b-2*a*b*co))/(1+a*a+b*b)
        max_norm_identity_error=max(max_norm_identity_error,error);assert error<=1e-10
def smooth_loss(distances,ids):
    # Direct independent soft ranks over every valid candidate position.
    similarity=1.0-distances.astype(np.float64)**2/2.0
    anchors=[]
    for anchor in range(64):
        keep=np.arange(len(ids))!=anchor
        pos=np.flatnonzero((ids==ids[anchor])&keep)
        assert len(pos)>0
        z=(similarity[anchor,None,:]-similarity[anchor,pos,None])/.01
        soft=1.0/(1.0+np.exp(-z))
        soft[:,~keep]=0.;soft[np.arange(len(pos)),pos]=0.
        # Include the positive itself once via the explicit +1; exclude it from comparisons.
        ranks=1.+soft.sum(axis=1)
        positive_ranks=1.+soft[:,ids==ids[anchor]].sum(axis=1)
        anchors.append(float(np.mean(positive_ranks/ranks)))
    return 1.-float(np.mean(anchors))
for index,cond in enumerate(summary['conditions']):
    fold=index//2;end=ends[index%2];f=protocol['folds'][fold];directory=root/cond['directory']
    assert cond['status']=='PASS_FIXED_STATE_OBJECTIVE_GRADIENTS' and cond['steps']==260
    assert cond['model_state_unchanged'] and cond['gradients_absent']
    assert cond['optimizer_updates']==cond['heldout_record_forwards']==cond['official_image_reads']==0
    original=q1['folds'][fold]['endpoints'][end]
    bind(Path(original['checkpoint']),original['checkpoint_sha256'])
    assert cond['model_state_sha256']==original['training']['final_state_sha256']
    names=[n for n in original['initialization']['trainable_names'] if n.startswith('encoder.')]
    assert cond['parameters']==names and len(names)==len(set(names))==189
    assert set(names)=={n for role in roles for n in names if n.startswith('encoder.'+role+'_')}
    for name,entry in cond['files'].items():
        assert (directory/name).stat().st_size==entry['bytes'];bind(directory/name,entry['sha256'])
    receipt=js(directory/'receipt.json');assert receipt=={k:v for k,v in cond.items() if k!='directory'}
    qlog=Path(original['checkpoint']).parent/'memory_steps.jsonl'
    bind(qlog,original['training']['audit_files']['memory_steps.jsonl']['sha256'])
    reference=[json.loads(s) for s in qlog.read_text().splitlines()]
    rows=[json.loads(s) for s in (directory/'steps.jsonl').read_text().splitlines()]
    assert len(rows)==len(reference)==260
    paired[(fold,end)]=[(r['record_indices'],r['pixel_sha256'],r['memory']) for r in rows]
    cache=OrderedDict();seen=set();source=set(f['source_record_indices']);used=0;memory_rows=0
    direct=cond['direct_history_proof'];check_comparison(direct['comparison'])
    direct_ratio=direct['comparison']['difference_norm']/max(direct['comparison']['first_norm'],direct['comparison']['second_norm'])
    assert math.isclose(direct_ratio,direct['relative_error'],rel_tol=1e-12,abs_tol=1e-15)
    assert direct['step']==67 and direct['history_groups']==1 and direct['tolerance']==.005 and direct_ratio<=.005
    max_direct_error=max(max_direct_error,direct_ratio)
    data=np.memmap(directory/'distances.f32',mode='r',dtype='<f4')
    for step,row in enumerate(rows):
        assert row['step']==step+1 and row['epoch']==step//13+1 and row['scale']==256.
        assert row['record_indices']==reference[step]['record_indices']==metadata['folds'][fold]['batches'][step]['record_indices']
        indices=row['record_indices'];seen.update(indices);assert len(indices)==64 and set(indices)<=source
        assert not set(indices)&set(f['gallery_record_indices'])
        assert row['pixel_sha256']==reference[step]['pixel_sha256']
        assert row['identities']==[protocol['records'][i]['identity'] for i in indices]
        assert sorted(Counter(row['identities']).values())==[8]*8
        for i in list(cache):
            if step-cache[i]>8:del cache[i]
        mem=[dict(record_index=i,identity=protocol['records'][i]['identity'],scene=protocol['records'][i]['scene'],stored_step=k,age=step-k)
             for i,k in cache.items() if i not in set(indices)]
        assert row['memory']==mem and len(mem)<=512
        active=end if step>=65 else 'control';assert row['active_fused_metric']==active
        count=64*(64+len(mem));assert row['distance_float_count']==count and row['distance_offset_bytes']==4*used
        part=np.asarray(data[used:used+count]);assert len(part)==count and np.isfinite(part).all()
        distance_min=min(distance_min,float(part.min()));distance_max=max(distance_max,float(part.max()))
        assert part.min()>=0 and part.max()<=2.00001
        d=part.reshape(64,64+len(mem)).astype(np.float64) if not len(mem) else np.concatenate((part[:4096].reshape(64,64),part[4096:].reshape(64,len(mem))),axis=1).astype(np.float64)
        max_symmetry_error=max(max_symmetry_error,float(np.max(np.abs(d[:,:64]-d[:,:64].T))))
        ids=np.asarray(row['identities']+[r['identity'] for r in mem])
        positive=np.asarray(row['identities'])[:,None]==ids[None,:]
        positive[np.arange(64),np.arange(64)]=False
        negative=np.asarray(row['identities'])[:,None]!=ids[None,:]
        assert positive.any(axis=1).all() and negative.any(axis=1).all()
        hard=float(np.maximum(0,np.where(positive,d,-np.inf).max(axis=1)-np.where(negative,d,np.inf).min(axis=1)+.3).mean())
        selected=hard if active=='control' else smooth_loss(d,ids)
        assert set(row['components'])==set(component_weights) and row['components']['triplet_fused']==row['fused_loss']
        assert all(math.isfinite(x) for x in row['components'].values())
        total=math.fsum(row['components'][k]*w for k,w in component_weights.items())
        other=math.fsum(row['components'][k]*w for k,w in component_weights.items() if k!='triplet_fused')
        errors=(abs(selected-row['fused_loss']),abs(total-row['total_loss']),abs(other-row['other_loss']))
        max_loss_error=max(max_loss_error,*errors);assert max(errors)<=2e-6,(cond['directory'],step,errors)
        assert set(row['roles'])==set(roles)
        for role in roles:
            metrics=row['roles'][role]
            assert set(metrics)=={'fused_vs_other','fused_vs_full','current_repeat','history_repeat'}
            for comparison in metrics.values():check_comparison(comparison)
            fo=metrics['fused_vs_other'];ft=metrics['fused_vs_full'];assert fo['first_norm']==ft['first_norm']
            for rep in ('current','history'):
                c=metrics[rep+'_repeat'];den=max(c['first_norm'],c['second_norm'])
                relative=c['difference_norm']/den if den else 0.
                max_repeat_relative[rep]=max(max_repeat_relative[rep],relative)
            all_rows.append(dict(fold=fold,endpoint=end,step=step+1,epoch=row['epoch'],role=role,active_fused_metric=active,
                fused_norm=fo['first_norm'],other_norm=fo['second_norm'],full_norm=ft['second_norm'],
                fused_other_ratio=fo['first_norm']/fo['second_norm'] if fo['second_norm'] else None,
                fused_full_ratio=ft['first_norm']/ft['second_norm'] if ft['second_norm'] else None,
                fused_other_cosine=fo['cosine'],fused_full_cosine=ft['cosine'],
                current_repeat_difference=metrics['current_repeat']['difference_norm'],history_repeat_difference=metrics['history_repeat']['difference_norm'],
                history_repeat_norm=metrics['history_repeat']['first_norm'],fused_loss=row['fused_loss'],other_loss=row['other_loss'],
                current_identity_error=row['current_decomposition']['relative_to_sum_of_component_norms'],
                full_identity_error=row['full_decomposition']['relative_to_sum_of_component_norms']))
        onorm=math.sqrt(math.fsum(row['roles'][r]['fused_vs_other']['second_norm']**2 for r in roles))
        for kind,metric in [('current_decomposition','current_repeat'),('full_decomposition','fused_vs_other')]:
            c=row[kind];check_comparison(c)
            fnorm=math.sqrt(math.fsum(row['roles'][r][metric]['first_norm']**2 for r in roles))
            den=fnorm+onorm;ratio=c['difference_norm']/den if den else 0.
            max_denominator_error=max(max_denominator_error,abs(ratio-c['relative_to_sum_of_component_norms']))
            assert math.isclose(ratio,c['relative_to_sum_of_component_norms'],rel_tol=1e-10,abs_tol=1e-14)
            assert c['tolerance']==.005 and ratio<=.005
        max_current_decomposition=max(max_current_decomposition,row['current_decomposition']['relative_to_sum_of_component_norms'])
        max_full_decomposition=max(max_full_decomposition,row['full_decomposition']['relative_to_sum_of_component_norms'])
        available=sorted({r['stored_step'] for r in mem});selected_groups=row['history_groups']
        assert selected_groups==sorted(set(selected_groups)) and set(selected_groups)<=set(available)
        assert row['refresh_record_forwards']==64*len(available)
        assert row['history_vjp_record_forwards']==64*len(selected_groups)
        assert row['direct_check_record_forwards']==(64 if step+1==67 else 0)
        if not mem:assert all(row['roles'][r]['history_repeat']['first_norm']==0 for r in roles)
        memory_rows+=len(mem);used+=count;steps+=1
        if step>=65:
            for i in indices:cache.pop(i,None);cache[i]=step
            while len(cache)>512:cache.popitem(last=False)
    assert seen==source and sorted(seen)==cond['observed_records'] and used==len(data)
    values_count+=used;del data
    cost=dict(fold=fold,endpoint=end,current_record_forwards=260*64,
              extra_record_forwards=64+sum(r['refresh_record_forwards']+r['history_vjp_record_forwards']+r['direct_check_record_forwards'] for r in rows),
              refresh_record_forwards=sum(r['refresh_record_forwards'] for r in rows),history_vjp_record_forwards=sum(r['history_vjp_record_forwards'] for r in rows),
              direct_check_record_forwards=sum(r['direct_check_record_forwards'] for r in rows),initial_reencode_record_forwards=64,
              elapsed_seconds=cond['elapsed_seconds'],peak_allocated_mib=cond['peak_allocated_mib'])
    assert cost['extra_record_forwards']==cond['extra_record_forwards'];costs.append(cost)
    conditions.append(dict(fold=fold,endpoint=end,steps=len(rows),source_records=len(seen),source_identities=len(f['source_ids']),
        distance_values=used,memory_candidate_rows=memory_rows,parameter_role_counts={r:sum(n.startswith('encoder.'+r+'_') for n in names) for r in roles}))
for fold in range(3):assert paired[(fold,'control')]==paired[(fold,'smooth_ap')]
assert steps==1560 and len(all_rows)==4680 and values_count==verification['distance_values']
assert costs==analysis['costs']
windows={'all':(1,260),'warmup':(1,65),'active':(66,260),'last65':(196,260)}
metric_names=('fused_norm','other_norm','full_norm','fused_other_ratio','fused_full_ratio','fused_other_cosine','fused_full_cosine',
              'current_repeat_difference','history_repeat_difference','history_repeat_norm','fused_loss','other_loss','current_identity_error','full_identity_error')
def distribution(values):
    a=np.asarray([v for v in values if v is not None],dtype=np.float64)
    if not len(a):return dict(count=0,undefined=len(values),zeros=0,mean=None,median=None,p10=None,p90=None,minimum=None,maximum=None)
    return dict(count=len(a),undefined=len(values)-len(a),zeros=int((a==0).sum()),mean=float(np.mean(a)),median=float(np.median(a)),
                p10=float(np.quantile(a,.1,method='linear')),p90=float(np.quantile(a,.9,method='linear')),minimum=float(a.min()),maximum=float(a.max()))
def equal(a,b,path='root'):
    if isinstance(a,dict):
        assert set(a)==set(b),(path,'keys')
        for k in a:equal(a[k],b[k],path+'.'+k)
    elif isinstance(a,list):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+f'[{i}]')
    elif isinstance(a,(float,int)) and not isinstance(a,bool):
        assert b is not None and math.isclose(a,b,rel_tol=2e-12,abs_tol=2e-12),(path,a,b)
    else:assert a==b,(path,a,b)
groups=[]
for f in range(3):
    for end in ends:
        for role in roles:
            for win,(first,last) in windows.items():
                subset=[r for r in all_rows if r['fold']==f and r['endpoint']==end and r['role']==role and first<=r['step']<=last]
                assert len(subset)==last-first+1
                group=dict(fold=f,endpoint=end,role=role,window=win,first_step=first,last_step=last,rows=len(subset),
                           statistics={k:distribution([r[k] for r in subset]) for k in metric_names},
                           negative_fused_other_cosines=sum(r['fused_other_cosine'] is not None and r['fused_other_cosine']<0 for r in subset),
                           negative_fused_full_cosines=sum(r['fused_full_cosine'] is not None and r['fused_full_cosine']<0 for r in subset))
                groups.append(group)
equal(groups,analysis['statistics'])
csv_raw=list(csv.DictReader((root/'analysis/all_role_steps.csv').open(newline='')))
assert len(csv_raw)==len(all_rows)
for expected,actual in zip(all_rows,csv_raw):
    assert set(expected)==set(actual)
    for k,v in expected.items():
        if v is None:assert actual[k]==''
        elif isinstance(v,str):assert actual[k]==v
        else:equal(v,float(actual[k]),k)
csv_groups=list(csv.DictReader((root/'analysis/all_group_statistics.csv').open(newline='')))
assert len(csv_groups)==72*14
cursor=0
for g in groups:
    for metric in metric_names:
        row=csv_groups[cursor];cursor+=1
        for k in ('fold','endpoint','role','window'):assert row[k]==str(g[k])
        assert row['metric']==metric
        for k,v in g['statistics'][metric].items():
            if v is None:assert row[k]==''
            else:equal(v,float(row[k]),k)
report=(root/'analysis/REPORT.md').read_text()
report_rows=[line for line in report.splitlines() if line.startswith('| ') and line.split('|')[1].strip().isdigit()]
active_groups=[g for g in groups if g['window']=='active'];assert len(report_rows)==len(active_groups)==18
for line,g in zip(report_rows,active_groups):
    cells=[x.strip() for x in line.split('|')[1:-1]]
    assert cells[:3]==[str(g['fold']),g['endpoint'],g['role']]
    late=next(x for x in groups if x['window']=='last65' and all(x[k]==g[k] for k in ('fold','endpoint','role')))
    expected=[g['statistics']['fused_other_ratio']['median'],g['statistics']['fused_other_cosine']['median'],g['negative_fused_other_cosines'],late['statistics']['fused_other_ratio']['median']]
    for a,b in zip(expected,cells[3:]):
        if a is None:assert b=='None'
        else:equal(a,float(b),'report')
source_log=Path(str(root)+'_source.log')
events=[json.loads(line) for line in source_log.read_text().splitlines() if line.startswith('{')]
epochs=[e for e in events if e.get('event')=='objective_gradient_epoch']
assert len(epochs)==120
for index in range(6):
    segment=epochs[index*20:(index+1)*20]
    assert [r['epoch'] for r in segment]==list(range(1,21))
    assert [r['batches'] for r in segment]==[13*i for i in range(1,21)]
    assert all(r['endpoint']==ends[index%2] for r in segment)
for path in root.rglob('*'):
    if path.is_file():bind(path)
for path in root.parent.glob(root.name+'_*'):
    if path.is_file():bind(path)
print(json.dumps(dict(status='PASS_ALL_SOURCE_SCALAR_ROWS_AND_ANALYSIS',steps=steps,role_rows=len(all_rows),groups=len(groups),
    group_metric_rows=cursor,source_epoch_events=len(epochs),distance_values=values_count,conditions=conditions,
    max_loss_absolute_error=max_loss_error,max_norm_identity_error=max_norm_identity_error,max_denominator_error=max_denominator_error,
    max_current_decomposition_ratio=max_current_decomposition,max_full_decomposition_ratio=max_full_decomposition,max_single_group_direct_ratio=max_direct_error,
    max_repeat_relative=max_repeat_relative,current_distance_symmetry_max_error=max_symmetry_error,distance_range=[distance_min,distance_max],
    total_current_record_forwards=sum(c['current_record_forwards'] for c in costs),total_extra_record_forwards=sum(c['extra_record_forwards'] for c in costs),
    all_statistics=groups,pipelines=pipelines,files=list(inputs.values()),
    new_image_reads=0,new_model_forwards=0,new_parameter_backwards=0,new_optimizer_updates=0,
    deterministic_scope='Exact source row bindings, full saved-distance objectives, loss ledgers, scalar vector identities, all aggregation and report cells.',
    runtime_witness_only=['Source image to feature/distance generation','Per-step encoder parameter gradients','Zero upstream history-group skip decision','RNG and buffer/model invariants'],
    unavailable_reconstruction='The 1560 full-source parameter gradient vectors were not persisted. No new model replay performed.')))
