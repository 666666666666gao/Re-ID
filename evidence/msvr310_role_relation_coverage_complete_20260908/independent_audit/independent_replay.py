"""Auditor-owned NumPy/stdlib replay; never imports project implementation.

Executed from stdin remotely. Reads sealed arrays, labels, receipts, and saved
results; all outputs go to stdout for local-only capture. No model libraries.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import resource
import sys
import time
import numpy as np

PROJECT=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
SOURCE=Path('/root/autodl-tmp/trifusion-v2/artifacts/msvr310_source_relations_v1_seed42_4e57e54')
RESULT=Path('/root/autodl-tmp/trifusion-v2/artifacts/msvr310_role_relation_coverage_v1_seed42_c46be4e')
ROLES=['cnn','transformer','mamba']
OUTPUTS=['fused']+ROLES
STATES=['initial','control_final','style_final']
VIEWS=['clean','augmented','coupled_style']
MODES=['identity_exclude_record','cross_scene']
WIDTHS={'fused':7680,'cnn':4608,'transformer':4608,'mamba':4608}
opened=Counter()
def readonly(event,args):
    if event=='open':
        path,mode,flags=args
        assert not flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND), (path,mode,flags)
        opened[str(path)]+=1
sys.addaudithook(readonly)

def emit(kind,**row):
    print(json.dumps(dict(kind=kind,**row),sort_keys=True,allow_nan=False),flush=True)

hashes={}
def hashed(path):
    digest=hashlib.sha256()
    size=0
    with path.open('rb') as stream:
        while block:=stream.read(8*1024*1024):
            size+=len(block)
            digest.update(block)
    actual=dict(bytes=size,sha256=digest.hexdigest())
    hashes[str(path)]=actual
    return actual

def read_json(path):
    hashed(path)
    return json.loads(path.read_bytes())

started=time.perf_counter()
contract_path=PROJECT/'configs/MSVR310/Role-relation-coverage-v1.json'
spec=read_json(contract_path)
assert spec['schema']=='msvr310-role-relation-coverage-v1' and spec['seed']==42
assert spec['model_forwards']==spec['optimizer_updates']==0
assert spec['outputs']==OUTPUTS and spec['margin']==0.3 and spec['source_root']==str(SOURCE)
for path,expected in spec['input_sha256'].items():
    assert hashed(Path(path))['sha256']==expected,path
protocol=read_json(Path(spec['protocol']))
source_summary=read_json(SOURCE/'summary.json')
source_cpu=read_json(SOURCE/'cpu_verification.json')
source_pipeline=read_json(SOURCE/'pipeline.json')
original=read_json(RESULT/'analysis/summary.json')
pipeline=read_json(RESULT/'pipeline.json')
assert source_summary['status']=='COMPLETE_FROZEN_SOURCE_EXTRACTION'
assert source_cpu['status']=='PASS_COMPLETE_SOURCE_RELATION_CENSUS'
assert source_pipeline['status']=='COMPLETE_VERIFIED_SOURCE_DIAGNOSTIC'
assert source_cpu['extraction_summary_sha256']==hashes[str(SOURCE/'summary.json')]['sha256']
assert source_pipeline['cpu_verification_sha256']==hashes[str(SOURCE/'cpu_verification.json')]['sha256']
assert all(stage['exit_code']==0 for stage in source_pipeline['stages'])
assert pipeline['status']=='COMPLETE_UNAUDITED_SOURCE_REANALYSIS' and pipeline['exit_code']==0
assert original['status']=='COMPLETE_SOURCE_ROLE_RELATION_COVERAGE'
assert original['contract_sha256']==pipeline['contract_sha256']==hashes[str(contract_path)]['sha256']
assert original['script_sha256']==hashes[str(PROJECT/'tools/analyze_msvr_role_relation_coverage.py')]['sha256']
for flag in ['model_forwards','optimizer_updates','heldout_image_reads','official_image_reads']:
    assert original[flag]==0
for pid in [pipeline['wrapper_pid'],pipeline['analysis_pid']]:
    assert not Path('/proc',str(pid)).exists()
rows_path=RESULT/'analysis/all_query_relations.jsonl'
assert hashed(rows_path)==original['query_rows']
analysis_log=[json.loads(line) for line in (RESULT/'analysis.log').read_text().splitlines()]
hashed(RESULT/'analysis.log')
expected_conditions=[(fold,state,view) for fold in range(3) for state in STATES for view in VIEWS]
assert [(c['fold'],c['state'],c['view']) for c in source_summary['conditions']]==expected_conditions
assert [(c['completed_conditions'],c['condition']) for c in analysis_log]==[
    (2*(i+1),f'fold_{f}_{s}_{v}') for i,(f,s,v) in enumerate(expected_conditions)]

# Independent label provenance and split reconstruction, using text labels only.
label_path=PROJECT/'evidence/vehicle_query_protocol_labels_20260905.json'
label_evidence=read_json(label_path)
assert hashes[str(label_path)]['sha256']==protocol['label_evidence_sha256']
label_manifest=next(x for x in label_evidence['datasets'] if x['dataset']=='MSVR310')['record_manifest']['bounding_box_train']
records=protocol['records']
assert len(label_manifest)==len(records)==1032
assert len({r['identity'] for r in records})==155
for index,(record,label) in enumerate(zip(records,label_manifest)):
    name=PurePosixPath(label['path']).name
    assert record['index']==index
    assert (record['identity'],record['scene'],record['camera'])==(int(name[:4]),int(name[6:9]),int(name[11]))
    assert all(record[key]==label[key] for key in ['identity','scene','camera'])
    assert record['paths']==[f'bounding_box_train/{record["identity"]:04d}/{m}/{name}' for m in ['vis','ni','th']]
identities=sorted({r['identity'] for r in records})
identity_scenes={identity:{r['scene'] for r in records if r['identity']==identity} for identity in identities}
split_groups=[[i for i in identities if len(identity_scenes[i])>1],
              [i for i in identities if len(identity_scenes[i])==1]]
assert list(map(len,split_groups))==[60,95]
fold_meta=[]
for f in range(3):
    fold=protocol['folds'][f]
    heldout=sorted(i for group in split_groups for j,i in enumerate(group) if j%3==f)
    source_ids=[i for i in identities if i not in heldout]
    indices=[i for i,r in enumerate(records) if r['identity'] in source_ids]
    assert heldout==fold['heldout_ids'] and source_ids==fold['source_ids']
    assert indices==fold['source_record_indices']
    assert fold['source_label_map']=={str(i):j for j,i in enumerate(source_ids)}
    assert len(indices)==[672,683,709][f]
    fold_meta.append(dict(fold=f,source_records=len(indices),source_identities=len(source_ids),
                         source_scenes=sorted({records[i]['scene'] for i in indices}),heldout_identities=len(heldout),
                         cross_scene_eligible=sum(len(identity_scenes[records[i]['identity']])>1 for i in indices)))

# All original condition receipts, view inputs and four selected arrays.
old_spec=read_json(PROJECT/'configs/MSVR310/Source-relation-census-v1.json')
old_q1=read_json(Path(old_spec['q1_summary']))
assert hashes[str(Path(old_spec['q1_summary']))]['sha256']==old_spec['fixed_file_sha256'][old_spec['q1_summary']]
assert source_summary['contract_sha256']==hashes[str(PROJECT/'configs/MSVR310/Source-relation-census-v1.json')]['sha256']
assert old_q1['status']=='Q1_FAIL'
receipts={}
normalized={}
input_rows={}
selected_input_hashes={}
array_stats=[]
for fold,state,view in expected_conditions:
    name=f'fold_{fold}_{state}_{view}'
    condition=next(c for c in source_summary['conditions'] if c['directory']==name)
    receipt_path=SOURCE/name/'receipt.json'
    receipt=read_json(receipt_path)
    assert receipt=={k:v for k,v in condition.items() if k!='directory'}
    receipts[name]=receipt
    selected_input_hashes[str(receipt_path)]=hashes[str(receipt_path)]
    f=protocol['folds'][fold]
    assert receipt['record_indices']==f['source_record_indices']
    assert receipt['source_ids']==f['source_ids'] and receipt['heldout_ids']==f['heldout_ids']
    assert not set(receipt['source_ids']) & set(receipt['heldout_ids'])
    assert all(receipt[k] for k in ['model_state_unchanged','gradients_absent','raw_pixels_match_other_states','augmented_pixels_match_style'])
    endpoints=old_q1['folds'][fold]['endpoints']
    assert endpoints['control']['initialization']==endpoints['source_style']['initialization']
    expected_state=(endpoints['control']['initialization']['initial_state_sha256'] if state=='initial' else
                    endpoints['control' if state=='control_final' else 'source_style']['training']['final_state_sha256'])
    assert receipt['model_state_sha256']==expected_state
    inp_path=SOURCE/name/'inputs.jsonl'
    assert hashed(inp_path)==receipt['files']['inputs.jsonl']
    inputs=[json.loads(line) for line in inp_path.read_text().splitlines()]
    input_rows[name]=inputs
    assert [i for batch in inputs for i in batch['record_indices']]==f['source_record_indices']
    assert [b['batch'] for b in inputs]==list(range(math.ceil(receipt['records']/64)))
    for batch in inputs:
        assert batch['statistics']['style_active']==int(view=='coupled_style')
        assert (batch['style_plan'] is not None)==(view=='coupled_style')
    assert sum(b['statistics']['additional_visual_passes'] for b in inputs)==receipt['additional_visual_passes']
    for output in OUTPUTS:
        path=SOURCE/name/(output+'.npy')
        actual=hashed(path)
        assert actual==receipt['files'][output+'.npy'],str(path)
        selected_input_hashes[str(path)]=actual
        values=np.load(path,allow_pickle=False)
        assert values.shape==(receipt['records'],WIDTHS[output]) and values.dtype==np.float32
        assert np.isfinite(values).all()
        values=values.astype(np.float64)
        lengths=np.sqrt(np.add.reduce(values*values,axis=1))
        assert np.all(lengths>0)
        array_stats.append(dict(condition=name,output=output,shape=list(values.shape),
                                original_dtype='float32',min_norm=float(lengths.min()),max_norm=float(lengths.max()),**actual))
        normalized[name,output]=values/lengths[:,None]
assert selected_input_hashes==original['inputs']
assert len(selected_input_hashes)==135 and len(array_stats)==108
for fold,state,view in expected_conditions:
    name=f'fold_{fold}_{state}_{view}'
    clean=f'fold_{fold}_{state}_clean'
    assert receipts[name]['record_indices']==receipts[clean]['record_indices']
    match_view='augmented' if view=='coupled_style' else view
    reference=f'fold_{fold}_initial_{match_view}'
    assert [b['pixel_sha256'] for b in input_rows[name]]==[b['pixel_sha256'] for b in input_rows[reference]]
emit('bindings',status='PASS',folds=fold_meta,selected_receipts=27,selected_arrays=108,
     total_array_bytes=sum(x['bytes'] for x in array_stats),array_stats=array_stats,
     old_source_runtime=dict(source_record_forwards=source_summary['source_record_forwards'],
                             forward_batches=source_summary['forward_batches'],
                             additional_visual_passes=source_summary['additional_visual_passes']),
     original_result_status=pipeline['status'],original_exit=pipeline['exit_code'])

# Recursive compare checks structure, every scalar, exact integers and booleans.
comparison=Counter()
differences=[]
maximum_float_error=0.0
def compare(expected,actual,path):
    global maximum_float_error
    if type(expected)!=type(actual):
        differences.append(dict(path=path,expected=expected,actual=actual,reason='type'))
        return
    if isinstance(expected,dict):
        comparison['dicts']+=1
        if expected.keys()!=actual.keys():
            differences.append(dict(path=path,expected=sorted(expected),actual=sorted(actual),reason='keys'))
        for key in expected.keys() & actual.keys():
            compare(expected[key],actual[key],path+'/'+key)
    elif isinstance(expected,list):
        comparison['lists']+=1
        if len(expected)!=len(actual):
            differences.append(dict(path=path,expected=len(expected),actual=len(actual),reason='list_length'))
        for i,(a,b) in enumerate(zip(expected,actual)):
            compare(a,b,path+'/'+str(i))
    elif isinstance(expected,float):
        comparison['float_values']+=1
        error=abs(expected-actual)
        maximum_float_error=max(maximum_float_error,error)
        comparison['bitwise_equal_float_values']+=int(expected==actual)
        if error>1e-12 or not math.isfinite(actual):
            differences.append(dict(path=path,expected=expected,actual=actual,reason='float'))
    else:
        comparison['exact_values']+=1
        if expected!=actual:
            differences.append(dict(path=path,expected=expected,actual=actual,reason='exact'))

def tally(rows):
    c=Counter()
    for r in rows:
        c['all_query_memberships']+=1
        if not r['eligible']:
            continue
        c['eligible_query_memberships']+=1
        for key in ['extra_negative_records','extra_positive_records','extra_negative_fused_hinge_active',
                    'extra_negative_fused_nonpositive_margin']:
            c[key]+=len(r[key])
        c['role_union_negative_records']+=len(r['role_negative_union'])
        c['role_union_positive_records']+=len(r['role_positive_union'])
        c['anchors_with_extra_hinge_active_negative']+=bool(r['extra_negative_fused_hinge_active'])
        c['anchors_with_extra_wrong_order_negative']+=bool(r['extra_negative_fused_nonpositive_margin'])
        delta=r['role_subset_fused_hinge']-r['full_fused_hinge']
        c['role_subset_lower_hinge']+=delta < -1e-12
        c['role_subset_equal_hinge']+=abs(delta)<=1e-12
        c['role_subset_higher_hinge']+=delta > 1e-12
        c['full_fused_hinge_active']+=r['full_fused_hinge']>0
        c['contains_both_fused_extrema']+=r['contains_fused_negative'] and r['contains_fused_positive']
        for role in ROLES:
            c[role+'_unique_negative']+=r['role_unique_negative'][role]
    return dict(c)

original_stream=rows_path.open(encoding='utf-8')
conditions=[]
all_stats=Counter()
direct_error=0.0
direct_checks=0
seen=set()
mask_hash=hashlib.sha256()
for fold,state,view in expected_conditions:
    name=f'fold_{fold}_{state}_{view}'
    gallery_name=f'fold_{fold}_{state}_clean'
    indices=protocol['folds'][fold]['source_record_indices']
    ids=[records[i]['identity'] for i in indices]
    scenes=[records[i]['scene'] for i in indices]
    size=len(indices)
    distance={}
    for output in OUTPUTS:
        product=np.einsum('ik,jk->ij',normalized[name,output],normalized[gallery_name,output],optimize=True)
        distance[output]=np.sqrt(np.clip(2.0-2.0*product,0.0,None))
    for mode in MODES:
        rows=[]
        ineligible_positions={q for q in range(size) if not any(
            j!=q and ids[j]==ids[q] and (mode!='cross_scene' or scenes[j]!=scenes[q]) for j in range(size))}
        for q in range(size):
            positive=[j for j in range(size) if j!=q and ids[j]==ids[q] and (mode!='cross_scene' or scenes[j]!=scenes[q])]
            negative=[j for j in range(size) if ids[j]!=ids[q]]
            assert negative
            mask_hash.update(json.dumps([name,mode,q,positive,negative],separators=(',',':')).encode())
            row=dict(condition=name,protocol=mode,query_position=q,record_index=indices[q],identity=ids[q],
                     scene=scenes[q],positive_count=len(positive),negative_count=len(negative),eligible=bool(positive))
            all_stats['ineligible_gallery_negative_memberships']+=len(set(negative) & ineligible_positions)
            all_stats['positive_candidate_memberships']+=len(positive)
            all_stats['negative_candidate_memberships']+=len(negative)
            if positive:
                # Python min/max preserve earliest source position on exact ties.
                far={o:max(positive,key=lambda j:float(distance[o][q,j])) for o in OUTPUTS}
                near={o:min(negative,key=lambda j:float(distance[o][q,j])) for o in OUTPUTS}
                union_p=[j for j in range(size) if j in {far[o] for o in ROLES}]
                union_n=[j for j in range(size) if j in {near[o] for o in ROLES}]
                fused=distance['fused'][q]
                full=max(0.0,float(fused[far['fused']]-fused[near['fused']])+0.3)
                subset=max(0.0,max(float(fused[j]) for j in union_p)-min(float(fused[j]) for j in union_n)+0.3)
                assert subset<=full+1e-12
                extras_n=[j for j in union_n if j!=near['fused']]
                extras_p=[j for j in union_p if j!=far['fused']]
                active=[j for j in extras_n if float(fused[far['fused']]-fused[j])+0.3>0.0]
                nonpositive=[j for j in extras_n if float(fused[j]-fused[far['fused']])<=0.0]
                row.update(hardest_positive={o:indices[far[o]] for o in OUTPUTS},
                    nearest_negative={o:indices[near[o]] for o in OUTPUTS},
                    role_positive_union=[indices[j] for j in union_p],role_negative_union=[indices[j] for j in union_n],
                    full_fused_hinge=full,role_subset_fused_hinge=subset,
                    extra_negative_records=[indices[j] for j in extras_n],extra_positive_records=[indices[j] for j in extras_p],
                    extra_negative_fused_hinge_active=[indices[j] for j in active],
                    extra_negative_fused_nonpositive_margin=[indices[j] for j in nonpositive],
                    role_unique_negative={o:sum(near[o]==near[r] for r in ROLES)==1 for o in ROLES},
                    negative_union_identities=len({ids[j] for j in union_n}),negative_union_scenes=len({scenes[j] for j in union_n}),
                    contains_fused_negative=near['fused'] in union_n,contains_fused_positive=far['fused'] in union_p,
                    fused_positive_extremum_ties=sum(float(fused[j])==float(fused[far['fused']]) for j in positive),
                    fused_negative_extremum_ties=sum(float(fused[j])==float(fused[near['fused']]) for j in negative))
                all_stats['extra_negative_tied_fused_minimum']+=sum(fused[j]==fused[near['fused']] for j in extras_n)
                all_stats['extra_positive_tied_fused_maximum']+=sum(fused[j]==fused[far['fused']] for j in extras_p)
                all_stats['fused_positive_tied_queries']+=row['fused_positive_extremum_ties']>1
                all_stats['fused_negative_tied_queries']+=row['fused_negative_extremum_ties']>1
                witness={}
                for output in OUTPUTS:
                    dp=distance[output][q]
                    ties_p=sum(dp[j]==dp[far[output]] for j in positive)
                    ties_n=sum(dp[j]==dp[near[output]] for j in negative)
                    all_stats[output+'_positive_tied_queries']+=ties_p>1
                    all_stats[output+'_negative_tied_queries']+=ties_n>1
                    selected=sorted(set([far[output],near[output]]+(union_p+union_n if output=='fused' else [])))
                    delta=normalized[gallery_name,output][selected]-normalized[name,output][q]
                    direct=np.sqrt(np.add.reduce(delta*delta,axis=1))
                    error=float(np.abs(direct-dp[selected]).max())
                    direct_error=max(direct_error,error)
                    direct_checks+=len(selected)
                    witness[output]=dict(positive_distance=float(dp[far[output]]),negative_distance=float(dp[near[output]]),
                        positive_ties=int(ties_p),negative_ties=int(ties_n),direct_distance_max_error=error)
                emit('extremum_witness',condition=name,protocol=mode,query_position=q,witness=witness)
            row_key=(name,mode,q)
            assert row_key not in seen
            seen.add(row_key)
            actual=json.loads(next(original_stream))
            compare(row,actual,f'row/{len(seen)}')
            emit('query',row=row)
            rows.append(row)
        result=dict(condition=name,fold=fold,state=state,view=view,protocol=mode,counts=tally(rows))
        conditions.append(result)
        compare(result,original['conditions'][len(conditions)-1],f'condition/{len(conditions)}')
        emit('condition',row=result)
assert original_stream.readline()==''
original_stream.close()
assert len(conditions)==54 and len(seen)==37152 and len(original['conditions'])==54
assert direct_error<1e-12

# Rehash every input after the replay, detecting changes during this audit.
before=dict(hashes)
for path,prior in before.items():
    assert hashed(Path(path))==prior,path
for path,value in hashes.items():
    emit('input_hash',path=path,**value)
emit('verification',status='PASS' if not differences else 'FAIL',
     query_rows=len(seen),condition_results=len(conditions),comparison=dict(comparison),
     differences=differences,maximum_saved_float_error=maximum_float_error,
     direct_distance_witness_checks=direct_checks,maximum_direct_distance_error=direct_error,
     masks_sha256=mask_hash.hexdigest(),additional_statistics={k:int(v) for k,v in all_stats.items()},
     all_input_hashes_unchanged=True,input_hashes=len(hashes),read_only_open_paths=dict(opened),
     torch_imported='torch' in sys.modules,project_modules_imported=sorted(k for k in sys.modules if k.startswith(('tools.','trifusion.'))),
     dont_write_bytecode=sys.dont_write_bytecode,cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
     thread_environment={k:os.getenv(k) for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']},
     elapsed_seconds=time.perf_counter()-started,maximum_resident_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
     ended_at=datetime.now(timezone.utc).isoformat())
assert not differences
