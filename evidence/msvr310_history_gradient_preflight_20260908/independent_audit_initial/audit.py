"""Independent standard-library audit. Reads project/evidence; writes only beside this script."""
from pathlib import Path
from collections import Counter, OrderedDict
from datetime import datetime
import ast
import csv
import hashlib
import json
import math
import re
import statistics
import sys

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
E = ROOT/'evidence/msvr310_history_gradient_preflight_20260908'
P = ROOT/'evidence/msvr310_fresh_coordinate_complete_q1_20260908'
OUT = Path(__file__).parent
failures = []
counts = Counter()
hashes = {}

def check(label, value):
    counts[label.split(':')[0]] += 1
    if not value:
        failures.append(label)

def close(label, actual, expected, rel=1e-10, abs_tol=1e-12):
    check(label, math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_tol))

def digest(path):
    result = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes[str(path)] = result
    return result

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def tree_equal(label, actual, expected):
    if isinstance(expected, dict):
        check(label+':keys', actual.keys() == expected.keys())
        for k in expected:
            tree_equal(label+'/'+k, actual[k], expected[k])
    elif isinstance(expected, list):
        check(label+':len', len(actual) == len(expected))
        for i, (a,b) in enumerate(zip(actual, expected)):
            tree_equal(label+'/'+str(i), a,b)
    elif isinstance(expected, float):
        close(label, actual, expected)
    else:
        check(label, actual == expected)

config_path=ROOT/'configs/MSVR310/TriFusion-history-candidate-gradient-v1.json'
config=read(config_path)
config_hash=digest(config_path)
bindings=[]
remote=[]
visited=set()
fixed_prefix='/root/autodl-tmp/trifusion-v2/artifacts/msvr310_fresh_coordinate_v1_seed42_b4501fa/'

def bound_file(parent,name,expected):
    if name.startswith('/root/'):
        local=P/name[len(fixed_prefix):] if name.startswith(fixed_prefix) else None
        if local is not None and local.is_file():
            check('remote_mirrored_hash:'+name,digest(local)==expected)
            bindings.append(dict(parent=parent,path=name,local=str(local),status='EXACT'))
        else:
            remote.append(dict(parent=parent,path=name,expected=expected,status='UNAVAILABLE_LOCALLY'))
        return
    path=ROOT/name
    check('bound_exists:'+name,path.is_file())
    if not path.is_file():return
    data=path.read_bytes();actual=digest(path)
    status='EXACT' if actual==expected else 'LF_ONLY_MATCH' if hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest()==expected else 'MISMATCH'
    bindings.append(dict(parent=parent,path=name,expected=expected,actual=actual,status=status))
    if status=='MISMATCH':check('bound_hash:'+name,False)
    if name.startswith('configs/') and path.suffix=='.json':walk_config(name)

def walk_config(name):
    if name in visited:return
    visited.add(name)
    obj=read(ROOT/name)
    def scan(v):
        if not isinstance(v,dict):return
        for key,child in v.items():
            if key in ('project_file_sha256','project_source_file_sha256','project_files','fixed_file_sha256','fixed_files'):
                for filename,sha in child.items():bound_file(name,filename,sha)
            elif isinstance(child,dict):scan(child)
            elif isinstance(child,str) and (key.endswith('_sha256') or key.endswith('_SHA256')):
                stem=key[:-7]
                if stem in v and isinstance(v[stem],str) and ('/' in v[stem] or '\\' in v[stem]):bound_file(name,v[stem],child)
        if 'PATH' in v and 'SHA256' in v:bound_file(name,v['PATH'],v['SHA256'])
    scan(obj)

walk_config(str(config_path.relative_to(ROOT)).replace('\\','/'))
manifest=read(E/'intake_manifest.json')
check('raw_manifest:count',len(manifest['files'])==25)
check('raw_manifest:unique',len({x['path'] for x in manifest['files']})==25)
for record in manifest['files']:
    path=E/record['path'];data=path.read_bytes();data.decode('utf-8')
    check('raw_size:'+record['path'],len(data)==record['bytes'])
    check('raw_hash:'+record['path'],digest(path)==record['sha256'])
    if path.suffix=='.json':json.loads(data)
    if path.suffix=='.jsonl':
        for line in data.decode('utf-8').splitlines():json.loads(line)
raw_bytes=sum(x['bytes'] for x in manifest['files'])
check('raw_manifest:bytes',raw_bytes==805652)
summary=read(E/'preflight/summary.json');cpu=read(E/'preflight/cpu_verification.json')
reaggregation=read(E/'reaggregation/complete_text_reaggregation.json')
protocol=read(ROOT/config['protocol'])
records=protocol['records']
original_labels_path=ROOT/'evidence/vehicle_query_protocol_labels_20260905.json'
check('protocol:original_labels_hash',digest(original_labels_path)==protocol['label_evidence_sha256'])
check('protocol:builder_hash',digest(ROOT/'tools/build_msvr310_train_oof_protocol.py')==protocol['builder_sha256'])
dataset=next(x for x in read(original_labels_path)['datasets'] if x['dataset']=='MSVR310')
original_train=dataset['record_manifest']['bounding_box_train']
check('protocol:original_training_count',len(original_train)==len(records)==1032)
for old,now in zip(original_train,records):
    check('protocol:original_training_labels',all(old[key]==now[key] for key in ('identity','camera','scene')) and 'bounding_box_train/'+old['path']==now['paths'][0])
q1=read(P/'q1/summary.json');q1cpu=read(P/'q1_cpu.json')
previous_manifest=read(P/'intake_manifest.json')
for record in previous_manifest['files']:
    path=P/record['path']
    check('previous_raw_size:'+record['path'],path.stat().st_size==record['bytes'])
    check('previous_raw_hash:'+record['path'],digest(path)==record['sha256'])
check('q1:config_fixed_summary',digest(P/'q1/summary.json')==config['fixed_file_sha256'][config['q1_summary']])
check('q1:config_fixed_cpu',digest(P/'q1_cpu.json')==config['fixed_file_sha256'][config['q1_cpu']])
check('q1:cpu_binding',q1cpu['summary_sha256']==digest(P/'q1/summary.json'))
check('q1:status',q1['status']=='Q1_FAIL')
check('q1:cpu_status',q1cpu['status']=='PASS_COMPLETE_FRESH_COORDINATE_Q1')
for name,endpoint in q1['comparison']['endpoints'].items():
    metrics=endpoint['metrics'];baseline_map=metrics['baseline_only']['mAP']
    expected_gates=dict(fused_gain_at_least_1pp=metrics['fused']['mAP']-baseline_map>=1.,all_fold_fused_gains_nonnegative=all(g>=0 for g in endpoint['fold_fused_gains_pp']),all_full_branches_not_below_signal=all(metrics[k]['mAP']>=baseline_map for k in ('cnn','transformer','mamba')),identity_bootstrap_lower_positive=endpoint['identity_bootstrap']['lower_bound_pp']>0,fused_strictly_best=all(metrics['fused']['mAP']>metrics[k]['mAP'] for k in ('baseline_only','cnn','transformer','mamba')))
    check('q1:gate_logic',expected_gates==endpoint['scientific_checks'] and sum(expected_gates.values())==0)
check('summary:status',summary['status']=='PASS_COMPLETE_FIXED_STATE_PROBE' and summary['mode']=='preflight')
check('summary:config',summary['config_sha256']==config_hash)
check('cpu:binding',cpu['summary_sha256']==digest(E/'preflight/summary.json'))
check('reaggregation:binding',reaggregation['summary_sha256']==digest(E/'preflight/summary.json'))
for obj in (summary,):
    check('summary:seed_and_counters',obj['seed']==42 and obj['optimizer_updates']==obj['official_image_reads']==obj['heldout_record_forwards']==0)

all_ids={r['identity'] for r in records}
membership={str(identity):sorted({r['scene'] for r in records if r['identity']==identity}) for identity in all_ids}
check('protocol:identity_membership',membership==protocol['identity_scene_membership'])
for i,r in enumerate(records):
    check('protocol:record_index',r['index']==i)
    check('protocol:modalities',len(r['paths'])==3 and [x.split('/')[2] for x in r['paths']]==['vis','ni','th'])
    for path in r['paths']:
        match=re.fullmatch(r'bounding_box_train/(\d+)/(vis|ni|th)/(\d+)_s(\d+)_v(\d+)_(\d+)\.jpg',path)
        check('protocol:filename_labels',bool(match) and int(match[1])==int(match[3])==r['identity'] and int(match[4])==r['scene'] and int(match[5])==r['camera'])
heldout_by_fold=[set() for _ in range(3)]
for eligible in (True,False):
    group=sorted(x for x in all_ids if (len(membership[str(x)])>=2)==eligible)
    for i,identity in enumerate(group):heldout_by_fold[i%3].add(identity)
protocol_scope=[]
for fold in protocol['folds']:
    fid=fold['fold'];source=set(fold['source_ids']);heldout=set(fold['heldout_ids'])
    check('protocol:round_robin',heldout==heldout_by_fold[fid])
    check('protocol:partition',source|heldout==all_ids and not source&heldout)
    check('protocol:source_records',set(fold['source_record_indices'])=={i for i,r in enumerate(records) if r['identity'] in source})
    check('protocol:gallery_records',set(fold['gallery_record_indices'])=={i for i,r in enumerate(records) if r['identity'] in heldout})
    check('protocol:source_label_map',fold['source_label_map']=={str(x):i for i,x in enumerate(sorted(source))})
    protocol_scope.append(dict(fold=fid,source_records=len(fold['source_record_indices']),source_identities=len(source),heldout_records=len(fold['gallery_record_indices']),heldout_identities=len(heldout)))

base_config=read(ROOT/'configs/MSVR310/TriFusion-source-style-paired-v1-r2.json')
source_meta=read(ROOT/base_config['SOURCE_METADATA']['PATH'])
weight=base_config['LOSS']['TRIPLET_FUSED']
csv_rows=[];states=[];scope=[];maximums=Counter();memory_sizes=[];all_step_rows=[]
for fold in summary['folds']:
    fid=fold['fold'];pf=protocol['folds'][fid]
    check('summary:states',set(fold['states'])==set(config['states']))
    matched=[]
    for state,receipt in fold['states'].items():
        name=f'fold_{fid}_{state}';directory=E/'preflight'/name
        saved_receipt=read(directory/'receipt.json')
        tree_equal('receipt:'+name,receipt,saved_receipt)
        check('receipt:status',receipt['status']=='PASS_FIXED_STATE_SOURCE_PROBE')
        check('receipt:state_hash',receipt['initial_state_sha256']==receipt['final_state_sha256'])
        check('receipt:no_updates',receipt['optimizer_updates']==receipt['heldout_record_forwards']==receipt['official_image_reads']==0)
        previous=q1['folds'][fid]['endpoints']['control' if state=='initial' else state]
        check('receipt:qualified_state',receipt['initial_state_sha256']==(previous['initialization']['initial_state_sha256'] if state=='initial' else previous['training']['final_state_sha256']))
        reference=[json.loads(x) for x in (P/'q1'/f'fold_{fid}_{"control" if state=="initial" else state}'/'memory_steps.jsonl').read_text(encoding='utf-8').splitlines()]
        rows=[json.loads(x) for x in (directory/'steps.jsonl').read_text(encoding='utf-8').splitlines()]
        check('receipt:step_hash',digest(directory/'steps.jsonl')==receipt['files']['steps.jsonl']['sha256'])
        check('receipt:step_bytes',(directory/'steps.jsonl').stat().st_size==receipt['files']['steps.jsonl']['bytes'])
        check('state:step_count',len(rows)==receipt['batches']==8)
        queue=OrderedDict();seen=set();seen_ids=set();seen_scenes=set();role_points=[];offset=0;history_count=0;nonzero_records=0;label_zero=0
        group_counts=[];full_group_counts=[]
        for zero,row in enumerate(rows):
            loc=f'{name}:{zero+1}'
            all_step_rows.append(row)
            current=row['record_indices'];seen.update(current);seen_ids.update(row['identities']);seen_scenes.update(row['scenes'])
            check('row:order',row['step']==zero+1 and row['epoch']==1)
            check('row:scope',row['optimizer_updates']==0 and row['historical_coordinate_mode']=='fixed_current_parameters' and row['age_semantics']=='batch_recency_not_parameter_updates')
            check('row:B64_P8K8',len(current)==64 and sorted(Counter(row['identities']).values())==[8]*8)
            check('row:source_only',set(current)<=set(pf['source_record_indices']) and not set(current)&set(pf['gallery_record_indices']))
            check('row:identities',row['identities']==[records[i]['identity'] for i in current])
            check('row:scenes',row['scenes']==[records[i]['scene'] for i in current])
            check('row:registered_batch',current==source_meta['folds'][fid]['batches'][zero]['record_indices'])
            check('row:previous_batch_and_pixels',current==reference[zero]['record_indices'] and row['pixel_sha256']==reference[zero]['pixel_sha256'])
            label_zero+=sum(pf['source_label_map'][str(x)]==0 for x in row['identities'])
            for i in [i for i,k in queue.items() if zero-k>8]:del queue[i]
            expected_memory=[dict(record_index=i,identity=records[i]['identity'],scene=records[i]['scene'],age=zero-k,stored_step=k) for i,k in queue.items() if i not in current]
            check('row:full_memory',row['memory']==expected_memory)
            m=len(expected_memory);memory_sizes.append(m)
            check('row:matrix_shape',row['distance_offset_bytes']==offset and row['distance_float_count']==64*(64+m))
            offset+=row['distance_float_count']*4
            stats=row['statistics'];pos=sum(a==b['identity'] for a in row['identities'] for b in expected_memory)
            cross=sum(a==b['identity'] and s!=b['scene'] for a,s in zip(row['identities'],row['scenes']) for b in expected_memory)
            check('row:pair_counts',stats['memory_records']==m and stats['memory_positive_pairs']==pos and stats['memory_negative_pairs']==64*m-pos and stats['memory_cross_scene_positive_pairs']==cross)
            check('row:max_age',stats['maximum_memory_age']==max((x['age'] for x in expected_memory),default=0))
            check('row:triplet_monotonic',stats['expanded_triplet']>=stats['current_triplet']-1e-12 and stats['expanded_wrong_order_anchors']>=stats['current_wrong_order_anchors'])
            for key in ('harder_positive_anchors','harder_negative_anchors','current_wrong_order_anchors','expanded_wrong_order_anchors','expanded_hinge_positive_anchors'):
                check('row:anchor_count',0<=stats[key]<=64)
            full_groups=sorted({r['stored_step'] for r in expected_memory});selected=row['candidate_vjp_groups']
            group_counts.append(len(selected));full_group_counts.append(len(full_groups))
            check('row:group_membership',selected==sorted(set(selected)) and set(selected)<=set(full_groups))
            check('row:extra_forwards',row['extra_role_record_forwards']==64*(len(selected)+(row['step']==receipt['candidate_gradient_chain_rule']['step'])))
            check('row:nonzero_records',0<=row['history_nonzero_gradient_records']<=m)
            check('row:finite',math.isfinite(row['loss']) and all(math.isfinite(x) for x in stats.values()))
            nonzero_records+=row['history_nonzero_gradient_records']
            if m:history_count+=1
            check('row:roles',set(row['roles'])==({'cnn','transformer','mamba'} if m else set()))
            for role,v in row['roles'].items():
                for metric,c in v.items():
                    a,b,d,cosa=c['first_norm'],c['second_norm'],c['difference_norm'],c['cosine']
                    check('metric:finite',all(math.isfinite(x) and x>=0 for x in (a,b,d)))
                    check('metric:cosine_definition',(cosa is None)==(a==0 or b==0))
                    if cosa is not None:
                        check('metric:cosine_range',math.isfinite(cosa) and -1-1e-12<=cosa<=1+1e-12)
                        error=abs(d*d-(a*a+b*b-2*a*b*cosa))/max(a*a+b*b,1e-30)
                        maximums['comparison_norm_identity']=max(maximums['comparison_norm_identity'],error)
                        check('metric:comparison_identity',error<1e-5)
                uv=v['current_vs_history'];both=v['current_vs_both'];total=v['total_vs_both'];u=uv['first_norm'];h=uv['second_norm'];dot=u*h*uv['cosine'] if uv['cosine'] is not None else 0
                close('metric:current_norm',u,both['first_norm'])
                close('metric:current_repeat_norm',u,v['current_repeat_noise']['first_norm'])
                close('metric:history_repeat_norm',h,v['history_repeat_noise']['first_norm'])
                closure=abs(both['second_norm']**2-(u*u+h*h+2*dot))/max(u*u+h*h,1e-30)
                maximums['gradient_norm_identity']=max(maximums['gradient_norm_identity'],closure)
                check('metric:sum_identity',closure<1e-5)
                close('metric:difference_is_history',both['difference_norm'],h,rel=1e-5,abs_tol=1e-8)
                close('metric:total_difference_is_weighted_history',total['difference_norm'],weight*h,rel=1e-5,abs_tol=1e-8)
                if u>0 and both['second_norm']>0:close('metric:sum_cosine',both['cosine'],(u*u+dot)/(u*both['second_norm']),rel=1e-5,abs_tol=1e-8)
                point=dict(fold=fid,state=state,step=row['step'],epoch=1,role=role,current_norm=u,history_norm=h,history_to_current_ratio=h/u if u>0 else None,current_history_cosine=uv['cosine'],current_both_cosine=both['cosine'],task_total_both_cosine=total['cosine'],history_noise=v['history_repeat_noise']['difference_norm'],history_above_repeat_noise=h>v['history_repeat_noise']['difference_norm'],history_nonzero_record_exposures=row['history_nonzero_gradient_records'])
                csv_rows.append(point);role_points.append(point)
            if zero>=2:
                for i in current:queue.pop(i,None);queue[i]=zero
                while len(queue)>512:queue.popitem(last=False)
        check('state:matrix_bytes',offset==receipt['files']['distances.f32']['bytes'])
        check('state:seen',sorted(seen)==receipt['observed_source_records'])
        check('state:history_count',history_count==receipt['history_batches']==5)
        check('state:extra',sum(r['extra_role_record_forwards'] for r in rows)+64==receipt['extra_role_record_forwards']==1088)
        proof=receipt['candidate_gradient_chain_rule'];agreement=proof['agreement']
        close('proof:ratio',proof['relative_l2_error'],agreement['difference_norm']/max(agreement['first_norm'],agreement['second_norm']))
        check('proof:tolerance',proof['tolerance']==.005 and proof['relative_l2_error']<=proof['tolerance'])
        check('proof:scope',proof['step']==4 and proof['history_groups']==1 and proof['direct_graph_record_forwards']==64)
        expected_cpu={};aggregated={}
        for role in ('cnn','transformer','mamba'):
            points=[p for p in role_points if p['role']==role]
            expected_cpu[role]=dict(rows=len(points),history_nonzero=sum(p['history_norm']>0 for p in points),above_repeated_noise=sum(p['history_above_repeat_noise'] for p in points),current_history_cosines=[p['current_history_cosine'] for p in points],total_change_cosines=[p['task_total_both_cosine'] for p in points],history_parameter_norm=[p['history_norm'] for p in points],current_parameter_norm=[p['current_norm'] for p in points])
            agg={}
            for field in ('history_to_current_ratio','current_history_cosine','current_both_cosine','task_total_both_cosine'):
                values=sorted(p[field] for p in points if p[field] is not None)
                def q(t):
                    pos=(len(values)-1)*t;lo=math.floor(pos);hi=math.ceil(pos)
                    return values[lo]+(values[hi]-values[lo])*(pos-lo)
                agg[field]=dict(defined=len(values),undefined=len(points)-len(values),mean=statistics.mean(values) if values else None,minimum=min(values) if values else None,maximum=max(values) if values else None,negative=sum(x<0 for x in values),quantiles=[q(t) for t in (0,.25,.5,.75,1)] if values else None)
            agg['history_above_repeat_noise']=sum(p['history_above_repeat_noise'] for p in points);aggregated[role]=agg
        cp=next(s for s in cpu['states'] if s['fold']==fid and s['state']==state)
        expected_cp=dict(fold=fid,state=state,batches=8,history_nonzero_record_exposures=nonzero_records,roles=expected_cpu,extra_role_record_forwards=receipt['extra_role_record_forwards'],peak_allocated_mib=receipt['peak_allocated_mib'])
        tree_equal('cpu_state:'+name,cp,expected_cp)
        expected_agg=dict(fold=fid,state=state,batches=8,roles=aggregated,extra_role_record_forwards=receipt['extra_role_record_forwards'],chain_rule_relative_error=proof['relative_l2_error'],peak_allocated_mib=receipt['peak_allocated_mib'])
        ap=next(s for s in reaggregation['states'] if s['fold']==fid and s['state']==state)
        tree_equal('reaggregation_state:'+name,ap,expected_agg)
        states.append(expected_agg)
        scope.append(dict(fold=fid,state=state,observed_source_records=len(seen),total_source_records=len(pf['source_record_indices']),observed_source_identities=len(seen_ids),observed_source_scenes=sorted(seen_scenes),remapped_class_zero_record_exposures=label_zero,history_nonzero_record_exposures=nonzero_records,vjp_groups_per_batch=group_counts,available_history_groups_per_batch=full_group_counts,distance_elements=offset//4))
        matched.append([(r['record_indices'],r['pixel_sha256'],r['memory']) for r in rows])
    check('fold:matched',matched[0]==matched[1]==matched[2])

saved_csv=list(csv.DictReader((E/'reaggregation/all_role_history_steps.csv').open(encoding='utf-8',newline='')))
check('csv:row_count',len(saved_csv)==len(csv_rows)==135)
for i,(a,b) in enumerate(zip(saved_csv,csv_rows)):
    check('csv:fields',a.keys()==b.keys())
    for key,value in b.items():
        if isinstance(value,float):close('csv:'+str(i)+':'+key,float(a[key]),value)
        else:check('csv:'+str(i)+':'+key,a[key]==('' if value is None else str(value)))
check('all:batch_count',len(all_step_rows)==cpu['batches']==reaggregation['batches']==72)
check('all:distance_elements',sum(r['distance_float_count'] for r in all_step_rows)==cpu['distance_elements']==625920)
check('all:role_rows',reaggregation['role_history_rows']==135)
close('all:reported_closure',reaggregation['max_gradient_norm_identity_relative_error'],maximums['gradient_norm_identity'])

report=(ROOT/'results/MSVR310_HISTORY_CANDIDATE_GRADIENT_PREFLIGHT_2026-09-08.md').read_text(encoding='utf-8')
table_counts=Counter()
for line_number,line in enumerate(report.splitlines(),1):
    if not re.match(r'\| [012] \|',line):continue
    parts=[x.strip() for x in line.strip('|').split('|')]
    fid=int(parts[0]);state=parts[1];s=next(x for x in states if x['fold']==fid and x['state']==state)
    if len(parts)==6:
        table_counts['engineering']+=1
        check('report_table:engineering_'+str(line_number),int(parts[2])==8 and parts[3]==format(s['chain_rule_relative_error'],'.8e') and int(parts[4])==s['extra_role_record_forwards'] and parts[5]==format(s['peak_allocated_mib'],'.3f'))
    elif len(parts)==7:
        table_counts['role']+=1;role=parts[2]
        for field,text_value in zip(('history_to_current_ratio','current_history_cosine','current_both_cosine','task_total_both_cosine'),parts[3:]):
            check('report_table:role_'+str(line_number)+':'+field,text_value==format(s['roles'][role][field]['mean'],'.6f'))
check('report_table:count',table_counts==dict(engineering=9,role=27))

pipeline=read(E/'pipeline.json');observation=read(E/'observation_053241.json')
tree_equal('observation:pipeline',observation['pipeline'],pipeline)
check('pipeline:binding',pipeline['config_sha256']==config_hash and pipeline['code_commit']==summary['project_commit'])
check('pipeline:stage_sequence',[s['stage'] for s in pipeline['stages']]==['t0','preflight','preflight_cpu','source'])
for i,stage in enumerate(pipeline['stages'][:3]):
    check('pipeline:exit',stage['exit_code']==0)
    elapsed=(datetime.fromisoformat(stage['ended_at'])-datetime.fromisoformat(stage['started_at'])).total_seconds()
    close('pipeline:time',elapsed,stage['elapsed_seconds'],rel=0,abs_tol=.005)
    check('pipeline:order',stage['ended_at']<pipeline['stages'][i+1]['started_at'])
check('pipeline:source_incomplete','ended_at' not in pipeline['stages'][-1] and pipeline['status']=='RUNNING')
events=[]
for line in (E/'preflight.log').read_text(encoding='utf-8').splitlines():
    if line.startswith('{'):events.append(json.loads(line))
check('logs:state_events',[(e['fold'],e['state']) for e in events if e['event']=='candidate_gradient_state']==[(f,s) for f in range(3) for s in config['states']])
check('logs:epoch_events',len([e for e in events if e['event']=='candidate_gradient_epoch' and e['epoch']==1 and e['batches']==8])==9)
check('logs:cpu',json.loads((E/'preflight_cpu.log').read_text(encoding='utf-8'))=={k:cpu[k] for k in ('status','batches','distance_elements')})
for filename in ('t0.log','preflight_cpu.log'):check('observation:logs',observation['logs'][filename]==(E/filename).read_text(encoding='utf-8'))
check('observation:preflight_log_tail',(E/'preflight.log').read_text(encoding='utf-8').endswith(observation['logs']['preflight.log']))

defined_fields=('history_to_current_ratio','current_history_cosine','current_both_cosine','task_total_both_cosine')
global_metrics={field:dict(defined=sum(r[field] is not None for r in csv_rows),minimum=min(r[field] for r in csv_rows if r[field] is not None),maximum=max(r[field] for r in csv_rows if r[field] is not None),mean=statistics.mean(r[field] for r in csv_rows if r[field] is not None),negative=sum(r[field] is not None and r[field]<0 for r in csv_rows)) for field in defined_fields}
for name in ['tools/analyze_msvr_history_gradient_text.py','refine-logs/msvr310_history_candidate_gradient_v1/EXPERIMENT_TRACKER.md','results/MSVR310_HISTORY_CANDIDATE_GRADIENT_PREFLIGHT_2026-09-08.md','docs/MSVR310_MEMORY_COORDINATE_VS_GRADIENT_BOUNDARIES_2026-09-07.md','evidence/msvr310_history_gradient_preflight_20260908/intake_manifest.json','evidence/msvr310_history_gradient_preflight_20260908/observation_053241.json','evidence/msvr310_history_gradient_preflight_20260908/reaggregation/complete_text_reaggregation.json','evidence/msvr310_history_gradient_preflight_20260908/reaggregation/all_role_history_steps.csv','evidence/msvr310_fresh_coordinate_complete_q1_20260908/intake_manifest.json']:
    digest(ROOT/name)
digest(ROOT/'docs/MSVR310_PARTIAL_VS_TOTAL_METRIC_GRADIENT_2026-09-08.md')
result=dict(status='PASS_LOCAL_DETERMINISTIC_CHECKS' if not failures else 'FAIL_LOCAL_DETERMINISTIC_CHECKS',failures=failures,check_counts=dict(counts),total_checks=sum(counts.values()),raw_files=len(manifest['files']),raw_bytes=raw_bytes,previous_manifest_files=len(previous_manifest['files']),config_sha256=config_hash,summary_sha256=digest(E/'preflight/summary.json'),protocol_scope=protocol_scope,states=scope,role_rows=len(csv_rows),history_nonzero_role_rows=sum(r['history_norm']>0 for r in csv_rows),history_above_noise_role_rows=sum(r['history_above_repeat_noise'] for r in csv_rows),all_history_batches=sum(bool(r['memory']) for r in all_step_rows),extra_role_record_forwards=sum(s['extra_role_record_forwards'] for s in states),distance_elements=sum(r['distance_float_count'] for r in all_step_rows),maximum_memory_records=max(memory_sizes),maximum_errors=dict(maximums),global_metrics=global_metrics,report_tables=dict(table_counts),bindings=bindings,remote_inputs_unavailable=remote,audited_input_hashes=hashes,review_independence='same-family',acceptance_status='provisional',limits=['No remote tensor, checkpoint or image access; matrix values, parameter VJPs, exact reencoding and state hashes were not independently reconstructed.','Text algebra verifies internal consistency only; not authenticity of runtime tensors.','Prior Q1 gate logic checked against sealed numeric evidence; prior identity bootstrap was not resampled.'])
(OUT/'independent_checks.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({k:result[k] for k in ('status','failures','total_checks','raw_files','raw_bytes','role_rows','history_nonzero_role_rows','history_above_noise_role_rows','all_history_batches','extra_role_record_forwards','distance_elements','maximum_memory_records','maximum_errors','global_metrics','report_tables','protocol_scope','states')},indent=2))
print('BOUND_STATUS_COUNTS',dict(Counter(x['status'] for x in bindings)))
print('NONEXACT_BINDINGS',json.dumps([b for b in bindings if b['status']!='EXACT'],indent=2))
print('REMOTE_UNAVAILABLE_COUNT',len(remote))
