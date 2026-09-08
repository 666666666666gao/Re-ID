"""Extract independent deterministic outputs and verify local/remote bindings."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')

conditions=[]
hashes={}
groups=defaultdict(Counter)
extras=defaultdict(Counter)
floats=defaultdict(lambda:defaultdict(list))
rows=0
witness_rows=0
with (OUT/'independently_recreated_query_rows.jsonl').open('w',encoding='utf-8',newline='\n') as queries, \
     (OUT/'independent_extremum_witnesses.jsonl').open('w',encoding='utf-8',newline='\n') as witnesses:
    for line in (OUT/'independent_replay.stdout.jsonl').open(encoding='utf-8'):
        entry=json.loads(line)
        kind=entry['kind']
        if kind=='bindings':
            bindings=entry
        elif kind=='verification':
            verification=entry
        elif kind=='input_hash':
            hashes[entry['path']]={k:entry[k] for k in ['bytes','sha256']}
        elif kind=='condition':
            row=entry['row']
            conditions.append(row)
            for key in [('all',),('protocol',row['protocol']),('state_protocol',row['state'],row['protocol']),
                        ('state_view_protocol',row['state'],row['view'],row['protocol'])]:
                groups[key].update(row['counts'])
        elif kind=='extremum_witness':
            witness_rows+=1
            witnesses.write(json.dumps(entry,sort_keys=True)+'\n')
        elif kind=='query':
            r=entry['row']
            rows+=1
            queries.write(json.dumps(r,sort_keys=True)+'\n')
            group=r['protocol']
            if r['eligible']:
                extras[group]['negative_union_identities_'+str(r['negative_union_identities'])]+=1
                extras[group]['negative_union_scenes_'+str(r['negative_union_scenes'])]+=1
                extras[group]['positive_extremum_ties_'+str(r['fused_positive_extremum_ties'])]+=1
                extras[group]['negative_extremum_ties_'+str(r['fused_negative_extremum_ties'])]+=1
                extras[group]['equal_positive_hinge_without_both_selected_fused_extrema']+=int(
                    abs(r['full_fused_hinge']-r['role_subset_fused_hinge'])<=1e-12 and r['full_fused_hinge']>0
                    and not(r['contains_fused_negative'] and r['contains_fused_positive']))
                for key in ['full_fused_hinge','role_subset_fused_hinge']:
                    floats[group][key].append(r[key])
            else:
                extras[group]['ineligible_queries']+=1
assert verification['status']=='PASS' and verification['differences']==[]
assert rows==37152 and len(conditions)==54 and witness_rows==29376
local_inputs=json.loads((OUT/'local_input_hashes.json').read_bytes())
inventory=json.loads((OUT/'remote_inventory.json').read_bytes())
remote_docs={r['path']:r for r in inventory if r['kind']=='document'}
new_output={r['path'].split('/msvr310_role_relation_coverage_v1_seed42_c46be4e/')[1]:r for r in inventory if r['kind']=='result_inventory'}
intake=json.loads((OUT/'local_snapshots/intake/intake_manifest.json').read_bytes())
assert intake['pipeline']==json.loads((OUT/'local_snapshots/intake/pipeline.json').read_bytes())
for entry in intake['files']:
    path=OUT/'local_snapshots/intake'/entry['path']
    actual=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert actual=={k:entry[k] for k in actual}=={k:new_output[entry['path']][k] for k in actual}
    remote='/root/autodl-tmp/trifusion-v2/artifacts/msvr310_role_relation_coverage_v1_seed42_c46be4e/'+entry['path']
    assert hashes[remote]==actual
local_remote_matches=[]
for entry in local_inputs:
    relative=entry['snapshot'].replace('\\','/')
    if relative.startswith('local_snapshots/project/'):
        name=relative[len('local_snapshots/project/'):]
        remote='/root/autodl-tmp/trifusion-v2/TriFusion-ReID/'+name
        if name.startswith('evidence/msvr310_source_relation_complete_20260907/'):
            remote='/root/autodl-tmp/trifusion-v2/artifacts/msvr310_source_relations_v1_seed42_4e57e54/'+name.split('/')[-1]
        if remote in remote_docs:
            assert entry['sha256']==remote_docs[remote]['sha256'] and entry['bytes']==remote_docs[remote]['bytes']
            local_remote_matches.append(dict(local=entry['path'],remote=remote,sha256=entry['sha256']))
assert len(new_output)==4
assert len(local_remote_matches)==15
summary=dict(status='PASS_DETERMINISTIC_FULL_FIELD_REPLAY',rows=rows,eligible_rows=witness_rows,ineligible_rows=rows-witness_rows,
             condition_results=len(conditions),bindings=bindings,verification=verification,
             aggregate_groups=[dict(group=list(k),counts=dict(v)) for k,v in groups.items()],
             per_protocol_additional_counts={k:dict(v) for k,v in extras.items()},
             mean_hinges={g:{k:sum(v)/len(v) for k,v in fields.items()} for g,fields in floats.items()},
             local_intake_hashes_equal_remote=True,local_remote_document_matches=local_remote_matches)
write('deterministic_verification.json',summary)
write('independently_recreated_conditions.json',conditions)
write('actual_remote_input_hashes.json',hashes)
write('verified_aggregate_counts.json',[dict(group=list(k),counts=dict(v)) for k,v in groups.items()])
print(json.dumps(dict(status=summary['status'],rows=rows,eligible_rows=witness_rows,
    array_bytes=bindings['total_array_bytes'],comparison=verification['comparison'],
    maximum_saved_float_error=verification['maximum_saved_float_error'],
    direct_checks=verification['direct_distance_witness_checks'],direct_error=verification['maximum_direct_distance_error'],
    additional_statistics=verification['additional_statistics'],per_protocol=summary['per_protocol_additional_counts'],
    aggregates=[dict(group=list(k),counts=dict(v)) for k,v in groups.items() if len(k)<=2],
    mean_hinges=summary['mean_hinges'],elapsed_seconds=verification['elapsed_seconds'],
    maximum_resident_kib=verification['maximum_resident_kib'],local_remote_document_matches=len(local_remote_matches),
    input_hashes=len(hashes)),indent=2))
