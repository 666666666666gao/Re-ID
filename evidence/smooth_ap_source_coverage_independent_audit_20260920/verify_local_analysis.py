from pathlib import Path
from collections import defaultdict,Counter
import json,csv,hashlib,math,datetime
out=Path(__file__).parent
repo=out/'snapshots/repo'
complete=repo/'evidence/smooth_ap_source_coverage_complete_20260909'
analysis=repo/'evidence/smooth_ap_source_coverage_analysis_20260909'
load=lambda p:json.loads(p.read_bytes())
inventory=load(complete/'inventory.json')['files']
for entry in inventory:
    p=complete/entry['name'];data=p.read_bytes()
    assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256'],entry['name']
assert len(inventory)==159 and sum(e['bytes'] for e in inventory)==13715377
coverage=load(complete/'coverage.json')
conditions=[]
for c in coverage['conditions']:
    parts=c['directory'].split('_');fold=int(parts[1]);endpoint='control' if parts[2]=='control' else 'smooth_ap';view=parts[-1]
    conditions.extend(dict(fold=fold,endpoint=endpoint,view=view,**m) for m in c['metrics'])
assert len(conditions)==120
assert conditions==load(analysis/'all120_conditions.json')
by_key=defaultdict(list)
for row in conditions:by_key[row['endpoint'],row['view'],row['output'],row['cross_scene']].append(row)
aggregates=[]
for key,rows in by_key.items():
    assert {r['fold'] for r in rows}=={0,1,2}
    members=sum(r['full_eligible'] for r in rows);exposures=sum(r['common_eligible'] for r in rows)
    aggregates.append(dict(endpoint=key[0],view=key[1],output=key[2],cross_scene=key[3],full_eligible_members=members,
      full_source_map=math.fsum(r['full_source_map']*r['full_eligible'] for r in rows)/members,
      common_eligible_exposures=exposures,total_anchor_exposures=sum(r['anchor_exposures'] for r in rows),pool_missing_positive=sum(r['pool_missing_positive'] for r in rows),
      common_candidate_map=math.fsum(r['common_candidate_map']*r['common_eligible'] for r in rows)/exposures,
      common_full_map=math.fsum(r['common_full_map']*r['common_eligible'] for r in rows)/exposures))
pairs=[]
for view in ('clean','augmented'):
    for output in ('baseline_only','fused','cnn','transformer','mamba'):
        for cross in (False,True):
            eligible=[]
            for f in range(3):
                name=output+('_cross_scene' if cross else '_all_identity')+'_full.json'
                a=load(complete/f'fold_{f}_control_{view}'/name);b=load(complete/f'fold_{f}_smooth_ap_{view}'/name)
                assert len(a)==len(b)
                for r,s in zip(a,b):
                    assert (r['record_index'],r['eligible'],r['positives'],r['negatives'])==(s['record_index'],s['eligible'],s['positives'],s['negatives'])
                    if r['eligible']:eligible.append((r,s))
            n=len(eligible)
            pairs.append(dict(view=view,output=output,cross_scene=cross,eligible_members=n,
              ap_gain_pp=100*math.fsum(b['ap']-a['ap'] for a,b in eligible)/n,
              control_rank1=100*sum(a['rank1'] for a,b in eligible)/n,candidate_rank1=100*sum(b['rank1'] for a,b in eligible)/n,
              better=sum(b['ap']-a['ap']>1e-12 for a,b in eligible),worse=sum(a['ap']-b['ap']>1e-12 for a,b in eligible),
              control_inverted_positive_positions=sum(a['inverted_positives'] for a,b in eligible),candidate_inverted_positive_positions=sum(b['inverted_positives'] for a,b in eligible)))
max_error=0.
def compare(a,b):
    global max_error
    assert len(a)==len(b)
    for row,expected in zip(a,b):
        assert set(row)==set(expected)
        for k,value in row.items():
            if isinstance(value,float):
                error=abs(value-expected[k]);max_error=max(max_error,error);assert error<1e-10,(k,value,expected[k])
            else:assert value==expected[k],(k,value,expected[k])
compare(aggregates,load(analysis/'all40_aggregates.json'))
compare(pairs,load(analysis/'all20_source_pairs.json'))
for name,rows in [('all120_conditions',conditions),('all40_aggregates',aggregates),('all20_source_pairs',pairs)]:
    with (analysis/(name+'.csv')).open(encoding='utf-8',newline='') as f:
        csv_rows=list(csv.DictReader(f))
    assert len(csv_rows)==len(rows)
    for row,saved in zip(rows,csv_rows):
        assert set(row)==set(saved)
        for key,value in row.items():
            if isinstance(value,float):assert abs(float(saved[key])-value)<1e-10
            else:assert str(value)==saved[key]
fused=[r for r in aggregates if r['output']=='fused']
highlight=[r for r in pairs if r['view']=='augmented' and r['cross_scene']]
text=(repo/'results/MSVR310_SMOOTH_AP_SOURCE_COVERAGE_2026-09-09.md').read_text(encoding='utf-8')
assert all(f"{r['full_source_map']:.6f}" in text for r in fused)
aug=[r for r in fused if r['view']=='augmented']
assert all(f"{r['common_candidate_map']:.6f}" in text and f"{r['common_full_map']:.6f}" in text for r in aug)
assert all(f"{r['ap_gain_pp']:.6f}" in text for r in highlight if r['output']!='baseline_only')
protocol=load(repo/'protocols/msvr310_train_oof_v1.json')
scopes=[]
for f in protocol['folds']:
    records=[protocol['records'][i] for i in f['source_record_indices']]
    scenes=defaultdict(set)
    for r in records:scenes[r['identity']].add(r['scene'])
    scopes.append(dict(fold=f['fold'],source_records=len(records),source_identities=len(scenes),cross_scene_identities=sum(len(v)>1 for v in scenes.values()),cross_scene_records=sum(len(scenes[r['identity']])>1 for r in records)))
result=dict(status='PASS_COMPLETE_LOCAL_ANALYSIS_AND_TEXT_INTAKE',generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),intake_files=159,intake_bytes=13715377,condition_rows=120,aggregate_rows=40,pair_rows=20,csv_rows=180,max_numeric_error=max_error,fold_scope=scopes,all40_aggregates=aggregates,all20_source_pairs=pairs,report_table_and_gain_claims_match=True)
(out/'local_analysis_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('all40_aggregates','all20_source_pairs')},indent=2))
print(json.dumps(dict(fused=fused,highlight=highlight),indent=2))
