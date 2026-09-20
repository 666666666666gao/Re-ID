from pathlib import Path
import csv,json,statistics
root=Path('C:/Users/gb/.codex_tmp/smooth_ap_source_coverage_complete_20260909')
checked=json.loads((root/'coverage_verification.json').read_bytes())
assert checked['status']=='PASS_ALL_SOURCE_COVERAGE_ROWS' and checked['candidate_rows']==1996800
data=json.loads((root/'coverage.json').read_bytes());assert len(data['conditions'])==12
records=[]
for cond in data['conditions']:
    parts=cond['directory'].split('_');fold=int(parts[1]);end='control' if '_control_' in cond['directory'] else 'smooth_ap'
    view=parts[-1]
    for m in cond['metrics']:records.append(dict(fold=fold,endpoint=end,view=view,**m))
assert len(records)==120
groups={}
for r in records:groups.setdefault((r['endpoint'],r['view'],r['output'],r['cross_scene']),[]).append(r)
aggregate=[]
for key,rows in groups.items():
    assert len(rows)==3
    n=sum(r['full_eligible'] for r in rows);common=sum(r['common_eligible'] for r in rows)
    aggregate.append(dict(endpoint=key[0],view=key[1],output=key[2],cross_scene=key[3],
        full_eligible_members=n,full_source_map=sum(r['full_source_map']*r['full_eligible'] for r in rows)/n,
        common_eligible_exposures=common,total_anchor_exposures=sum(r['anchor_exposures'] for r in rows),
        pool_missing_positive=sum(r['pool_missing_positive'] for r in rows),
        common_candidate_map=sum(r['common_candidate_map']*r['common_eligible'] for r in rows)/common,
        common_full_map=sum(r['common_full_map']*r['common_eligible'] for r in rows)/common))
pairs=[]
for view in ('clean','augmented'):
    for output in ('baseline_only','fused','cnn','transformer','mamba'):
        for cross in (False,True):
            rows=[]
            for fold in range(3):
                filename=output+('_cross_scene' if cross else '_all_identity')+'_full.json'
                control=json.loads((root/f'fold_{fold}_control_{view}'/filename).read_bytes())
                candidate=json.loads((root/f'fold_{fold}_smooth_ap_{view}'/filename).read_bytes())
                for a,b in zip(control,candidate,strict=True):
                    assert a['record_index']==b['record_index'] and a['eligible']==b['eligible']
                    if a['eligible']:rows.append((a,b))
            pairs.append(dict(view=view,output=output,cross_scene=cross,eligible_members=len(rows),
                ap_gain_pp=100*statistics.mean(b['ap']-a['ap'] for a,b in rows),
                control_rank1=100*statistics.mean(a['rank1'] for a,b in rows),candidate_rank1=100*statistics.mean(b['rank1'] for a,b in rows),
                better=sum(b['ap']>a['ap']+1e-12 for a,b in rows),worse=sum(b['ap']<a['ap']-1e-12 for a,b in rows),
                control_inverted_positive_positions=sum(a['inverted_positives'] for a,b in rows),
                candidate_inverted_positive_positions=sum(b['inverted_positives'] for a,b in rows)))
dest=Path('C:/Users/gb/.codex_tmp/smooth_ap_source_coverage_analysis_20260909');dest.mkdir()
for name,rows in [('all120_conditions',records),('all40_aggregates',aggregate),('all20_source_pairs',pairs)]:
    (dest/(name+'.json')).write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
    with (dest/(name+'.csv')).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps(dict(fused=[r for r in aggregate if r['output']=='fused'],
    source_pairs=[r for r in pairs if r['view']=='augmented' and r['cross_scene']]),indent=2))
