from pathlib import Path
import csv,hashlib,json,math

b=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
source=b/'checked_m0_terminal_failure887/REMOTE.json'
packet=json.loads(source.read_bytes())
out=b/'checked_m0_support_analysis889';assert not out.exists();out.mkdir()
assert packet['exit']['exit_code']==1 and len(packet['rows'])==6
input_hashes={str(source):hashlib.sha256(source.read_bytes()).hexdigest()}
rows=[]
for previous in packet['rows']:
    dataset,objective=previous['dataset'],previous['objective']
    run=f'trained-model/incremental_role_objective_m0_v1_20261007_886_m0_{objective}_{dataset}'
    needed=[run+'/training.json',run+'/training_steps.jsonl']
    for name in needed:
        artifact=packet['files'][name]
        assert hashlib.sha256(artifact['text'].encode('utf-8')).hexdigest()==artifact['sha256']
        input_hashes[name]=artifact['sha256']
    training=json.loads(packet['files'][needed[0]]['text'])
    steps=[json.loads(line) for line in packet['files'][needed[1]]['text'].splitlines()]
    assert len(steps)==8 and training['production_m0_diagnostics']['effective_optimizer_updates']==8
    assert all(s['incremental_isolated_vjp_autocast_enabled'] is False and s['incremental_isolated_shared_global_gradient_absent'] for s in steps)
    norms=previous['query_key_norm_sums']
    assert all(math.isfinite(v) and v>=0 for v in norms.values())
    row=dict(dataset=dataset,objective=objective,production_steps=8,qualified=previous['accepted'],
        batch_size=training['initializer']['batch_size'],query_exposures=8*training['initializer']['batch_size'],
        correction_gradient_norm_sum=previous['correction_norm_sum'],
        query_key_norm_sums=norms,
        query_key_zero_steps={name:sum(s['incremental_isolated_query_key_gradient_norms'][name]==0 for s in steps) for name in norms},
        query_key_unused_steps={name:sum(s['incremental_isolated_query_key_unused'][name] for s in steps) for name in norms},
        incremental_loss_range=[min(s['incremental_loss'] for s in steps),max(s['incremental_loss'] for s in steps)])
    if objective=='repair_keep':
        keys=('legal_positive_pairs','legal_triplets','eligible_queries','multiple_positive_queries',
              'repair_triplets','keep_triplets','repair_queries','keep_queries','repair_active_triplets','keep_active_triplets')
        row['support_exposure_totals']={k:sum(s[k] for s in steps) for k in keys}
        row['support_batch_ranges']={k:[min(s[k] for s in steps),max(s[k] for s in steps)] for k in keys}
        row['eligible_query_fraction']=row['support_exposure_totals']['eligible_queries']/row['query_exposures']
        row['multiple_positive_fraction_of_eligible']=row['support_exposure_totals']['multiple_positive_queries']/row['support_exposure_totals']['eligible_queries']
        row['repair_active_batches']=sum(s['repair_active_triplets']>0 for s in steps)
        row['keep_active_batches']=sum(s['keep_active_triplets']>0 for s in steps)
        assert all(s['repair_triplets']+s['keep_triplets']==s['legal_triplets'] for s in steps)
    rows.append(row)
failed=next(r for r in rows if (r['dataset'],r['objective'])==('RGBNT100','repair_keep'))
assert failed['eligible_query_fraction']==1 and failed['multiple_positive_fraction_of_eligible']==1
assert failed['repair_active_batches']==failed['keep_active_batches']==8
assert not failed['qualified']
assert failed['query_key_zero_steps']['evidence_model.roles.query_projections.2.weight']==8
assert failed['query_key_zero_steps']['evidence_model.roles.key_projections.2.weight']==8
summary=dict(status='COMPLETED_SAVED_EIGHT_STEP_SUPPORT_ANALYSIS',input_sha256=input_hashes,rows=rows,
    conclusion='Failed100 repair_keep has legal positive/negative relations and both active cells in every saved batch, with all queries eligible and multiple positive support. No-evaluable-relation explanation is contradicted for these eight batches. Third-role Q/K isolated unscaled norm remains0/nonunused. Neither its numerical cause nor formal training efficacy is established.',
    boundary='CPU saved-text analysis only. Eight batches per endpoint, repeated relation/query exposures, not unique identities or complete training. No forward/optimizer/server query/source sync/rerun/gate change/new retrieval score. Different loss norms are not directly comparable efficacy. Broadgoal ACTIVE_UNMET.')
(out/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
fields=['dataset','objective','qualified','batch_size','query_exposures','eligible_query_fraction','multiple_positive_fraction_of_eligible','repair_active_batches','keep_active_batches','correction_gradient_norm_sum']
with (out/'SUPPORT.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
lines=['# Saved M0 support analysis (eight steps only)','',
    '| Dataset | Objective | Gate | Eligible query exposure | Multiple positive / eligible | Repair / keep active batches |',
    '|---|---|---|---:|---:|---:|']
for r in rows:
    if r['objective']=='repair_keep':
        lines.append(f'| {r["dataset"]} | repair_keep | {"PASS" if r["qualified"] else "FAIL"} | {r["eligible_query_fraction"]:.6f} | {r["multiple_positive_fraction_of_eligible"]:.6f} | {r["repair_active_batches"]}/8, {r["keep_active_batches"]}/8 |')
lines += ['',summary['conclusion'],'',summary['boundary']]
(out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(status=summary['status'],repair_rows=[{k:r[k] for k in ('dataset','eligible_query_fraction','multiple_positive_fraction_of_eligible','repair_active_batches','keep_active_batches','support_exposure_totals')} for r in rows if r['objective']=='repair_keep'],boundary=summary['boundary']),indent=2))
