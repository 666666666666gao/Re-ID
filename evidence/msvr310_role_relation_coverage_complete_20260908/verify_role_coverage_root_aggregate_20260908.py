from pathlib import Path
import hashlib,json
t=Path('C:/Users/gb/.codex_tmp')
original=t/'msvr_role_relation_coverage_all18_aggregation_20260908.json'
independent=t/'msvr_role_relation_coverage_independent_audit_20260908/verified_aggregate_counts.json'
a=json.loads(original.read_bytes())['conditions']
b=json.loads(independent.read_bytes())
checked=0
for r in a:
    group=['state_view_protocol',r['state'],r['view'],r['protocol']]
    matches=[x for x in b if x['group']==group]
    assert len(matches)==1
    assert r['counts']==matches[0]['counts'],group
    checked+=len(r['counts'])
result=dict(status='PASS_ROOT_18_AGGREGATES_MATCH_INDEPENDENT_REPLAY',groups=len(a),counter_values=checked,inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (original,independent)},scope='Root arithmetic comparison; does not replace independent semantic audit or establish training gradients.')
(t/'role_coverage_root_aggregate_regression_20260908.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
