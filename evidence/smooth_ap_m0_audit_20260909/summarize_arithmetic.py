import json
from pathlib import Path
OUT=Path(__file__).resolve().parent
x=json.loads((OUT/'remote_arithmetic.stdout').read_bytes())
summary={k:v for k,v in x.items() if k!='steps'}
summary['verified_steps']=len(x['steps']);summary['verified_anchors']=64*len(x['steps'])
summary['errors']={k:max(r[k] for r in x['steps']) for k in ('max_saved_ap_error','max_scalar_error','smooth_distance_jacobian_max_error_float64','smooth_distance_jacobian_max_error_float32','ledger_error')}
summary['maximum_direct_relative_l2_error']=max(r['direct_relative_l2_error'] or 0 for r in x['steps'])
summary['maximum_role_triplet_error']=max(v for r in x['steps'] for v in r['role_triplet_errors'].values())
summary['verified_distance_elements']=sum(r['matrix_elements'] for r in x['endpoints'])
summary['verified_fused_derivative_elements_per_objective_per_dtype']=summary['verified_distance_elements']//4
(OUT/'arithmetic_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
with (OUT/'independent_step_checks.jsonl').open('w',encoding='utf-8') as f:
    for row in x['steps']:f.write(json.dumps(row)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='checkpoints'},indent=2))
