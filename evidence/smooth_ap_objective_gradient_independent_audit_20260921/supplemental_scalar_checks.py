"""Inspect the registered error denominator's cancellation boundary on every source row."""
from pathlib import Path
import hashlib,json,math
root=Path(__file__).parent
directory=root/'snapshots/terminal_texts/artifacts/msvr310_smooth_ap_objective_gradients_source_seed42_c6fbfb4_20260920'
summary=json.loads((directory/'summary.json').read_bytes())
max_relative_to_total={'current_decomposition':0.,'full_decomposition':0.}
min_total_to_component_sum={'current_decomposition':math.inf,'full_decomposition':math.inf}
max_full_sum_norm_relative_error=0.;max_repeat_bound_to_fused=0.;rows_checked=0
locations={};zero_history_rows=0;zero_fused_rows=0;undefined_FO=0
for condition in summary['conditions']:
    p=directory/condition['directory']/'steps.jsonl'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==condition['files']['steps.jsonl']['sha256']
    for line in p.read_text().splitlines():
        row=json.loads(line);rows_checked+=1;rs=list(row['roles'].values())
        onorm=math.sqrt(math.fsum(x['fused_vs_other']['second_norm']**2 for x in rs))
        for name,kind in [('current_decomposition','current_repeat'),('full_decomposition','fused_vs_other')]:
            c=row[name];fnorm=math.sqrt(math.fsum(x[kind]['first_norm']**2 for x in rs));total=max(c['first_norm'],c['second_norm'])
            ratio=c['difference_norm']/total if total else 0.
            if ratio>max_relative_to_total[name]:locations[name]=dict(condition=condition['directory'],step=row['step'])
            max_relative_to_total[name]=max(max_relative_to_total[name],ratio)
            min_total_to_component_sum[name]=min(min_total_to_component_sum[name],total/(fnorm+onorm))
        predicted=math.sqrt(math.fsum(c['fused_vs_other']['first_norm']**2+c['fused_vs_other']['second_norm']**2+
                              2*c['fused_vs_other']['first_norm']*c['fused_vs_other']['second_norm']*c['fused_vs_other']['cosine'] for c in rs))
        measured=row['full_decomposition']['first_norm'];error=abs(predicted-measured)/max(predicted,measured)
        assert error<1e-6
        max_full_sum_norm_relative_error=max(max_full_sum_norm_relative_error,error)
        for c in rs:
            f=c['fused_vs_other']['first_norm']
            if f: max_repeat_bound_to_fused=max(max_repeat_bound_to_fused,(c['current_repeat']['difference_norm']+c['history_repeat']['difference_norm'])/f)
            else:zero_fused_rows+=1
            zero_history_rows+=int(c['history_repeat']['first_norm']==0)
            undefined_FO+=int(c['fused_vs_other']['cosine'] is None)
assert rows_checked==1560 and zero_fused_rows==0 and zero_history_rows==1191 and undefined_FO==0
out=dict(status='PASS_ALL_SOURCE_SCALAR_DENOMINATOR_BOUNDARIES',source_rows=rows_checked,role_rows=rows_checked*3,
         max_error_relative_to_max_sum_or_total_norm=max_relative_to_total,
         minimum_max_sum_or_total_norm_divided_by_sum_of_component_norms=min_total_to_component_sum,
         max_full_sum_norm_analytic_vs_saved_relative_error=max_full_sum_norm_relative_error,
         max_current_plus_history_repeat_difference_bound_divided_by_fused_norm=max_repeat_bound_to_fused,
         maximum_error_locations=locations,zero_history_role_rows=zero_history_rows,zero_fused_role_rows=zero_fused_rows,undefined_fused_other_cosines=undefined_FO,
         scope='Arithmetic on saved scalars only; no parameter-vector reconstruction and no new experimental acceptance gate.')
(root/'SUPPLEMENTAL_SCALAR_CHECKS.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out))
