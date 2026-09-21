from pathlib import Path
import csv,hashlib,json
tmp=Path('D:/Program Files/UserCache/gb/codex/tmp')
source=tmp/'trifusion_supported_task_state_m0_complete_20260921'
output=tmp/'trifusion_supported_task_state_m0_analysis_20260921'
assert not output.exists()
inventory=json.loads((source/'intake_inventory.json').read_bytes())
for row in inventory['files']:
    data=(source/row['path']).read_bytes()
    assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
summary=json.loads((source/'m0/summary.json').read_bytes())
assert summary['status']=='PASS_ENGINEERING_ONLY'
endpoints=[];training_rows=[];role_rows=[];reference_errors=[]
directories=[f'fold_{f}_{end}' for f in range(3) for end in ('control','split')]+['overfit_control','overfit_split']
for directory in directories:
    tr=json.loads((source/'m0'/directory/'training.json').read_bytes())
    audits=[json.loads(line) for line in (source/'m0'/directory/'memory_steps.jsonl').read_text().splitlines()]
    assert len(audits)==tr['optimizer_steps']
    split=tr['task_state_split'];observed=0;active=0;missing=0
    for row,audit in zip(tr['steps'],audits,strict=True):
        assert row['step']==audit['step']
        observed+=int(audit['rank_observed']);active+=int(audit['replacement_active'])
        missing+=int(audit['replacement_active'] and not audit['rank_observed'])
        training_rows.append(dict(endpoint=directory,step=row['step'],epoch=row['epoch'],loss=row['loss'],
            rank_observed=audit['rank_observed'],eligible_anchors=audit['support']['eligible_anchors'],
            active_fused_metric=row['active_fused_metric'],**row['components']))
        for role in ('cnn','transformer','mamba'):
            update=audit['actual_parameter_updates'][role]
            tasks=audit['task_states'][role]
            role_rows.append(dict(endpoint=directory,step=row['step'],role=role,split=split,
                observed_rank_count=observed,actual_update_norm=update['difference_norm'],
                parameter_norm_before=update['first_norm'],
                relative_update_norm=update['difference_norm']/update['first_norm'],
                task_states=json.dumps(tasks,sort_keys=True)))
            if audit['rank_auxiliary_reference_checks']:
                for name,check in audit['rank_auxiliary_reference_checks'][role].items():
                    assert check['passed']
                    reference_errors.append(dict(endpoint=directory,step=row['step'],role=role,component=name,
                        relative_l2_error=check['relative_l2_error'],absolute_error=check['difference_norm']))
    endpoints.append(dict(endpoint=directory,steps=len(audits),active_ap_steps=active,
        observed_rank_steps=observed,absent_ap_steps=missing,final_task_states=audits[-1]['task_states'],
        trainable_tensors=tr['trainable_tensors'],nonzero_gradient_tensors=tr['nonzero_gradient_tensors'],
        overflow_events=tr['overflow_events'],optimizer_state=tr['optimizer_state'],
        peak_allocated_mib=tr['peak_allocated_mib'],peak_reserved_mib=tr['peak_reserved_mib']))
assert len(training_rows)==248 and len(role_rows)==744 and len(reference_errors)==90
output.mkdir()
for name,rows in [('training_steps.csv',training_rows),('role_steps.csv',role_rows),('reference_checks.csv',reference_errors)]:
    with (output/name).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
result=dict(status='COMPLETE_M0_TEXT_ANALYSIS_ENGINEERING_ONLY',endpoints=endpoints,
    steps=248,role_steps=744,component_references=90,
    maximum_reference_relative_error=max(r['relative_l2_error'] for r in reference_errors if r['relative_l2_error'] is not None),
    overfit={end:dict(gate=item['gate'],checks=item['checks']) for end,item in summary['overfit'].items()},
    source_only=True,heldout_or_official_result=False,
    limitations=['Scalar update witnesses do not recover per-task preconditioned vectors.',
                 'Cumulative nonzero gradients do not imply every tensor is nonzero every step.',
                 'M0 PASS does not imply Q1 qualification.'])
(output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=result['status'],steps=248,references=90,maximum_reference_relative_error=result['maximum_reference_relative_error'])))
