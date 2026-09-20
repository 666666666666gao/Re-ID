"""Summarize complete saved M0 text witnesses, without inventing gradient replay."""
from pathlib import Path
import csv, hashlib, json, statistics, sys

source=Path(sys.argv[1]); output=Path(sys.argv[2])
assert not output.exists()
inventory=json.loads((source/'inventory.json').read_bytes())
for item in inventory['files']:
    data=(source/item['path']).read_bytes()
    assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
summary=json.loads((source/'m0/summary.json').read_bytes())
cpu=json.loads((source/'m0_cpu.json').read_bytes())
assert summary['status']=='PASS_ENGINEERING_ONLY'
assert cpu['status']=='PASS_COMPLETE_SUPPORTED_GRADIENT_BALANCE_M0'
assert cpu['checked_training_steps']==248
roles=('cnn','transformer','mamba')
rows=[]; endpoints=[]; references=[]; epochs=[]
for path in sorted((source/'m0').glob('*/training.json')):
    training=json.loads(path.read_bytes()); name=path.parent.name
    audits=[json.loads(line) for line in (path.parent/'memory_steps.jsonl').read_text().splitlines()]
    assert len(audits)==len(training['steps'])==training['optimizer_steps']
    supported=zero=0
    for item,step in zip(audits,training['steps'],strict=True):
        assert item['step']==step['step']
        support=item['support']; active=item['replacement_active']
        supported+=int(active and support['eligible_anchors']>0)
        zero+=int(active and support['eligible_anchors']==0)
        for role in roles:
            balance=item['gradient_balance'][role]
            pair=balance['rank_vs_auxiliary']
            sub=balance['subtraction_auxiliary_vs_direct']
            direct=balance['direct_sum_vs_original']
            row=dict(endpoint=name,step=item['step'],epoch=step['epoch'],role=role,
                active_fused_metric=step['active_fused_metric'],supported=balance['supported'],
                eligible_anchors=support['eligible_anchors'],eligible_identities=support['eligible_identities'],
                identity_directed_scene_relations=support['identity_directed_scene_relations'],
                rank_norm=pair['first_norm'],auxiliary_norm=pair['second_norm'],rank_auxiliary_cosine=pair['cosine'],
                rank_to_auxiliary_norm=(pair['first_norm']/pair['second_norm'] if pair['second_norm'] else None),
                proposed_rank_weight=balance['proposed_rank_weight'],applied_rank_weight=balance['applied_rank_weight'],
                applied_auxiliary_weight=balance['applied_auxiliary_weight'],ratio=balance['ratio'],
                history_rank_norm=balance['current_rank_vs_history']['second_norm'],
                current_history_cosine=balance['current_rank_vs_history']['cosine'],
                original_to_applied_difference=balance['original_sum_vs_applied']['difference_norm'],
                direct_sum_to_original_difference=direct['difference_norm'],
                subtraction_to_direct_auxiliary_difference=sub['difference_norm'],
                actual_adamw_parameter_delta_norm=item['actual_parameter_updates'][role]['difference_norm'],
                classification_head_gradient_preserved=item['classification_head_gradients_unchanged'])
            rows.append(row)
            for component,check in item['rank_auxiliary_reference_checks'].get(role,{}).items():
                references.append(dict(endpoint=name,step=item['step'],role=role,component=component,**check))
    endpoints.append(dict(endpoint=name,steps=len(audits),supported_steps=supported,zero_supported_steps=zero,
        warmup_steps=audits[0]['warmup_steps'],live_tensors=training['nonzero_gradient_tensors'],
        overflow=training['overflow_events'],peak_reserved_mib=training['peak_reserved_mib'],
        training_seconds=sum(e['elapsed_seconds'] for e in training['history']),
        current_rank_backward_calls=training['current_rank_backward_calls'],
        current_auxiliary_backward_calls=training['current_auxiliary_backward_calls'],
        direct_component_backward_calls=training['direct_component_backward_calls'],
        extra_fresh_role_record_forwards=training['extra_fresh_role_record_forwards'],
        extra_history_vjp_record_forwards=training['extra_history_vjp_record_forwards']))
    epochs.extend(dict(endpoint=name,**e) for e in training['history'])
assert sum(e['steps'] for e in endpoints)==248 and len(rows)==744
assert len(references)==90 and all(r['passed'] for r in references)
def describe(values):
    values=[v for v in values if v is not None]
    return dict(count=len(values),minimum=min(values),median=statistics.median(values),maximum=max(values)) if values else dict(count=0)
aggregate=[]
for endpoint in endpoints:
    for role in roles:
        selected=[r for r in rows if r['endpoint']==endpoint['endpoint'] and r['role']==role and r['supported']]
        aggregate.append(dict(endpoint=endpoint['endpoint'],role=role,supported_role_steps=len(selected),
            rank_to_auxiliary_norm=describe([r['rank_to_auxiliary_norm'] for r in selected]),
            applied_rank_weight=describe([r['applied_rank_weight'] for r in selected]),
            applied_auxiliary_weight=describe([r['applied_auxiliary_weight'] for r in selected]),
            actual_adamw_parameter_delta_norm=describe([r['actual_adamw_parameter_delta_norm'] for r in selected]),
            negative_rank_auxiliary_cosines=sum(r['rank_auxiliary_cosine'] is not None and r['rank_auxiliary_cosine']<0 for r in selected)))
report=dict(status='COMPLETE_M0_SAVED_TEXT_SUMMARY',seed=42,steps=248,role_step_rows=744,
    source_files_verified=len(inventory['files']),endpoint_summaries=endpoints,role_summaries=aggregate,
    direct_reference_checks=len(references),maximum_direct_reference_relative_l2_error=max(r['relative_l2_error'] or 0 for r in references),
    overfit_gates={k:v['gate'] for k,v in summary['overfit'].items()},
    all_head_gradient_witnesses_preserved=all(r['classification_head_gradient_preserved'] for r in rows),
    scope='Saved M0 runtime witnesses and deterministic text aggregation. No independent parameter-gradient regeneration, no retrieval evidence, no attribution of AdamW update fractions. Capacity and overfit are distinct distributions; not a Q1 trajectory estimate.',
    summary_sha256=hashlib.sha256((source/'m0/summary.json').read_bytes()).hexdigest(),
    cpu_receipt_sha256=hashlib.sha256((source/'m0_cpu.json').read_bytes()).hexdigest())
output.mkdir(parents=True)
for filename,data in [('role_steps.csv',rows),('direct_reference_checks.csv',references),('epochs.csv',epochs)]:
    with (output/filename).open('w',encoding='utf-8',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
(output/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ('endpoint_summaries','role_summaries')}))
