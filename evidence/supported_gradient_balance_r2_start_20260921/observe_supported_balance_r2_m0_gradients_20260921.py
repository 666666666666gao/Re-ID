from pathlib import Path
import json,datetime
root=Path('/root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_r2_seed42_1381639')
rows=[]
for path in sorted((root/'m0').glob('*/memory_steps.jsonl')):
    lines=path.read_text().splitlines()
    # The writer flushes complete JSON lines; the last incomplete line is not evidence.
    if lines and not lines[-1].endswith('}'):lines=lines[:-1]
    items=[json.loads(line) for line in lines]
    direct=[dict(step=r['step'],checks=r['rank_auxiliary_reference_checks']) for r in items if r['rank_auxiliary_reference_checks']]
    if items:
        last=items[-1]
        rows.append(dict(endpoint=path.parent.name,completed_step_records=len(items),direct_reference=direct,
            last_step=last['step'],last_support=last['support'],
            last_coefficients={e:[v['applied_rank_weight'],v['applied_auxiliary_weight']] for e,v in last['gradient_balance'].items()},
            head_gradient_preserved=all(r['classification_head_gradients_unchanged'] for r in items),
            source_only_runtime_witness=True))
print(json.dumps(dict(observed_at=datetime.datetime.now().astimezone().isoformat(),root=str(root),endpoints=rows,
    scope='Saved partial M0 runtime witnesses only; no independent regeneration, no M0 pass implied')))
