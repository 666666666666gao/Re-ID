from pathlib import Path
import json,hashlib,math
root=Path('/data/gaob/Re-ID/Trifusion');c=root/'logs/incremental_role_objective_m0_v1_20261007_878';j=root/'logs/incremental_role_objective_m0_launch_20261007_878'
state=json.loads((c/'campaign.json').read_text());exit=json.loads((j/'EXIT.json').read_text());files={}
for folder in (c,j):
    for f in folder.rglob('*'):
        if f.is_file() and f.suffix in ('.json','.jsonl','.log','.py'):files[str(f.relative_to(root))]=f.read_text()
out=root/'trained-model/incremental_role_objective_m0_v1_20261007_878_m0_md_batch_ratio_RGBNT201'
for f in out.iterdir():
    if f.is_file() and f.suffix in ('.json','.jsonl'):files[str(f.relative_to(root))]=f.read_text()
receipt=json.loads((out/'training.json').read_text());steps=[json.loads(l) for l in (out/'training_steps.jsonl').read_text().splitlines()]
qk={n:sum(r['incremental_isolated_query_key_gradient_norms'][n] for r in steps) for n in steps[0]['incremental_isolated_query_key_gradient_norms']}
new=[json.loads(l) for l in (out/'training_batch_order.jsonl').read_text().splitlines()][:8]
old=[json.loads(l) for l in (root/'trained-model/global_task_role_v1_20261004_824_m0_semantic_RGBNT201/training_batch_order.jsonl').read_text().splitlines()][:8]
print(json.dumps(dict(exit=exit,state=state,receipt=receipt,files=files,qk=qk,
    c_norm_sum=sum(r['incremental_isolated_correction_gradient_norm'] for r in steps),
    g_absent=all(r['incremental_isolated_shared_global_gradient_absent'] for r in steps),
    old_new_batch_equal=old==new,first_new_batch=new[0],first_old_batch=old[0],
    retained_probe_bytes=(out/'m0_reload_probe.pth').stat().st_size if (out/'m0_reload_probe.pth').exists() else None)))
