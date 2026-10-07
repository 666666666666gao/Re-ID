from pathlib import Path
from datetime import datetime
import hashlib,json
root=Path('/data/gaob/Re-ID/Trifusion');campaign=root/'logs/incremental_role_objective_m0_v1_20261007_886';journal=root/'logs/incremental_role_objective_m0_launch_20261007_886'
exit=json.loads((journal/'EXIT.json').read_text());assert exit['exit_code']==1
child=json.loads((journal/'CHILD.json').read_text());assert not (Path('/proc')/str(child['pid'])).exists()
state=json.loads((campaign/'campaign.json').read_text())
files={p for folder in (campaign,journal) for p in folder.rglob('*') if p.is_file()}
rows=[]
for job in state['jobs']:
    output=root/f'trained-model/{campaign.name}_m0_{job["objective"]}_{job["dataset"]}'
    training=json.loads((output/'training.json').read_text())
    steps=[json.loads(line) for line in (output/'training_steps.jsonl').read_text().splitlines()]
    norms={name:sum(s['incremental_isolated_query_key_gradient_norms'][name] for s in steps)
        for name in steps[0]['incremental_isolated_query_key_gradient_norms']}
    rows.append(dict(dataset=job['dataset'],objective=job['objective'],status=training['status'],
        production_updates=training['production_m0_diagnostics']['effective_optimizer_updates'],
        accepted='acceptance' in job,correction_norm_sum=sum(s['incremental_isolated_correction_gradient_norm'] for s in steps),
        query_key_norm_sums=norms,qp_unused=[s['incremental_isolated_query_key_unused'] for s in steps],
        context_false=all(s['incremental_isolated_vjp_autocast_enabled'] is False for s in steps),
        probe_exists=(output/'m0_reload_probe.pth').exists()))
    files.update(p for p in output.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl','.log','.txt'))
print(json.dumps(dict(status='ORIGINAL_CHECKED_CONTROLLER_FAILURE_RECEIVED',at=datetime.now().astimezone().isoformat(),
    exit=exit,rows=rows,controller_log=(journal/'controller.log').read_text(),
    files={str(p.relative_to(root)):dict(text=p.read_text(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)})))
