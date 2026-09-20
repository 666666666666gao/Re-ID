from pathlib import Path
import json,hashlib
out=Path(__file__).parent
error=dict(stage='private_trace_packaging',error_type='AssertionError',error_text="seal_trace.py line 44: assert status==(out/'initial_git_status.txt').read_text(encoding='utf-8')",initial_git_status=(out/'initial_git_status.txt').read_text(),observed_git_status=(out/'final_git_status.txt').read_text(),observed_addition='?? evidence/roadmap_primary_read_20260920/',resolution='Preserve both shared-workspace observations; report whether equal rather than assume the entire shared worktree cannot change during a delegated audit.',scientific_checks_repeated=False,scientific_verdict_changed=False)
p=out/'trace_sealing_error.json';p.write_text(json.dumps(error,indent=2)+'\n',encoding='utf-8')
report=out/'EXPERIMENT_AUDIT.md'
with report.open('a',encoding='utf-8') as f:
    f.write('\nFinal packaging note: the first trace-sealing attempt stopped on an overly broad whole-worktree status-equality assertion because `evidence/roadmap_primary_read_20260920/` appeared concurrently. The complete error and before/after observations are preserved in `trace_sealing_error.json`. No files in that directory were changed by this auditor. Packaging now records the actual shared-workspace difference; no scientific verification was rerun and no verdict changed.\n')
p=out/'EXPERIMENT_AUDIT.json';audit=json.loads(p.read_bytes());audit['packaging_events']=[error];audit['audited_input_hashes']['trace_sealing_error.json']='sha256:'+hashlib.sha256((out/'trace_sealing_error.json').read_bytes()).hexdigest();p.write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print('Preserved packaging failure; scientific outputs unchanged.')
