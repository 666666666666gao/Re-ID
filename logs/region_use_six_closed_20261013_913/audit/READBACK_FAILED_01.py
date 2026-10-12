from pathlib import Path
import json
p=Path(r'C:\Users\gb\.codex_tmp\independent_evidence_draft\evidence_comparison903\review\six_results913')
a=json.loads((p/'AUDIT.json').read_text(encoding='utf-8'))
print('verdict',a['verdict'],'blockers',a['blockers'],'checks',a['deterministic_checks']['total_primary_checks'])
print('writer_failed_exit',a['failed_artifact_writer_invocations'][0]['receipt']['exit_code'])
print('writer_final_exit',json.loads((p/'WRITE_AUDIT_EXECUTION_02.json').read_text(encoding='utf-8-sig'))['exit_code'])
print('audit_files',[(name,(p/name).stat().st_size) for name in ('AUDIT.json','AUDIT.md','CHECKS.json','CHECKS_CHAIN.json','CHECKS_FINAL.json')])
print('\n'.join((p/'AUDIT.md').read_text(encoding='utf-8').splitlines()[:19]))
