from pathlib import Path
import ast
t=Path('D:/Program Files/UserCache/gb/codex/tmp');p=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
names=['analyze_supported_gradient_balance_q1_20260921.py','analyze_trifusion_supported_balance_q1_gradients_20260921.py','export_supported_balance_training_tables_20260921.py']
for name in names:
    s=(t/name).read_text()
    s=s.replace("root/'q1_cpu.json'","root/'q1_cpu_arithmetic_recheck/verification.json'").replace("source/'q1_cpu.json'","source/'q1_cpu_arithmetic_recheck/verification.json'").replace("source/(mode+'_cpu.json')","source/'q1_cpu_arithmetic_recheck/verification.json'")
    s=s.replace("assert inventory['pipeline_snapshot']['status'] in ('COMPLETE_VERIFIED_Q1_PASS','COMPLETE_VERIFIED_Q1_FAIL')","assert inventory['pipeline_snapshot']['status']=='STOPPED_AT_Q1_CPU'")
    s=s.replace("assert inventory['pipeline_snapshot']['status']=='COMPLETE_VERIFIED_'+summary['status']","assert inventory['pipeline_snapshot']['status']=='STOPPED_AT_Q1_CPU'")
    ast.parse(s);(t/name.replace('.py','_repaired.py')).write_text(s,encoding='utf-8')
s=(p/'tools/audit_msvr_paired_ranking_text.py').read_text()
s=s.replace("root/'q1_cpu.json'","root/'q1_cpu_arithmetic_recheck/verification.json'")
s=s.replace(" == pipeline['terminal_summary_sha256']","")
s=s.replace("    assert digest(root/'q1_cpu_arithmetic_recheck/verification.json') == pipeline['terminal_cpu_sha256']", "    assert json.loads((root/'q1_cpu_arithmetic_recheck/receipt.json').read_bytes())['exit_code']==0")
s=s.replace("    assert pipeline['status'] == 'COMPLETE_VERIFIED_' + summary['status']", "    assert pipeline['status'] == 'STOPPED_AT_Q1_CPU'")
s=s.replace("    assert all(stage['exit_code'] == 0 for stage in pipeline['stages'])", "    assert all(stage['exit_code']==(1 if stage['stage']=='q1_cpu' else 0) for stage in pipeline['stages'])\n    inventory=json.loads((root/'inventory.json').read_bytes())\n    for item in inventory['files']:\n        assert digest(root/item['path'])==item['sha256']")
ast.parse(s);(t/'audit_msvr_paired_ranking_text_repaired_20260921.py').write_text(s,encoding='utf-8')
print('Four explicit posthoc-receipt readers prepared; original failed pipeline remains input.')
