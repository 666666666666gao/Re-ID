from pathlib import Path
import ast,hashlib,json

temp=Path('D:/Program Files/UserCache/gb/codex/tmp')
base=temp/'analyze_trifusion_supported_balance_m0_20260921.py'
target=temp/'analyze_trifusion_supported_balance_q1_gradients_20260921.py'
assert not target.exists()
code=base.read_text(encoding='utf-8')
changes={
    'Summarize complete saved M0 text witnesses':'Summarize complete saved Q1 text witnesses',
    "source/'m0/summary.json'":"source/'q1/summary.json'",
    "source/'m0_cpu.json'":"source/'q1_cpu.json'",
    "source/'m0'":"source/'q1'",
    "assert summary['status']=='PASS_ENGINEERING_ONLY'":"assert summary['status'] in ('Q1_PASS','Q1_FAIL')\nassert inventory['pipeline_snapshot']['status']=='COMPLETE_VERIFIED_'+summary['status']",
    'PASS_COMPLETE_SUPPORTED_GRADIENT_BALANCE_M0':'PASS_COMPLETE_SUPPORTED_GRADIENT_BALANCE_Q1',
    "cpu['checked_training_steps']==248":"cpu['checked_training_steps']==1560",
    "==248 and len(rows)==744":"==1560 and len(rows)==4680",
    "len(references)==90":"len(references)==0",
    "status='COMPLETE_M0_SAVED_TEXT_SUMMARY',seed=42,steps=248,role_step_rows=744":"status='COMPLETE_Q1_SAVED_GRADIENT_SUMMARY',seed=42,steps=1560,role_step_rows=4680",
    "max(r['relative_l2_error'] or 0 for r in references)":"max((r['relative_l2_error'] or 0 for r in references),default=None)",
    "overfit_gates={k:v['gate'] for k,v in summary['overfit'].items()}":"scientific_status=summary['status']",
    'Saved M0 runtime witnesses and deterministic text aggregation. No independent parameter-gradient regeneration, no retrieval evidence, no attribution of AdamW update fractions. Capacity and overfit are distinct distributions; not a Q1 trajectory estimate.':'Saved complete Q1 runtime witnesses and deterministic text aggregation. No independent parameter-gradient regeneration or attribution of AdamW update fractions. Scientific status is copied from the final summary and requires full retrieval CPU verification and independent audit.',
    "writer=csv.DictWriter(file,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)":"writer=csv.DictWriter(file,fieldnames=list(data[0]) if data else ['endpoint','step','role','component']);writer.writeheader();writer.writerows(data)",
}
for old,new in changes.items():
    assert old in code,old
    code=code.replace(old,new)
ast.parse(code)
target.write_text(code,encoding='utf-8')
receipt=dict(status='PREPARED_NOT_EXECUTED_ON_Q1',base_complete_m0_script_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
    q1_script_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),ast_parse=True,
    scope='Adaptation of actual completed M0 text aggregation. Q1 requires complete six endpoints and CPU receipt; no Q1 result used or asserted. No training code/config change.')
out=temp/'supported_balance_q1_gradient_summary_preparation_20260921.json'
assert not out.exists();out.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
