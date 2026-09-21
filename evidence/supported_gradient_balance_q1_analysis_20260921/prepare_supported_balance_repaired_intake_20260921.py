from pathlib import Path
import ast
t=Path('D:/Program Files/UserCache/gb/codex/tmp')
s=(t/'intake_trifusion_supported_balance_phase_20260921.py').read_text()
s=s.replace("assert mode in ('m0','q1') and not destination.exists()","assert mode == 'q1' and not destination.exists()")
s=s.replace("assert stages[mode]['exit_code']==stages[mode+'_cpu']['exit_code']==0", "assert stages['q1']['exit_code']==0 and stages['q1_cpu']['exit_code']==1\nassert pipeline['status']=='STOPPED_AT_Q1_CPU'\nrecheck=root/'q1_cpu_arithmetic_recheck'\nassert json.loads((recheck/'receipt.json').read_bytes())['exit_code']==0")
s=s.replace("proof=json.loads((root/(mode+'_cpu.json')).read_bytes())","proof=json.loads((recheck/'verification.json').read_bytes())")
s=s.replace("if mode=='q1':assert pipeline['status']=='COMPLETE_VERIFIED_'+summary['status']", "assert proof['model_forwards']==proof['optimizer_updates']==0")
s=s.replace("paths += [root/'t0.json',root/'t0.log',root/(mode+'.log'),root/(mode+'_cpu.json'),root/(mode+'_cpu.log')]", "paths += [root/'t0.json',root/'t0.log',root/'q1.log',root/'q1_cpu.log']\nfor name in ['q1_cpu_sqrt_recheck','q1_cpu_bound_sqrt_recheck','q1_cpu_arithmetic_recheck']:\n    paths += [p for p in (root/name).iterdir() if p.is_file() and p.suffix in ('.json','.log','.py')]")
s=s.replace("mode=mode,pipeline_snapshot=pipeline,", "mode=mode,pipeline_snapshot=pipeline,verification_path='q1_cpu_arithmetic_recheck/verification.json',recheck_receipt_path='q1_cpu_arithmetic_recheck/receipt.json',")
s=s.replace("mode=mode,files=len(value['files']),", "mode=mode,verification_path=value['verification_path'],original_pipeline_status=value['pipeline_snapshot']['status'],files=len(value['files']),")
ast.parse(s)
(t/'intake_trifusion_supported_balance_repaired_q1_20260921.py').write_text(s,encoding='utf-8')
print('Prepared explicit repaired-verification intake; original pipeline preserved.')
