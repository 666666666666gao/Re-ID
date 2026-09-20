from pathlib import Path
import json,datetime
root=Path('/root/trifusion-storage/artifacts/smooth_ap_source_coverage_independent_audit_20260920/attempt02')
result=dict(observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(root))
for name in ('launch.json','terminal.json'):
    p=root/name
    if p.exists():result[name]=json.loads(p.read_bytes())
if 'launch.json' in result:
    result['process_exists']={k:Path('/proc/'+str(result['launch.json'][k])).exists() for k in ('wrapper_pid','child_pid')}
p=root/'independent_verify.jsonl'
lines=p.read_text().splitlines()
rows=[json.loads(s) for s in lines]
result['stdout_lines']=len(lines)
result['completed_conditions']=[r for r in rows if r['kind']=='condition_complete']
result['last_records']=rows[-3:]
result['stderr']=(root/'independent_verify.stderr').read_text()
result['wrapper_stderr']=(root/'wrapper.stderr').read_text()
print(json.dumps(result,indent=2))
