"""Remove the unnecessary /proc enumeration after its observed exit race."""
from pathlib import Path
import ast,json

private=Path('C:/Users/gb/.codex_tmp');base=private/'independent_evidence_draft'
assert json.loads((base/'historical_probe_retirement841/EXIT.json').read_bytes())['exit_code']==1
source=(private/'retire_historical_probes841.py').read_text(encoding='utf-8')
old='''owned_commands=[]
for proc in Path('/proc').iterdir():
 if proc.name.isdigit() and proc.stat().st_uid==root.stat().st_uid and (proc/'cmdline').exists():
  owned_commands.append((proc/'cmdline').read_bytes().decode('utf-8',errors='replace'))'''
assert old in source
source=source.replace(old,"active_command=' '.join(active['command'])")
source=source.replace("assert not any(str(p.parent) in cmd for cmd in owned_commands)","assert str(p.parent) not in active_command")
source=source.replace("packet=base/'historical_probe_retirement841'","packet=base/'historical_probe_retirement841b'")
source=source.replace('active process directories','current project active command directory')
ast.parse(source);target=private/'retire_historical_probes841b.py';assert not target.exists();target.write_text(source,encoding='utf-8')
registrar=private/'register_deployment_metric_rgb201_pair841.py'
source=registrar.read_text(encoding='utf-8')
source=source.replace("retiredir=base/'historical_probe_retirement841'","retiredir=base/'historical_probe_retirement841b'")
source=source.replace("native_dir,closed,m0dir,retiredir,","native_dir,closed,m0dir,retiredir,base/'historical_probe_retirement841',")
source=source.replace("'read_historical_probe_receipts841.py','retire_historical_probes841.py'","'read_historical_probe_receipts841.py','retire_historical_probes841.py','retire_historical_probes841b.py','prepare_exact_probe_retirement841b.py'")
source=source.replace('当前控制输入/活动进程目录','当前控制输入/当前项目活动命令目录')
source=source.replace('删除前可用{retired', '原首次清理因/proc进程退出竞态在preflight前停止，尚未生成远端证书或删除；保留原failure，简化为当前项目队列实际路径核对后再执行明确白名单，不新增try/except。删除前可用{retired')
ast.parse(source);registrar.write_text(source,encoding='utf-8')
print('Prepared exact whitelist retirement; original failed attempt retained and not replayed.')
