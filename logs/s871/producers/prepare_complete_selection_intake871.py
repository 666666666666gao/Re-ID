from pathlib import Path
import ast
base=Path('C:/Users/gb/.codex_tmp')
s=(base/'collect_signal_selection_complete870.py').read_text(encoding='utf-8')
s=s.replace('signal_selection_complete870','signal_selection_complete871')
s=s.replace("report=root/'results/signal_selection_reference_complete_20261006_870'","report=root/'results/signal_selection_reference_complete_20261006_871'")
s=s.replace("assert json.loads((launch/'EXIT.json').read_text())['exit_code']==0","assert json.loads((launch/'EXIT.json').read_text())['exit_code']==1")
s=s.replace("state['report_exit_code']==0","state['report_exit_code']==1")
addition="""cpu=root/'logs/selection_report_completion_20261006_871'
assert json.loads((cpu/'EXIT.json').read_text())['exit_code']==0
for n in ('LAUNCH.json','CHILD.json'):
 r=json.loads((cpu/n).read_text());p=Path('/proc')/str(r['pid']);assert not p.exists() or int((p/'stat').read_text().split()[21])!=r['start_ticks']
qualification=json.loads((cpu/'QUALIFIED.json').read_text())
assert all(panel.sha(p)==d for p,d in qualification['original_immutable_sha256'].items())
assert not (root/'results/signal_selection_reference_complete_20261006_870').exists()
texts=[]
texts.extend(p for p in cpu.rglob('*') if p.is_file())
"""
assert 'texts=[]\n' in s;s=s.replace('texts=[]\n',addition,1)
s=s.replace('after original870 controller and CPU report terminate','after original870 NN and distinct871 CPU report terminate')
s=s.replace('onceCPUreport EXIT0','distinct871CPUreport EXIT0; original870EXIT1 preserved')
tree=ast.parse(s);embedded=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in n.targets));compile(embedded,'complete871_gate','exec')
out=base/'collect_signal_selection_complete871.py';assert not out.exists();out.write_text(s,encoding='utf-8')
print('AST/embedded compile only; first PowerShell builder failed before file creation, no NN or intake executed by it.')
