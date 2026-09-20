from pathlib import Path
import importlib.util,json,hashlib

temp=Path('D:/Program Files/UserCache/gb/codex/tmp')
script=temp/'analyze_cross_scene_smooth_ap_q1_20260921.py'
spec=importlib.util.spec_from_file_location('cross_terminal',script)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
root=temp/'trifusion_cross_scene_m0_complete_20260921'
inventory=json.loads((root/'inventory.json').read_bytes())
for row in inventory['files']:
    data=(root/row['path']).read_bytes()
    assert hashlib.sha256(data).hexdigest()==row['sha256']
groups=[]
for folder in sorted((root/'m0').iterdir()):
    if not folder.is_dir():continue
    training=json.loads((folder/'training.json').read_bytes())
    rows=[json.loads(x) for x in (folder/'memory_steps.jsonl').read_text().splitlines()]
    result=module.analyze(rows,training['steps'])
    assert result['counts']['steps']==training['optimizer_steps']
    groups.append(dict(group=folder.name,result=result))
assert len(groups)==8 and sum(x['result']['counts']['steps'] for x in groups)==248
receipt=dict(status='PASS_ANALYSIS_FUNCTION_ON_COMPLETE_M0_TEXT',script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),input_inventory_sha256=hashlib.sha256((root/'inventory.json').read_bytes()).hexdigest(),groups=groups,steps=248,model_forwards=0,optimizer_updates=0,scope='Only reusable descriptive function on actual completed M0 text. Complete Q1 main, 260-step assertions, final retrieval and scientific gates remain unexecuted.')
output=temp/'cross_scene_terminal_analyzer_m0_check_20260921.json'
assert not output.exists();output.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=receipt['status'],groups=len(groups),steps=248)))
