from pathlib import Path
import json,hashlib
root=Path(__file__).parent
d=json.loads((root/'remote_intake_02.json').read_bytes())
manifest=[]
for name,text in d['text_inputs'].items():
    p=root/'remote_text'/name.lstrip('/')
    p.parent.mkdir(exist_ok=True,parents=True)
    p.write_text(text,encoding='utf-8')
    manifest.append(dict(remote_path=name,local_path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
(root/'remote_text_inventory.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in d.items() if k not in ('text_inputs','configs','m0_inventory','registered_bindings','executed_bindings')},indent=2))
print(json.dumps(dict(registered_bindings=len(d['registered_bindings']),all_registered_match=all(r['match'] for r in d['registered_bindings']),executed_bindings=len(d['executed_bindings']),executed_mismatches=[r for r in d['executed_bindings'] if not r['match']],text_files=len(manifest),m0_files=len(d['m0_inventory'])),indent=2))
for name,c in d['configs'].items():
    print(json.dumps(dict(config=name,links={k:v for k,v in c.items() if k in ('previous_config','base_config','memory_config','coordinate_config','BASELINE','SOURCE_METADATA','protocol','signal_source','dataset_root','MODEL','LOSS','OPTIMIZATION')}),indent=2))
run='/root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_r2_seed42_1381639'
pipeline=json.loads(d['text_inputs'][run+'/pipeline.json'])
print(json.dumps(dict(pipeline=pipeline),indent=2))
