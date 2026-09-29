from pathlib import Path
import hashlib
import json
import subprocess

root = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
scp = ['scp','-P','2026','-i','C:/Users/gb/.ssh/id_ed25519','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ProxyCommand=none']
remote = 'gaob@172.19.12.138:/data/gaob/Re-ID/Trifusion/'
for source,name in (('.git/cross_depth_launch_658_20260929.json','cross_depth_launch_658_20260929.json'),
                    ('.git/cross_depth_sync_658_20260929.json','cross_depth_sync_658_20260929.json'),
                    ('logs/cross_depth_role_state_20260929/manifest.json','cross_depth_manifest_658_20260929.json')):
    target = root/'logs'/name
    assert not target.exists()
    subprocess.run(scp+[remote+source,str(target)],check=True)
snapshot = json.loads((root/'logs/cross_depth_progress_659_20260929.json').read_bytes())
archived = []
for row in snapshot['rows']:
    if row['status'] != 'RUNNING':
        continue
    m0 = row['jobs'][0]
    assert m0['mode'] == 'm0' and m0['status'] == 'COMPLETE' and m0['exit_code'] == 0
    target = root/'logs'/f"cross_depth_m0_{row['variant']}_{row['dataset']}_658_20260929"
    assert not target.exists()
    target.mkdir()
    for name in ('training.json','training_steps.jsonl'):
        subprocess.run(scp+[f"gaob@172.19.12.138:{m0['output_dir']}/{name}",str(target/name)],check=True)
    training = json.loads((target/'training.json').read_bytes())
    steps = [json.loads(line) for line in (target/'training_steps.jsonl').read_text().splitlines()]
    assert training['status'] == 'M0_PASS' and training['history'][0]['steps'] == 8
    assert [r['batch'] for r in steps] == list(range(8))
    assert training['initializer'] == m0['initializer']
    assert training['initializer']['depth_mode'] == row['variant']
    assert training['m0'] == m0['m0_evidence'] and training['m0']['reload_max_abs_difference'] == 0
    archived.append({'dataset':row['dataset'],'depth_mode':row['variant'], 'run_dir':m0['output_dir'],
                     'directory':str(target),'initial_model_state_sha256':training['initializer']['initial_model_state_sha256'],
                     'files':{name:hashlib.sha256((target/name).read_bytes()).hexdigest()
                              for name in ('training.json','training_steps.jsonl')}})
assert len(archived) == 4
assert len({r['initial_model_state_sha256'] for r in archived if r['dataset']=='RGBNT201'}) == 1
path = root/'logs/cross_depth_m0_archive_658_20260929.json'
assert not path.exists()
path.write_text(json.dumps({'status':'FIRST_FOUR_REAL_M0_TEXTS_ARCHIVED','snapshot_at':snapshot['at'],
                            'archived':archived},indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'FOUR_PRODUCTION_M0_PASS_ARCHIVED','snapshot_at':snapshot['at'],
                  'source_paths':snapshot['source_paths'],'formal_complete':0}))
