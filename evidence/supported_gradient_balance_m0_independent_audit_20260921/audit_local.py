from pathlib import Path
import hashlib,json,sys,shutil,datetime

ROOT=Path(__file__).parent
REPO=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
INTAKE=Path('D:/Program Files/UserCache/gb/codex/tmp/trifusion_supported_balance_m0_complete_20260921')
NAMES=['msvr_supported_gradient_balance.py','train_msvr_supported_gradient_balance.py','check_msvr_supported_gradient_balance_math.py','check_msvr_supported_gradient_balance.py','verify_msvr_supported_gradient_balance.py','verify_msvr_supported_gradient_balance_stats.py','run_msvr_supported_gradient_balance.py','msvr_cross_scene_smooth_ap.py','msvr_instance_memory.py','probe_msvr_history_candidate_gradients.py','probe_msvr_role_set_gradients.py','msvr_freshness_probe.py','train_msvr310_trifusion_oof.py','train_msvr310_signal_oof.py']
DOCS=['EXPERIMENT_PLAN.md','EXPERIMENT_TRACKER.md','IMPLEMENTATION_NOTES.md','EXPERIMENT_CODE_REVIEW.md','EXPERIMENT_CODE_REVIEW.json']

if sys.argv[1]=='snapshot':
    paths=[REPO/'tools'/n for n in NAMES]+[REPO/'refine-logs/msvr310_supported_gradient_balance_v1'/n for n in DOCS]
    paths += [REPO/'configs/MSVR310/TriFusion-supported-gradient-balance-paired-v1.json',REPO/'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md',REPO/'AGENTS.md']
    for d in ['supported_gradient_balance_m0_r1_failure_20260921','supported_gradient_balance_r2_review_20260921','supported_gradient_balance_r2_start_20260921']:
        paths.extend((REPO/'evidence'/d).rglob('*'))
    paths.extend(INTAKE.rglob('*'))
    rows=[]
    for p in paths:
        if not p.is_file(): continue
        label=('repo/'+p.relative_to(REPO).as_posix()) if p.is_relative_to(REPO) else ('intake/'+p.relative_to(INTAKE).as_posix())
        data=p.read_bytes()
        data.decode('utf-8-sig')
        dest=ROOT/'snapshots'/label
        dest.parent.mkdir(parents=True,exist_ok=True)
        assert not dest.exists(),str(dest)
        dest.write_bytes(data)
        rows.append(dict(source=str(p),snapshot=str(dest),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),lines=len(data.splitlines())))
    (ROOT/'snapshot_inventory.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps({'files':len(rows),'bytes':sum(x['bytes'] for x in rows),'primary_code':[x for x in rows if '/tools/' in x['snapshot'].replace('\\','/') and '/executed_snapshot/' not in x['snapshot'].replace('\\','/')]},indent=2))
elif sys.argv[1]=='read':
    blocks=[]
    for arg in sys.argv[2:]:
        name,_,span=arg.partition('@')
        p=ROOT/'snapshots'/name
        if not p.exists(): p=Path(name)
        lines=p.read_text(encoding='utf-8-sig').splitlines()
        start,end=(map(int,span.split(':')) if span else (1,len(lines)))
        blocks.append('\nFILE '+str(p)+'\n'+'\n'.join(f'{i+1}: {line}' for i,line in enumerate(lines) if start<=i+1<=end))
    text='\n'.join(blocks)
    output=ROOT/'read_outputs';output.mkdir(exist_ok=True)
    n=len(list(output.glob('*.txt')))+1
    (output/f'{n:03d}.txt').write_text(text,encoding='utf-8')
    (output/f'{n:03d}.request.json').write_text(json.dumps(sys.argv,indent=2),encoding='utf-8')
    print(text)
