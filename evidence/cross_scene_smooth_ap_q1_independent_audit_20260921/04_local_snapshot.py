"""Immutable local text snapshots and comparison with fresh remote intake."""
from pathlib import Path
import ast
import hashlib
import json
import shutil

OUT=Path(__file__).resolve().parent
REPO=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
INTAKE=Path('D:/Program Files/UserCache/gb/codex/tmp/trifusion_cross_scene_q1_complete_20260921')
REMOTE='/root/autodl-tmp/trifusion-v2/TriFusion-ReID/'
RUN='/root/trifusion-storage/artifacts/msvr310_cross_scene_smooth_ap_v1_seed42_d35864d/'
read=lambda p: json.loads(Path(p).read_bytes())
sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
data=read(OUT/'03_remote_intake.stdout')
inventory={r['path']:r for r in data['inventory']}
pending=[REPO/'tools'/n for n in (
    'run_msvr_cross_scene_smooth_ap.py','train_msvr_cross_scene_smooth_ap.py',
    'verify_msvr_cross_scene_smooth_ap.py','msvr_cross_scene_smooth_ap.py',
    'check_msvr_cross_scene_smooth_ap.py','check_msvr_cross_scene_smooth_ap_math.py',
    'audit_msvr_paired_ranking_text.py','build_msvr310_train_oof_protocol.py',
    'audit_vehicle_query_protocol_labels.py')]
pending += [REPO/'AGENTS.md',REPO/'.aris/traces/experiment-audit/2026-09-21_cross_scene_q1/001-audit-request.md']
pending += list((REPO/'refine-logs/msvr310_cross_scene_smooth_ap_v1').glob('*'))
for p,c in data['configs'].items():
    pending.append(REPO/p.removeprefix(REMOTE))
    for k in ('project_file_sha256','project_source_file_sha256'):
        pending += [REPO/n for n in c.get(k,{})]
pending += [REPO/data[k].removeprefix(REMOTE) for k in ('protocol_path','metadata_path')]
pending += [REPO/'evidence/smooth_ap_cross_scene_support_20260909/smooth_ap_cross_scene_support_20260909.json',
            REPO/'evidence/vehicle_query_protocol_labels_20260905.json',
            REPO/'evidence/msvr310_dataset_install_20260905.json']
pending += list(INTAKE.rglob('*'))
pending += [Path('D:/Program Files/UserCache/gb/codex/tmp/trifusion_cross_scene_q1_source_log_analysis_20260921.json')]
pending += list(Path('D:/Program Files/UserCache/gb/codex/tmp/trifusion_cross_scene_q1_analysis_20260921_v2').glob('*'))
seen=set();manifest=[];missing=[];parsed_code=[]
snap=OUT/'snapshots';snap.mkdir()
while pending:
    p=pending.pop(0)
    if p in seen:continue
    seen.add(p)
    if not p.exists():missing.append(str(p));continue
    if not p.is_file():continue
    assert p.suffix.lower() in ('.py','.md','.json','.jsonl','.log','.csv','.yml','.yaml','.txt','.toml','.stdout','.stderr'),p
    target=snap/(str(len(manifest)).zfill(4)+'_'+p.name)
    shutil.copyfile(p,target)
    row=dict(path=str(p),snapshot=str(target.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p))
    if p.is_relative_to(INTAKE):
        remote=RUN+p.relative_to(INTAKE).as_posix()
        if remote in inventory:
            row['remote_sha256']=inventory[remote]['sha256']
            row['remote_exact']=row['sha256']==row['remote_sha256']
            assert row['remote_exact'],p
    manifest.append(row)
    if p.suffix=='.py' and p.is_relative_to(REPO):
        tree=ast.parse(p.read_text(encoding='utf-8-sig'))
        parsed_code.append(str(p))
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.module:
                candidates=[node.module]+[node.module+'.'+a.name for a in node.names]
            elif isinstance(node,ast.Import):candidates=[a.name for a in node.names]
            else:continue
            for name in candidates:
                base=REPO/'modeling' if name.startswith('trifusion.') else REPO
                q=base/(name.replace('.','/')+'.py')
                if q.is_file():pending.append(q)
report=dict(status='PASS_LOCAL_TEXT_SNAPSHOTS',files=manifest,missing=missing,
            parsed_project_python_files=parsed_code,
            intake_text_files_matching_remote=sum('remote_exact' in r for r in manifest))
(OUT/'local_input_manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=report['status'],files=len(manifest),parsed_python=len(parsed_code),missing=missing,
                     remote_matches=report['intake_text_files_matching_remote'])))
