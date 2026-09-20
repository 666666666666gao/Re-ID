"""Read-only supplemental integer masks, source closure and final immutability."""
from pathlib import Path
import ast
import datetime
import hashlib
import json
import subprocess
import shutil
import numpy as np

ROOT=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
RUN=Path('/root/trifusion-storage/artifacts/msvr310_cross_scene_smooth_ap_v1_seed42_d35864d')
EXEC='d35864d6411591e05c8ac3e5164ebae48063ad99'
read=lambda p:json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
summary=read(RUN/'q1/summary.json');mask_results=[]
for fold in summary['folds']:
    for end,item in fold['endpoints'].items():
        p=RUN/'q1'/f"fold_{fold['fold']}_{end}"
        rows=[json.loads(x) for x in (p/'memory_steps.jsonl').read_text().splitlines()]
        totals=dict(ignored_same_id_same_scene_positions=0,retained_same_scene_negative_positions=0,
                    positive_positions=0,negative_positions=0,ineligible_anchor_candidate_positions=0)
        with (p/'memory_distances.f32').open('rb') as stream:
            for a in rows:
                ids=np.array(a['identities']);sc=np.array(a['scenes']);mids=np.array([r['identity'] for r in a['memory']]);msc=np.array([r['scene'] for r in a['memory']]);m=len(mids)
                d=np.fromfile(stream,dtype='<f4',count=a['distance_float_count']).reshape(4,64,64+m)[0]
                dc,dm=d[:,:64],d[:,64:]
                own=ids[:,None]==ids[None,:];positive=own.copy();np.fill_diagonal(positive,False);negative=~own
                hp=np.where(positive,dc,-np.inf).max(1);hn=np.where(negative,dc,np.inf).min(1)
                mp=ids[:,None]==mids[None,:];mn=~mp
                if m:
                    mhp=np.where(mp,dm,-np.inf).max(1);mhn=np.where(mn,dm,np.inf).min(1)
                    up=np.maximum(hp,mhp);un=np.minimum(hn,mhn)
                    harder_p=int((mhp>hp).sum());harder_n=int((mhn<hn).sum())
                else:up=hp;un=hn;harder_p=harder_n=0
                computed=dict(memory_records=m,memory_positive_pairs=int(mp.sum()),memory_negative_pairs=int(mn.sum()),
                    memory_cross_scene_positive_pairs=int((mp & (sc[:,None]!=msc[None,:])).sum()),
                    memory_negative_violations_against_batch_hard_positive=int((mn & (dm<hp[:,None]+np.float32(.3))).sum()),
                    harder_positive_anchors=harder_p,harder_negative_anchors=harder_n,current_wrong_order_anchors=int((hp>=hn).sum()),
                    expanded_wrong_order_anchors=int((up>=un).sum()),expanded_hinge_positive_anchors=int((up-un+np.float32(.3)>0).sum()),
                    maximum_memory_age=max([r['age'] for r in a['memory']],default=0))
                assert all(v==a['statistics'][k] for k,v in computed.items()),(fold['fold'],end,a['step'])
                allids=np.concatenate((ids,mids));allsc=np.concatenate((sc,msc))
                same=ids[:,None]==allids[None,:];scene=sc[:,None]==allsc[None,:]
                cross=same & ~scene;eligible=cross.any(1)
                totals['ignored_same_id_same_scene_positions']+=int((same & scene).sum())
                totals['retained_same_scene_negative_positions']+=int((~same & scene).sum())
                totals['positive_positions']+=int(cross.sum());totals['negative_positions']+=int((~same).sum())
                totals['ineligible_anchor_candidate_positions']+=int((~same[eligible,:64][:,~eligible]).sum())
            assert stream.read()==b''
        mask_results.append(dict(fold=fold['fold'],endpoint=end,all_steps=len(rows),all_integer_statistics_exact=True,**totals))
todo=[ROOT/'tools'/n for n in ('run_msvr_cross_scene_smooth_ap.py','train_msvr_cross_scene_smooth_ap.py',
    'verify_msvr_cross_scene_smooth_ap.py','msvr_cross_scene_smooth_ap.py','check_msvr_cross_scene_smooth_ap.py',
    'check_msvr_cross_scene_smooth_ap_math.py','audit_msvr_paired_ranking_text.py','build_msvr310_train_oof_protocol.py','audit_vehicle_query_protocol_labels.py')]
seen=set();source_rows=[];texts={}
while todo:
    p=todo.pop(0)
    if p in seen:continue
    seen.add(p);text=p.read_text();rel=p.relative_to(ROOT).as_posix()
    code=ast.parse(text)
    blob=subprocess.run(['git','show',EXEC+':'+rel],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    actual=sha(p);original=hashlib.sha256(blob.stdout).hexdigest() if blob.returncode==0 else None
    source_rows.append(dict(path=str(p),current_sha256=actual,execution_blob_sha256=original,
                            match=actual==original,git_show_exit=blob.returncode))
    texts[str(p)]=text
    for node in ast.walk(code):
        if isinstance(node,ast.ImportFrom) and node.module:
            candidates=[node.module]+[node.module+'.'+a.name for a in node.names]
        elif isinstance(node,ast.Import):candidates=[a.name for a in node.names]
        else:continue
        for name in candidates:
            parent=ROOT/'modeling' if name.startswith('trifusion.') else ROOT
            q=parent/(name.replace('.','/')+'.py')
            if q.is_file():todo.append(q)
assert all(row['match'] for row in source_rows)
inventory=[]
for p in sorted(RUN.rglob('*')):
    if p.is_file():inventory.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
pipeline=read(RUN/'pipeline.json');pids=[pipeline['wrapper_pid']]+[r['original_pid'] for r in pipeline['stages']]
print(json.dumps(dict(status='PASS_FULL_MASK_AND_FINAL_IMMUTABILITY_INTAKE',
    observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),masks=mask_results,source_matches=source_rows,
    source_texts=texts,inventory=inventory,pipeline=pipeline,
    original_process_presence={str(pid):Path('/proc',str(pid)).exists() for pid in pids},
    output_parent_free_bytes=shutil.disk_usage(RUN.parent).free,
    model_forwards=0,optimizer_updates=0,official_test_reads=0)))
