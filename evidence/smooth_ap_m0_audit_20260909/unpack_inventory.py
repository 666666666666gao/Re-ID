import hashlib,json
from pathlib import Path
OUT=Path(__file__).resolve().parent
x=json.loads((OUT/'remote_inventory.stdout').read_bytes())
local=json.loads((OUT/'local_input_manifest.json').read_bytes())
remote='/root/autodl-tmp/trifusion-v2/TriFusion-ReID/'
repo='C:/Users/gb/.trifusion_github_publish_22c3bee/'
comparisons=[]
for p,content in x['texts'].items():
    target=OUT/'snapshots'/'remote'/p.lstrip('/')
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(content,encoding='utf-8',newline='')
for p,proof in local.items():
    name=p.replace('\\','/')
    target=None
    if name.startswith(repo):target=remote+name[len(repo):]
    intake='C:/Users/gb/.codex_tmp/smooth_ap_m0_complete_20260908/'
    if name.startswith(intake):target='/root/trifusion-storage/artifacts/msvr310_smooth_ap_v1_seed42_2e947a4/'+name[len(intake):]
    if target in x['files']:
        recorded=x['files'][target]
        comparisons.append({'local_path':p,'remote_path':target,'local_sha256':proof['sha256'],'remote_sha256':recorded['sha256'],'byte_equal':proof['sha256']==recorded['sha256']})
        if proof['sha256']!=recorded['sha256'] and target in x['texts']:
            data=(OUT/proof['snapshot']).read_bytes()
            comparisons[-1]['equal_after_CRLF_to_LF']=data.replace(b'\r\n',b'\n')==x['texts'][target].encode()
summary={'remote_files':len(x['files']),'remote_texts':len(x['texts']),'recursive_bindings':len(x['bindings']),'binding_failures':[r for r in x['bindings'] if not r['match']],'local_remote_comparisons':comparisons,'processes':x['processes'],'remote_head':x['remote_head'],'signal_commit_match':x['signal_commit_match'],'signal_diff_match':x['signal_diff_match']}
(OUT/'inventory_check.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ('local_remote_comparisons','processes')},indent=2))
print('local_remote_mismatches',json.dumps([c for c in comparisons if not c['byte_equal']],indent=2))
for path in ('protocols/msvr310_train_oof_v1.json','evidence/msvr310_style_t0_runtime_binding_20260907/msvr_style_metadata_remote_numpy_20260907.json'):
    obj=json.loads(x['texts'][remote+path]);print(path,'keys',list(obj));print('record0',str(obj.get('records',[None])[0])[:2000]);print('fold0keys',list(obj['folds'][0]));print('fold0brief',str({k:v for k,v in obj['folds'][0].items() if k not in ('batches','record_indices','source_record_indices','query_rows','gallery_record_indices')})[:1500])
