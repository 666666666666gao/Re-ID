"""Independent read-only evidence audit; never imports producer scoring/verifier code."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib
import json
import math
import re
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
torch.set_num_threads(2)
ROOT = Path('/data/gaob/Re-ID/Trifusion')
OUT = ROOT / '.codex_tmp/integrity678_cpu_evidence.json'
assert not OUT.exists()
report = {'reviewer': '/root/patch_memory_panel_integrity_678', 'started_at': datetime.now(timezone.utc).isoformat(),
          'gpu_use': False, 'neural_forward_executed': False, 'checks': [], 'hashes': {}, 'datasets': {}, 'endpoints': [], 'diagnostics': {}}
def sha(path):
    p = Path(path)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    value = h.hexdigest()
    report['hashes'][str(p)] = {'sha256': value, 'bytes': p.stat().st_size}
    return value
def read(path):
    sha(path)
    return json.loads(Path(path).read_text())
def ck(name, result, details=None):
    report['checks'].append({'name': name, 'status': 'PASS' if bool(result) else 'FAIL', 'details': details})
def save():
    OUT.write_text(json.dumps(report, indent=2) + '\n')
def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
campaign = ROOT / 'logs/patch_memory_roles_recovery_20261001'
matrix = read(campaign / 'accepted_matrix.json')
manifest = read(campaign / 'manifest.json')
recovery = read(campaign / 'recovery_manifest.json')
terminal = read(campaign / 'campaign.json')
ck('six_terminal_endpoints', matrix['verified_complete'] == matrix['expected_endpoints'] == len(matrix['rows']) == 6)
ck('recovery_parent_terminal', terminal['status'] == 'COMPLETE' and all(x['status'] == 'COMPLETE' and x['exit_code'] == 0 for x in terminal['jobs']))
source_results = {name: sha(ROOT / name) == expected for name, expected in manifest['source_sha256'].items()}
ck('210_runtime_files', len(source_results) == 210 and all(source_results.values()), {'count': len(source_results), 'mismatches': [p for p,v in source_results.items() if not v]})
original = Path(recovery['original_campaign'])
ck('original_failed_campaign_preserved', sha(original / 'campaign.json') == recovery['original_campaign_sha256'] and read(original / 'campaign.json')['status'] == 'FAILED')
ck('original_manifest_preserved', sha(original / 'manifest.json') == recovery['original_manifest_sha256'] == sha(campaign / 'manifest.json'))
ck('recovery_source_bound', sha(ROOT / 'refine-logs/patch_memory_roles_v1/RECOVER_PANEL_20261001.py') == recovery['recovery_script_sha256'])
ck('device_witness_bound', sha(ROOT / 'logs/patch_memory_resumed_device_check_20261001.json') == recovery['device_witness_sha256'])
baseline_names = {'RGBNT201': ('RGBNT201_PlainBaseline_50.pth','789e5e14aacd74ad122aad701389eb216ca5b4fda92687e27351a513023b4407'),
                  'RGBNT100': ('RGBNT100_PlainBaseline_30.pth','299a28bfb3e3180eeae0736cf8638cd162525dce0f2192a940b62b97f6e67dcd'),
                  'MSVR310': ('MSVR310_PlainBaseline_50.pth','69c5e71b75036d7216ece3ff84450f0052f5e70dfaba46bf73f3e1d40992bb37')}
protocols = {}
for ds,(weight,expected) in baseline_names.items():
    ck(ds + '/baseline_file_hash', sha(ROOT / 'pertrained-model' / weight) == expected)
    protocol = read(ROOT / f'logs/official_three_dataset_protocols_20260923/{ds}.json')
    protocols[ds] = protocol
    root = Path(protocol['dataset_root'])
    counts, field = protocol['counts'], protocol['environment_key']
    info = {'counts': counts, 'environment_key': field, 'dataset_root': str(root), 'identity_counts': {}, 'metadata_paths_verified': 0}
    train_ids = sorted({row['identity'] for row in protocol['records']['train']})
    test_ids = {row['identity'] for row in protocol['records']['gallery']}
    ck(ds + '/disjoint_identities', not set(train_ids) & test_ids)
    ck(ds + '/train_label_map', protocol['train_label_map'] == {str(pid): i for i,pid in enumerate(train_ids)})
    for split,rows in protocol['records'].items():
        ck(ds + '/' + split + '/count_order', len(rows) == counts[split] and [r['index'] for r in rows] == list(range(len(rows))))
        info['identity_counts'][split] = len({r['identity'] for r in rows})
        actual_paths = set()
        metadata_ok = True
        for row in rows:
            for rel in row['paths']:
                path = root / rel
                actual_paths.add(rel)
                metadata_ok &= path.is_file()
                base = path.name
                if ds == 'RGBNT201':
                    m = re.fullmatch(r'(\d+)_cam(\d+)_.+\.jpg', base)
                    pid,cam,scene = int(m[1]), int(m[2])-1, int(m[2])-1
                elif ds == 'RGBNT100':
                    m = re.fullmatch(r'(\d+)_c(\d+)_.+\.jpg', base)
                    pid,cam,scene = int(m[1]), int(m[2])-1, int(m[2])-1
                else:
                    m = re.fullmatch(r'(\d+)_s(\d+)_v(\d+)_.+\.jpg', base)
                    pid,cam,scene = int(m[1]), int(m[3]), int(m[2])
                metadata_ok &= (pid,cam,scene) == (row['identity'],row['camera'],row['scene'])
                metadata_ok &= row['label'] == (train_ids.index(pid) if split == 'train' else None)
                info['metadata_paths_verified'] += 1
        split_root = root / protocol[split + '_split'] if ds != 'RGBNT100' else root / 'rgbir' / protocol[split + '_split']
        disk_paths = {str(p.relative_to(root)) for p in split_root.rglob('*.jpg')}
        ck(ds + '/' + split + '/dataset_metadata_and_full_path_set', metadata_ok and disk_paths == actual_paths,
           {'protocol_paths': len(actual_paths), 'disk_paths': len(disk_paths), 'missing': len(actual_paths-disk_paths), 'unlisted': len(disk_paths-actual_paths)})
    q=protocol['records']['query']; g=protocol['records']['gallery']
    positives = [sum(x['identity']==r['identity'] and x[field]!=r[field] for x in g) for r in q]
    ck(ds + '/all_query_valid_positive', min(positives)>0, {'minimum': min(positives), 'maximum':max(positives)})
    report['datasets'][ds]=info
save()
print('HASH_AND_DATASET_CHECKS_COMPLETE', flush=True)
paired = {}
for row in matrix['rows']:
    ds,variant = row['dataset'],row['variant']; label=ds+'/'+variant
    run,m0run=Path(row['run_dir']),Path(row['m0_run_dir'])
    training,official,m0 = read(run/'training.json'),read(run/'official_metrics.json'),read(m0run/'training.json')
    child=read(Path(row['campaign_dir'])/'campaign.json')
    ck(label+'/terminal_jobs', child['status']=='COMPLETE' and all(x['status']=='COMPLETE' and x['exit_code']==0 for x in child['jobs']))
    ck(label+'/status', training['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and official['status']=='COMPLETE' and m0['status']=='M0_PASS')
    ck(label+'/initializer_training_M0', training['initializer']==m0['initializer'])
    ck(label+'/seed_budget', training['seed']==m0['seed']==official['seed']==42 and training['epochs']==m0['epochs']==official['training_epochs']==50)
    ck(label+'/epochs_contiguous', [x['epoch'] for x in training['history']]==list(range(1,51)))
    best = max(training['history'], key=lambda x:(x['official_fused']['mAP'],x['epoch']))
    ck(label+'/selected_epoch', best['epoch']==training['best_epoch']==official['selected_epoch']==row['best_epoch'])
    ck(label+'/same_epoch_metrics', all(abs(best['official_fused'][k]-official['metrics'][k])<1e-5 for k in official['metrics']))
    cp,dist,probe=run/'best_map.pth',run/'official_distances.pt',m0run/'m0_reload_probe.pth'
    ck(label+'/checkpoint_hash', sha(cp)==official['checkpoint_sha256']==row['checkpoint_sha256'])
    ck(label+'/distance_hash', sha(dist)==official['distance_sha256']==row['distance_sha256'])
    ck(label+'/receipt_hash', sha(run/'official_metrics.json')==row['receipt_sha256'])
    ck(label+'/probe_hash', sha(probe)==m0['m0']['reload_probe_sha256'])
    payload=torch.load(cp,map_location='cpu',weights_only=True)
    probe_payload=torch.load(probe,map_location='cpu',weights_only=True)
    for typename,pay in [('best',payload),('m0',probe_payload)]:
        ck(label+'/'+typename+'/schema_binding', pay['schema']=='trifusion-patch-memory-roles-v1' and pay['dataset']==ds and pay['seed']==42 and pay['baseline_sha256']==baseline_names[ds][1]
           and pay['protocol_sha256']==sha(ROOT/f'logs/official_three_dataset_protocols_20260923/{ds}.json') and pay['condition']=={'query_mode':'context','auxiliary_target':'none'} and pay['memory_mode']==row['memory_mode'] and pay['variants']=={'m1':True,'m2':True,'m3':False})
        ck(label+'/'+typename+'/finite_state', all(bool(torch.isfinite(x).all()) for x in pay['state'].values()))
    ck(label+'/checkpoint_metrics', payload['epoch']==best['epoch'] and all(abs(payload['metrics'][k]-official['metrics'][k])<1e-5 for k in official['metrics']))
    ck(label+'/checkpoint_key_schema', set(payload['state'])==set(probe_payload['state']) and all(payload['state'][k].shape==probe_payload['state'][k].shape for k in payload['state']))
    if row['recovery_action']=='evaluate_existing':
        ck(label+'/preserved_full50', sha(run/'training.json')==child['preserved_training_sha256'] and sha(cp)==child['preserved_checkpoint_sha256'])
    history = training['history']; sha(run/'training_steps.jsonl'); sha(m0run/'training_steps.jsonl')
    steps=[json.loads(l) for l in (run/'training_steps.jsonl').read_text().splitlines()]
    msteps=[json.loads(l) for l in (m0run/'training_steps.jsonl').read_text().splitlines()]
    ck(label+'/M0_8', len(msteps)==8 and [s['batch'] for s in msteps]==list(range(8)) and all(s['epoch']==1 for s in msteps) and len(m0['history'])==1 and m0['history'][0]['steps']==8)
    ck(label+'/M0_receipted_grad_freeze_reload', m0['m0']['nonzero_gradient_parameters']==m0['m0']['trainable_parameters']==127 and m0['m0']['frozen_signal_unchanged'] and m0['m0']['reload_max_abs_difference']<=1e-5)
    max_err=max(abs(s['loss']-math.fsum([s['id'],s['triplet'],s['auxiliary_id']])) for s in steps)
    ck(label+'/loss_algebra_finite', max_err<1e-5 and all(s['auxiliary_target']=='none' and s['auxiliary_id']==0 and all(math.isfinite(s[k]) for k in ('loss','id','triplet','auxiliary_id')) for s in steps),max_err)
    epoch_logs_ok = len(steps)==sum(h['steps'] for h in history)
    for h in history:
        group=[s for s in steps if s['epoch']==h['epoch']]
        epoch_logs_ok &= [s['batch'] for s in group]==list(range(h['steps'])) and abs(math.fsum(s['loss'] for s in group)/len(group)-h['mean_loss'])<1e-8
    ck(label+'/all_step_epoch_means',epoch_logs_ok)
    arrays=torch.load(dist,map_location='cpu',weights_only=False)
    protocol=protocols[ds]
    for split in ('query','gallery'):
        for key,field in [('ids','identity'),('cameras','camera'),('scenes','scene')]:
            ck(label+'/'+split+'/'+key+'/exact_order',np.array_equal(arrays[split+'_'+key],np.array([r[field] for r in protocol['records'][split]])))
    qid,gid=arrays['query_ids'],arrays['gallery_ids']
    field='scenes' if ds=='MSVR310' else 'cameras'
    qe,ge=arrays['query_'+field],arrays['gallery_'+field]
    rec={'dataset':ds,'variant':variant,'run_dir':str(run),'m0_run_dir':str(m0run),'best_epoch':best['epoch'],'steps':len(steps),'epoch_step_counts':sorted({h['steps'] for h in history}),
         'loss_max_error':max_err,'positive_triplet_steps':sum(s['triplet']>0 for s in steps),'M0':m0['m0'],
         'initializer':training['initializer'],'checkpoint_state_hash':state_hash(payload['state']),
         'checkpoint_state_tensors':len(payload['state']),'m0_state_hash':state_hash(probe_payload['state']), 'metrics':{},'ranking':{},
         'training_seconds':(datetime.fromisoformat(training['completed_at'])-datetime.fromisoformat(training['started_at'])).total_seconds()}
    for name in ('fused','shared_global','joint_local'):
        a=arrays[name].numpy(); finite=bool(np.isfinite(a).all()); shape=tuple(a.shape)
        ck(label+'/'+name+'/full_finite_matrix', finite and shape==(protocol['counts']['query'],protocol['counts']['gallery']))
        aps=[]; first=[]; ties=0
        for i in range(len(qid)):
            order=np.argsort(a[i])
            valid=order[~((gid[order]==qid[i])&(ge[order]==qe[i]))]
            positives=np.flatnonzero(gid[valid]==qid[i])+1
            assert positives.size>0
            aps.append(float(np.sum(np.arange(1,len(positives)+1,dtype=np.float64)/positives)/len(positives)))
            first.append(int(positives[0]))
            ties+=int(np.any(np.diff(a[i,valid])==0))
        metrics={'mAP':math.fsum(aps)/len(aps)*100,'Rank-1':sum(r<=1 for r in first)/len(first)*100,'Rank-5':sum(r<=5 for r in first)/len(first)*100,'Rank-10':sum(r<=10 for r in first)/len(first)*100}
        expected=official['metrics'] if name=='fused' else official['diagnostic_metrics'][name]
        difference={k:abs(metrics[k]-expected[k]) for k in metrics}
        ck(label+'/'+name+'/independent_ranking',max(difference.values())<1e-5,difference)
        rec['metrics'][name]=metrics;rec['ranking'][name]={'shape':list(shape),'max_metric_error_pp':max(difference.values()),'queries_with_ties':ties,'dist_min':float(a.min()),'dist_max':float(a.max())}
        if name=='fused': paired[(ds,variant)]=(np.array(aps),np.array(first),qid)
    report['endpoints'].append(rec)
    save();print('ENDPOINT_COMPLETE',label,rec['steps'],rec['metrics']['fused'],flush=True)
report['pairwise']={}
for ds in protocols:
    a,ar,q=paired[(ds,'local_memory')]; b,br,_=paired[(ds,'full_memory')]
    r=[e for e in report['endpoints'] if e['dataset']==ds]
    ck(ds+'/paired_init_and_parameters',r[0]['initializer']['initial_model_state_sha256']==r[1]['initializer']['initial_model_state_sha256'] and r[0]['initializer']['trainable_parameters']==r[1]['initializer']['trainable_parameters'])
    report['pairwise'][ds]={'delta_mAP_pp':float(np.mean(b-a)*100),'delta_R1_pp':float(np.mean(br==1)*100-np.mean(ar==1)*100),'repairs':int(((ar!=1)&(br==1)).sum()),'new_errors':int(((ar==1)&(br!=1)).sum()),'AP_improved':int(((b-a)>1e-8).sum()),'AP_worsened':int(((b-a)<-1e-8).sum())}
report['gate_pass']=all(r['delta_mAP_pp']>0 and r['delta_R1_pp']>=0 for r in report['pairwise'].values()) and report['pairwise']['RGBNT201']['delta_mAP_pp']>=.5 and report['pairwise']['MSVR310']['delta_mAP_pp']>=.5
ck('cuda_not_initialized',not torch.cuda.is_initialized())
report['completed_at']=datetime.now(timezone.utc).isoformat()
report['status']='COMPLETE' if all(c['status']=='PASS' for c in report['checks']) else 'COMPLETED_WITH_FAILURES'
save();print('FINAL',report['status'],len(report['checks']),report['pairwise'],'gate',report['gate_pass'],flush=True)
