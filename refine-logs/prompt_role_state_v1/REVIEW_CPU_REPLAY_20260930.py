"""Read-only CPU replay. Imports no project code or project/author metric functions."""
import collections
import datetime
import hashlib
import json
import pathlib
import re
import numpy as np
import torch

torch.set_num_threads(2)
ROOT = pathlib.Path('/data/gaob/Re-ID/Trifusion')
PANEL = ROOT/'logs/prompt_role_state_20260930'
METRICS = ('mAP','Rank-1','Rank-5','Rank-10')
DATASETS = ('RGBNT201','RGBNT100','MSVR310')
VARIANTS = ('reset_roles','carry_roles')
hashes = {}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            h.update(data)
    digest = h.hexdigest()
    hashes[str(path)] = digest
    return digest

def read(path):
    sha(path)
    return json.loads(path.read_text())

def score(matrix, qids, gids, qenv, genv):
    """AP from relevant-item ranks; CMC from the first relevant rank."""
    ap, first, valid = [], [], []
    for q in range(len(qids)):
        order = np.argsort(matrix[q])
        allowed = ~((gids[order] == qids[q]) & (genv[order] == qenv[q]))
        order = order[allowed]
        positive_ranks = np.flatnonzero(gids[order] == qids[q]) + 1
        if len(positive_ranks) == 0:
            continue
        ap.append(float(np.mean(np.arange(1, len(positive_ranks)+1) / positive_ranks)))
        first.append(int(positive_ranks[0]))
        valid.append(q)
    first, ap = np.array(first), np.array(ap)
    rank_hits = [int(np.sum(first <= k)) for k in (1,5,10)]
    precise = {'mAP':100*float(np.mean(ap)), **{f'Rank-{k}':100*hit/len(ap) for k,hit in zip((1,5,10),rank_hits)}}
    source_precision = {'mAP':precise['mAP'], **{f'Rank-{k}':100*float(np.float32(hit)/np.float32(len(ap))) for k,hit in zip((1,5,10),rank_hits)}}
    return {'metrics_float64':precise,'metrics_source_cmc_float32':source_precision,
            'valid_queries':len(ap),'skipped_queries':len(qids)-len(ap),'rank_hits':rank_hits,
            'per_query_ap':ap.tolist(),'first_positive_ranks':first.tolist(),'valid_indices':valid}

manifest = read(PANEL/'manifest.json')
source_mismatches = {name:{'expected':digest,'actual':sha(ROOT/name)}
                     for name,digest in manifest['source_sha256'].items()
                     if sha(ROOT/name) != digest}
assert source_mismatches == {}
assert len(manifest['jobs']) == 6 and manifest['seed'] == 42 and manifest['epochs'] == 50
campaign = read(PANEL/'campaign.json')
assert campaign['status'] == 'COMPLETE'
assert all(j['status'] == 'COMPLETE' and j['exit_code'] == 0 for j in campaign['jobs'])
accepted = read(PANEL/'accepted_matrix.json')
assert accepted['verified_complete'] == 6
report = {'schema':'fresh-prompt-role-state-cpu-audit-v1','generated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'read_only':True,'cuda_visible_devices':'','project_imports':[],
          'source_manifest_count':len(manifest['source_sha256']),'source_mismatches':source_mismatches,
          'datasets':{},'runs':[],'pairs':[]}
protocols = {}
for dataset in DATASETS:
    protocol_path = ROOT/f'logs/official_three_dataset_protocols_20260923/{dataset}.json'
    protocol = read(protocol_path)
    protocols[dataset] = protocol
    data_root = pathlib.Path(protocol['dataset_root'])
    split_checks = {}
    for split in ('train','query','gallery'):
        rows = protocol['records'][split]
        assert len(rows) == protocol['counts'][split]
        assert [r['index'] for r in rows] == list(range(len(rows)))
        pathset = set()
        for row in rows:
            name = pathlib.Path(row['paths'][0]).name
            if dataset == 'RGBNT201':
                pid,cam = int(name.split('_')[0][:6]),int(name.split('_')[1][3])-1
                scene,view = cam,-1
            elif dataset == 'RGBNT100':
                pid,cam = map(int,re.match(r'([-\d]+)_c([-\d]+)',name).groups())
                cam -= 1
                scene,view = cam,-1
            else:
                pid,cam,scene = int(name[:4]),int(name[11]),int(name[6:9])
                view = scene
            assert (pid,cam,scene,view) == tuple(row[k] for k in ('identity','camera','scene','view'))
            if split == 'train':
                assert row['label'] == protocol['train_label_map'][str(pid)]
            else:
                assert row['label'] is None
            for p in row['paths']:
                f = data_root/p
                assert f.is_file() and f.stat().st_size > 0
                pathset.add(p)
        dirname = protocol[f'{split}_split']
        if dataset == 'RGBNT100':
            files = list((data_root/'rgbir'/dirname).glob('*.jpg'))
        else:
            files = list((data_root/dirname).rglob('*.jpg'))
        expected_paths = set(str(p.relative_to(data_root)) for p in files)
        assert pathset == expected_paths
        split_checks[split] = {'records':len(rows),'unique_identities':len({r['identity'] for r in rows}),
                               'image_paths':len(pathset),'filesystem_paths':len(expected_paths),
                               'all_metadata_filename_derived':True,'complete_split':True}
    train_ids = {r['identity'] for r in protocol['records']['train']}
    eval_ids = {r['identity'] for s in ('query','gallery') for r in protocol['records'][s]}
    assert not train_ids & eval_ids
    report['datasets'][dataset] = {'counts':protocol['counts'],'split_checks':split_checks,
        'identity_train_eval_overlap':0,'environment_key':protocol['environment_key'],
        'protocol_sha256':hashes[str(protocol_path)]}

for variant in VARIANTS:
    for dataset in DATASETS:
        stem = f'prompt_role_state_20260930_prompt_{variant}_{dataset}'
        run = ROOT/f'trained-model/{stem}_seed42_full'
        probe_dir = ROOT/f'trained-model/{stem}_seed42_m0'
        child = read(PANEL/stem/'campaign.json')
        assert child['status'] == 'COMPLETE'
        assert [j['mode'] for j in child['jobs']] == ['m0','train','evaluate']
        assert all(j['status'] == 'COMPLETE' and j['exit_code'] == 0 for j in child['jobs'])
        tr,of,m0 = (read(p) for p in (run/'training.json',run/'official_metrics.json',probe_dir/'training.json'))
        assert tr['status'] == 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and of['status'] == 'COMPLETE' and m0['status'] == 'M0_PASS'
        assert tr['epochs'] == of['training_epochs'] == m0['epochs'] == 50
        assert tr['seed'] == of['seed'] == m0['seed'] == 42
        assert tr['initializer'] == m0['initializer']
        assert tr['initializer']['prompt_mode'] == of['prompt_mode'] == variant.split('_')[0]
        assert [r['epoch'] for r in tr['history']] == list(range(1,51))
        best = max(tr['history'], key=lambda r:(r['official_fused']['mAP'],r['epoch']))
        assert best['epoch'] == tr['best_epoch'] == of['selected_epoch']
        assert tr['checkpoint_policy'] == 'best_official_map' and of['reranking'] is False
        assert sorted(p.name for p in run.glob('*.pth')) == ['best_map.pth']
        assert sha(run/'best_map.pth') == of['checkpoint_sha256']
        assert sha(run/'official_distances.pt') == of['distance_sha256']
        assert sha(probe_dir/'m0_reload_probe.pth') == m0['m0']['reload_probe_sha256']
        baseline = pathlib.Path(tr['initializer']['author_checkpoint'])
        baseline_hash = hashes[str(baseline)] if str(baseline) in hashes else sha(baseline)
        assert baseline_hash == of['baseline_sha256'] == tr['initializer']['author_checkpoint_sha256']
        cp = torch.load(run/'best_map.pth',map_location='cpu',weights_only=True)
        probe = torch.load(probe_dir/'m0_reload_probe.pth',map_location='cpu',weights_only=True)
        for payload in (cp,probe):
            assert payload['schema'] == 'trifusion-prompt-role-state-v1'
            assert payload['dataset'] == dataset and payload['seed'] == 42
            assert payload['prompt_mode'] == of['prompt_mode']
            assert payload['baseline_sha256'] == baseline_hash
            assert payload['protocol_sha256'] == report['datasets'][dataset]['protocol_sha256']
            assert payload['variants'] == {'m1':True,'m2':True,'m3':False}
            assert payload['condition'] == {'query_mode':'context','auxiliary_target':'none'}
            assert not any(k.startswith(('backbone.signal.','teacher.')) for k in payload['state'])
            assert all(torch.isfinite(v).all().item() for v in payload['state'].values())
        assert cp['epoch'] == best['epoch'] and probe['epoch'] == 0
        assert set(cp['state']) == set(probe['state'])
        assert all(cp['state'][k].shape == probe['state'][k].shape for k in cp['state'])
        assert m0['m0']['frozen_signal_unchanged'] and m0['m0']['reload_max_abs_difference'] <= 1e-5
        assert m0['m0']['trainable_parameters'] == m0['m0']['nonzero_gradient_parameters']
        for k in METRICS:
            assert abs(cp['metrics'][k]-of['metrics'][k]) < 1e-10
            assert abs(best['official_fused'][k]-of['metrics'][k]) < 1e-10
        loss_checks = {}
        for label, directory, receipt in (('full',run,tr),('m0',probe_dir,m0)):
            sha(directory/'training_steps.jsonl')
            steps = [json.loads(x) for x in (directory/'training_steps.jsonl').read_text().splitlines()]
            grouped = collections.defaultdict(list)
            for step in steps:
                grouped[step['epoch']].append(step)
                assert all(np.isfinite(step[k]) for k in ('loss','id','triplet','auxiliary_id'))
                assert step['auxiliary_target'] == 'none' and step['auxiliary_id'] == 0
            for h in receipt['history']:
                group = grouped[h['epoch']]
                assert len(group) == h['steps']
                assert [s['batch'] for s in group] == list(range(len(group)))
                assert abs(np.mean([s['loss'] for s in group])-h['mean_loss']) < 1e-10
            error = max(abs(s['loss']-s['id']-s['triplet']-s['auxiliary_id']) for s in steps)
            assert error < 1e-5
            if label == 'm0':
                assert len(steps) == 8 and list(grouped) == [1]
            loss_checks[label] = {'steps':len(steps),'epochs':len(grouped),'loss_reconstruction_max_abs':error,
                'nonzero_triplet_steps':sum(s['triplet'] != 0 for s in steps),
                'nonzero_auxiliary_steps':sum(s['auxiliary_id'] != 0 for s in steps)}
        arrays = torch.load(run/'official_distances.pt',map_location='cpu',weights_only=False)
        protocol = protocols[dataset]
        for split in ('query','gallery'):
            for key,field in (('ids','identity'),('cameras','camera'),('scenes','scene')):
                assert np.array_equal(arrays[f'{split}_{key}'],np.array([r[field] for r in protocol['records'][split]]))
        env = 'scenes' if dataset == 'MSVR310' else 'cameras'
        paths = {}
        for name in ('fused','shared_global','joint_local'):
            matrix = arrays[name].numpy()
            assert matrix.shape == (protocol['counts']['query'],protocol['counts']['gallery'])
            assert np.isfinite(matrix).all()
            replay = score(matrix,arrays['query_ids'],arrays['gallery_ids'],arrays[f'query_{env}'],arrays[f'gallery_{env}'])
            expected = of['metrics'] if name == 'fused' else of['diagnostic_metrics'][name]
            replay['shape'] = list(matrix.shape)
            replay['reported_metrics'] = expected
            replay['differences_float64_pp'] = {k:abs(replay['metrics_float64'][k]-expected[k]) for k in METRICS}
            replay['differences_source_cmc_float32_pp'] = {k:abs(replay['metrics_source_cmc_float32'][k]-expected[k]) for k in METRICS}
            assert max(replay['differences_source_cmc_float32_pp'].values()) < 1e-10
            paths[name] = replay
        accepted_row = next(r for r in accepted['rows'] if (r['dataset'],r['variant']) == (dataset,variant))
        assert accepted_row['metrics'] == of['metrics'] and accepted_row['diagnostic_metrics'] == of['diagnostic_metrics']
        assert accepted_row['checkpoint_sha256'] == of['checkpoint_sha256']
        assert accepted_row['distance_sha256'] == of['distance_sha256']
        assert accepted_row['receipt_sha256'] == hashes[str(run/'official_metrics.json')]
        for field,path in (('model_source_sha256','modeling/trifusion/correspondence_roles.py'),
                           ('prompt_source_sha256','modeling/trifusion/prompt_role_state.py'),
                           ('entry_sha256','tools/run_prompt_role_state.py'),
                           ('reused_context_entry_sha256','tools/run_correspondence_context_identity.py'),
                           ('context_source_sha256','modeling/trifusion/correspondence_context_identity.py'),
                           ('evidence_source_sha256','modeling/trifusion/correspondence_evidence_readout.py'),
                           ('reused_entry_sha256','tools/run_correspondence_roles.py')):
            assert tr['initializer'][field] == manifest['source_sha256'][path]
        report['runs'].append({'dataset':dataset,'variant':variant,'run_dir':str(run),'best_epoch':best['epoch'],
            'checkpoint_sha256':of['checkpoint_sha256'],'distance_sha256':of['distance_sha256'],
            'receipt_sha256':hashes[str(run/'official_metrics.json')],
            'initializer':tr['initializer'],'m0':m0['m0'],'state_tensor_keys':len(cp['state']),
            'loss_checks':loss_checks,'metrics':of['metrics'],'diagnostic_metrics':of['diagnostic_metrics'],
            'final_epoch_metrics':tr['history'][-1]['official_fused'],'paths':paths})
        del cp,probe,arrays

for dataset in DATASETS:
    reset,carry = (next(r for r in report['runs'] if r['dataset']==dataset and r['variant']==v) for v in VARIANTS)
    a,b = reset['initializer'].copy(),carry['initializer'].copy()
    a.pop('prompt_mode'); b.pop('prompt_mode')
    assert a == b
    qids = np.asarray([r['identity'] for r in protocols[dataset]['records']['query']])
    rp,cp = reset['paths']['fused'],carry['paths']['fused']
    assert rp['valid_indices'] == cp['valid_indices']
    rap,cap = np.array(rp['per_query_ap']),np.array(cp['per_query_ap'])
    rh,ch = np.array(rp['first_positive_ranks'])==1,np.array(cp['first_positive_ranks'])==1
    delta = cap-rap
    report['pairs'].append({'dataset':dataset,'initial_bindings_equal_except_mode':True,
        'initial_model_state_sha256':a['initial_model_state_sha256'],
        'carry_minus_reset_pp':{k:carry['metrics'][k]-reset['metrics'][k] for k in METRICS},
        'diagnostic_carry_minus_reset_pp':{p:{k:carry['diagnostic_metrics'][p][k]-reset['diagnostic_metrics'][p][k] for k in METRICS} for p in ('shared_global','joint_local')},
        'top1_repaired':int(np.sum(~rh & ch)),'top1_new_errors':int(np.sum(rh & ~ch)),
        'query_ap_up':int(np.sum(delta>0)),'query_ap_down':int(np.sum(delta<0)),'query_ap_equal':int(np.sum(delta==0)),
        'identity_mean_ap_delta_pp':float(100*np.mean([np.mean(delta[qids[rp['valid_indices']]==pid]) for pid in np.unique(qids[rp['valid_indices']])]))})
report['totals'] = {'runs':len(report['runs']),'matrices':sum(len(r['paths']) for r in report['runs']),
    'metric_values':sum(len(p['reported_metrics']) for r in report['runs'] for p in r['paths'].values()),
    'formal_steps':sum(r['loss_checks']['full']['steps'] for r in report['runs']),
    'm0_steps':sum(r['loss_checks']['m0']['steps'] for r in report['runs']),
    'max_source_precision_metric_error_pp':max(v for r in report['runs'] for p in r['paths'].values() for v in p['differences_source_cmc_float32_pp'].values()),
    'max_float64_metric_error_pp':max(v for r in report['runs'] for p in r['paths'].values() for v in p['differences_float64_pp'].values())}
report['hashes'] = hashes
print(json.dumps(report,indent=2))
