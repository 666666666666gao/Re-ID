"""Exact F1 foundation and common V8 roles with a fresh full50 monotone AP stage."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'modeling'))
sys.path.insert(0, str(ROOT))

from tools.f1_monotone_initialized_model import build_model
from tools.official_three_dataset_model import sha256
from tools.official_three_dataset_data import records_for, loader_for
from tools.run_official_three_dataset_roles import read_protocol, distance_matrix, official_fused_metrics
from tools.train_f1_monotone_roles import train_f1_monotone_ap
from tools.train_official_three_dataset_roles import PLAIN_WIDTHS
from tools.run_signal_preserving_v5 import _module_state_sha256, _training_batch
from tools.train_signal_preserving_v18 import image_batch
from tools.train_msvr310_trifusion_oof import output_mapping

SCHEMA = 'trifusion-f1-legal-monotone-ap-v1'


def paired_diagnosis(reference, fused):
    import numpy as np
    before=np.asarray(reference['first_match_rank']);after=np.asarray(fused['first_match_rank'])
    delta=np.asarray(fused['average_precision'])-np.asarray(reference['average_precision'])
    masks={'kept_correct':(before==1)&(after==1),'repaired':(before>1)&(after==1),
           'new_error':(before==1)&(after>1),'still_wrong':(before>1)&(after>1)}
    groups={name:dict(count=int(mask.sum()),map_contribution_pp=float(delta[mask].sum()*100/len(before)))
            for name,mask in masks.items()}
    assert sum(x['count'] for x in groups.values())==len(before)
    assert abs(sum(x['map_contribution_pp'] for x in groups.values())-(fused['metrics']['mAP']-reference['metrics']['mAP']))<1e-10
    return dict(metric_delta_pp={k:v-reference['metrics'][k] for k,v in fused['metrics'].items()},
                rank1_repairs=groups['repaired']['count'],rank1_new_errors=groups['new_error']['count'],
                rank1_net_hits=groups['repaired']['count']-groups['new_error']['count'],ap_groups=groups)


def extract_f1(model, protocol, split):
    import torch
    parts={name:[] for name in (*PLAIN_WIDTHS,'pure_bank')}
    model.eval()
    with torch.no_grad():
        for raw in loader_for(protocol,records_for(protocol,split),training=False,method='PLAIN_V8'):
            batch=image_batch(raw[0],raw[3]) if protocol['dataset']=='RGBNT201' else _training_batch(raw)[0]
            values=output_mapping(model(batch,return_aux=True),widths=PLAIN_WIDTHS)
            values['pure_bank']=values['fused'][:,1536:]
            # F1's original forward normalizes on GPU before the scorer's second normalization.
            values['baseline_only']=torch.nn.functional.normalize(values['baseline_only'].float(),dim=1)
            for name,value in values.items():parts[name].append(value.float().cpu())
    result={name:torch.cat(values) for name,values in parts.items()}
    assert all(result[name].shape==(protocol['counts'][split],width) for name,width in PLAIN_WIDTHS.items())
    assert result['pure_bank'].shape==(protocol['counts'][split],4608)
    return result


def initialize(args, protocol):
    return build_model(protocol, args.signal_source, args.clip_weight, args.foundation_manifest, seed=args.seed)


def save_roles(path, model, binding, protocol_digest, epoch, metrics):
    import torch
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()
             if not name.startswith('baseline.')}
    torch.save(dict(schema=SCHEMA, dataset=binding['dataset'], method='F1_MONOTONE_AP',
                    foundation_checkpoint_sha256=binding['foundation_checkpoint_sha256'],
                    protocol_sha256=protocol_digest, epoch=epoch, metrics=metrics,
                    full_model_state_sha256=_module_state_sha256(model), role_state_dict=state), path)


def reload_roles(path, model, binding, protocol_digest):
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=True)
    assert payload['schema'] == SCHEMA and payload['dataset'] == binding['dataset']
    assert payload['method'] == 'F1_MONOTONE_AP' and payload['protocol_sha256'] == protocol_digest
    assert payload['foundation_checkpoint_sha256'] == binding['foundation_checkpoint_sha256']
    state = model.state_dict()
    assert set(payload['role_state_dict']) == {name for name in state if not name.startswith('baseline.')}
    state.update(payload['role_state_dict'])
    model.load_state_dict(state, strict=True)
    assert _module_state_sha256(model) == payload['full_model_state_sha256']
    return payload


def prepare(args, protocol):
    import numpy as np
    import torch
    from tools.analyze_correspondence_distances import camera_scores, scene_scores
    model, cfg, config, binding = initialize(args, protocol)
    binding['dataset'] = args.dataset
    model.eval()
    raw = next(iter(loader_for(protocol, records_for(protocol, 'query'), training=False, method='PLAIN_V8')))
    batch = image_batch(raw[0], raw[3]) if args.dataset == 'RGBNT201' else _training_batch(raw)[0]
    before = _module_state_sha256(model)
    with torch.no_grad():
        direct = model.baseline.signal(batch['images'], cam_label=batch['camera_ids'], training=False, sge=cfg.MODEL.stageName)
        wrapped = model(batch, retrieval_output='baseline_only')
        assert torch.equal(direct, wrapped)
        output_mapping(model(batch, return_aux=True), widths=PLAIN_WIDTHS)
    assert before == _module_state_sha256(model)
    parts={}
    with torch.no_grad():
        for split in ('query','gallery'):
            values=[]
            for raw in loader_for(protocol,records_for(protocol,split),training=False,method='PLAIN_V8'):
                batch=image_batch(raw[0],raw[3]) if args.dataset=='RGBNT201' else _training_batch(raw)[0]
                value=model(batch,retrieval_output='baseline_only')
                values.append(torch.nn.functional.normalize(value.float(),dim=1).cpu())
            parts[split]=torch.cat(values)
            assert parts[split].shape==(protocol['counts'][split],1536)
    distance=distance_matrix(parts['query'],parts['gallery'])
    rows=[protocol['records'][split] for split in ('query','gallery')]
    identities=[np.asarray([row['identity'] for row in records]) for records in rows]
    env='scene' if args.dataset=='MSVR310' else 'camera'
    environments=[np.asarray([row[env] for row in records]) for records in rows]
    scores=(scene_scores if env=='scene' else camera_scores)(distance.numpy(),*identities,*environments)
    assert all(abs(scores['metrics'][key]-value)<1e-5 for key,value in binding['baseline_metrics'].items())
    assert before == _module_state_sha256(model)
    result = dict(schema=SCHEMA, status='F1_INITIALIZATION_AND_FORWARD_VERIFIED', dataset=args.dataset,
                  seed=args.seed, initializer=binding, config=config,
                  direct_f1_and_wrapped_forward_bitwise_equal=True,
                  complete_baseline_metrics_check='PASS before fresh full50: actual complete query/gallery F1 retrieval',
                  complete_f1_metrics=scores['metrics'],query_count=len(rows[0]),gallery_count=len(rows[1]),
                  completed_at=datetime.now().astimezone().isoformat())
    args.output_dir.mkdir()
    (args.output_dir/'initialization.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


def train(args, protocol):
    import torch
    args.output_dir.mkdir()
    model, _cfg, config, binding = initialize(args, protocol)
    binding['dataset'] = args.dataset
    receipt = dict(schema=SCHEMA, dataset=args.dataset, method='F1_MONOTONE_AP', mode=args.mode, status='RUNNING',
                   seed=args.seed, foundation_training_epochs=50, role_training_epochs=50,
                   initializer=binding, protocol=str(args.protocol), protocol_sha256=sha256(args.protocol),
                   config=config, checkpoint_policy='best_official_map',
                   commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   started_at=datetime.now().astimezone().isoformat())
    path = args.output_dir/'training.json'
    path.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    best = dict(epoch=None, mAP=-1.)

    def select_epoch(model, epoch):
        # Existing PLAIN_V8 extraction selects the 1536/6144 widths; training remains R2.
        values = official_fused_metrics(model, protocol, 'PLAIN_V8', args.signal_source)
        row = dict(epoch=epoch, metrics=values)
        with (args.output_dir/'epoch_official_metrics.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row)+'\n')
        if values['mAP'] >= best['mAP']:
            save_roles(args.output_dir/'roles_best_map.pth', model, binding, receipt['protocol_sha256'], epoch, values)
            best.update(epoch=epoch, mAP=values['mAP'], metrics=values, model_state_sha256=_module_state_sha256(model))
        print(json.dumps(dict(event='f1_r2_epoch_selection', dataset=args.dataset, **row)), flush=True)

    result = train_f1_monotone_ap(model, protocol, records_for(protocol, 'train'), config, m0=args.mode == 'm0',
                         directory=args.output_dir, seed=args.seed, training_epochs=50,
                         on_epoch_end=select_epoch if args.mode == 'train' else None)
    receipt['training'] = result
    receipt['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    receipt['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    if args.mode == 'm0':
        assert result['epochs'] == 1 and result['optimizer_steps'] == 8
        probe = args.output_dir/'m0_roles_probe.pth'
        save_roles(probe, model, binding, receipt['protocol_sha256'], 0, {})
        fresh, _cfg, _config, fresh_binding = initialize(args, protocol)
        fresh_binding['dataset'] = args.dataset
        assert fresh_binding == binding
        reload_roles(probe, fresh, binding, receipt['protocol_sha256'])
        model.eval();fresh.eval()
        raw = next(iter(loader_for(protocol, records_for(protocol, 'query'), training=False, method='PLAIN_V8')))
        batch = image_batch(raw[0], raw[3]) if args.dataset == 'RGBNT201' else _training_batch(raw)[0]
        with torch.no_grad():
            a=model(batch);b=fresh(batch)
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)
        receipt.update(status='M0_PASS', reload_max_abs_difference=float((a-b).abs().max()))
    else:
        assert result['epochs'] == 50 and [x['epoch'] for x in result['history']] == list(range(1, 51))
        assert best['epoch'] is not None
        receipt.update(status='BEST_OFFICIAL_MAP_TRAINING_COMPLETE', selected_epoch=best['epoch'],
                       selected_metrics=best['metrics'], checkpoint=str(args.output_dir/'roles_best_map.pth'),
                       checkpoint_sha256=sha256(args.output_dir/'roles_best_map.pth'),
                       checkpoint_state_sha256=best['model_state_sha256'])
    receipt['completed_at']=datetime.now().astimezone().isoformat()
    path.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt), flush=True)


def evaluate(args, protocol):
    import numpy as np
    import torch
    from tools.analyze_correspondence_distances import camera_scores, scene_scores
    started = time.perf_counter()
    summary = json.loads((args.output_dir/'training.json').read_bytes())
    assert summary['status'] == 'BEST_OFFICIAL_MAP_TRAINING_COMPLETE'
    assert summary['dataset'] == args.dataset and summary['seed'] == args.seed
    history=[json.loads(x) for x in (args.output_dir/'epoch_official_metrics.jsonl').read_text().splitlines()]
    assert [x['epoch'] for x in history] == list(range(1,51))
    chosen=max(history, key=lambda x:(x['metrics']['mAP'], x['epoch']))
    assert chosen['epoch'] == summary['selected_epoch'] and chosen['metrics'] == summary['selected_metrics']
    assert sha256(summary['checkpoint']) == summary['checkpoint_sha256']
    model,_cfg,_config,binding=initialize(args,protocol);binding['dataset']=args.dataset
    assert binding == summary['initializer'] and sha256(args.protocol) == summary['protocol_sha256']
    payload=reload_roles(Path(summary['checkpoint']),model,binding,summary['protocol_sha256'])
    assert payload['epoch'] == chosen['epoch'] and payload['metrics'] == chosen['metrics']
    assert payload['full_model_state_sha256'] == summary['checkpoint_state_sha256']
    query=extract_f1(model,protocol,'query');gallery=extract_f1(model,protocol,'gallery')
    assert _module_state_sha256(model) == summary['checkpoint_state_sha256']
    qrows,grows=protocol['records']['query'],protocol['records']['gallery']
    arrays={key:[np.asarray([row[key] for row in rows]) for rows in (qrows,grows)] for key in ('identity','camera','scene')}
    from utils import metrics as upstream
    assert Path(upstream.__file__).resolve() == (args.signal_source/'utils/metrics.py').resolve()
    scores={};distances={};os.chdir(args.output_dir)
    for name in (*PLAIN_WIDTHS,'pure_bank'):
        d=distance_matrix(query[name],gallery[name]);distances[name]=d
        env='scene' if args.dataset=='MSVR310' else 'camera'
        scorer=scene_scores if env=='scene' else camera_scores
        values=scorer(d.numpy(),*arrays['identity'],*arrays[env]);scores[name]=values
        if env=='scene':
            cmc,ap=upstream.eval_func_msrv(d.numpy(),*arrays['identity'],*arrays['camera'],*arrays['scene'])
        else:
            cmc,ap=upstream.eval_func(d.numpy(),*arrays['identity'],*arrays['camera'])
        assert abs(values['metrics']['mAP']-100*float(ap))<1e-10
        assert all(abs(values['metrics'][f'Rank-{k}']-100*float(cmc[k-1]))<1e-5 for k in (1,5,10))
    assert all(abs(scores['baseline_only']['metrics'][k]-v)<1e-5 for k,v in binding['baseline_metrics'].items())
    assert all(abs(scores['fused']['metrics'][k]-v)<1e-5 for k,v in chosen['metrics'].items())
    f1_diagnosis=paired_diagnosis(scores['baseline_only'],scores['fused'])
    reference=next(row for row in json.loads(args.signal_references.read_bytes())['rows'] if row['dataset']==args.dataset)
    assert reference['protocol_sha256']==summary['protocol_sha256'] and reference['filter']==protocol['filter']
    assert sha256(reference['metrics_path'])==reference['metrics_sha256']
    assert sha256(reference['distance_path'])==reference['distance_sha256']
    old=json.loads(Path(reference['metrics_path']).read_bytes())
    assert old['status']=='COMPLETE' and old['author_checkpoint_sha256']==reference['author_checkpoint_sha256']
    cached=torch.load(reference['distance_path'],map_location='cpu',weights_only=False)
    assert cached['protocol_sha256']==summary['protocol_sha256']
    for key,values in arrays.items():
        plural={'identity':'ids','camera':'cameras','scene':'scenes'}[key]
        assert np.array_equal(cached['query_'+plural],values[0]) and np.array_equal(cached['gallery_'+plural],values[1])
    env='scene' if args.dataset=='MSVR310' else 'camera'
    signal=(scene_scores if env=='scene' else camera_scores)(cached['distances']['baseline_only'].numpy(),*arrays['identity'],*arrays[env])
    assert signal==old['outputs']['baseline_only'] and signal['metrics']==reference['metrics']
    signal_diagnosis=paired_diagnosis(signal,scores['fused'])
    target=args.output_dir/'official_distances.pt'
    torch.save(dict(distances=distances,query_ids=arrays['identity'][0],gallery_ids=arrays['identity'][1],
                    query_cameras=arrays['camera'][0],gallery_cameras=arrays['camera'][1],
                    query_scenes=arrays['scene'][0],gallery_scenes=arrays['scene'][1],
                    protocol_sha256=summary['protocol_sha256']),target)
    result=dict(schema=SCHEMA,status='COMPLETE',dataset=args.dataset,method='F1_MONOTONE_AP',seed=args.seed,
                training_epochs=50,foundation_training_epochs=50,selected_epoch=chosen['epoch'],
                checkpoint_policy='best_official_map',foundation_checkpoint_sha256=binding['foundation_checkpoint_sha256'],
                role_checkpoint_sha256=summary['checkpoint_sha256'],model_state_sha256=summary['checkpoint_state_sha256'],
                protocol_sha256=summary['protocol_sha256'],outputs=scores,query_count=len(qrows),gallery_count=len(grows),
                filter=protocol['filter'],reranking=False,distance_arrays=str(target),distance_arrays_sha256=sha256(target),
                independent_upstream_metrics_equal=True,baseline_f1_metrics_equal=True,
                baseline_retrieval_normalization='Match original F1 GPU normalization followed by scorer normalization',
                fused_vs_f1=f1_diagnosis,fused_vs_full_signal=signal_diagnosis,
                full_signal_reference=reference,pure_bank_in_same_encoding=True,
                evaluation_seconds=time.perf_counter()-started,completed_at=datetime.now().astimezone().isoformat())
    (args.output_dir/'official_metrics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='COMPLETE',dataset=args.dataset,metrics=scores['fused']['metrics'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=('RGBNT201','RGBNT100','MSVR310'),required=True)
    parser.add_argument('--mode',choices=('prepare','m0','train','evaluate'),required=True)
    for name in ('protocol','signal-source','clip-weight','foundation-manifest','signal-references','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args();assert args.seed==42
    for name in ('protocol','signal_source','clip_weight','foundation_manifest','signal_references','output_dir'):
        setattr(args,name,getattr(args,name).resolve())
    assert not (args.output_dir/'official_metrics.json').exists()
    if args.mode != 'evaluate':assert not args.output_dir.exists()
    protocol=read_protocol(args.protocol,args.dataset)
    {'prepare':prepare,'m0':train,'train':train,'evaluate':evaluate}[args.mode](args,protocol)


if __name__=='__main__':main()
