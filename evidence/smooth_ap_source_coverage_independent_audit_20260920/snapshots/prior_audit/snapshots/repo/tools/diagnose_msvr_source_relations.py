"""Complete frozen MSVR310 source census; no optimization or heldout image reads."""
import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from tools.msvr_source_relation_math import WIDTHS, STATES, VIEWS, PROTOCOLS, EXPERTS, MODALITIES, math_check
from tools.train_msvr310_signal_oof import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def contract(path):
    spec = json.loads(path.read_bytes())
    assert spec['seed'] == 42 and spec['states'] == list(STATES) and spec['views'] == list(VIEWS)
    assert spec['protocols'] == list(PROTOCOLS) and spec['optimizer_updates'] == 0
    for name, expected in spec['project_file_sha256'].items():
        assert sha256(ROOT/name) == expected, name
    for name, expected in spec['fixed_file_sha256'].items():
        assert sha256(name) == expected, name
    q1 = json.loads(Path(spec['q1_summary']).read_bytes())
    assert q1['status'] == 'Q1_FAIL' and q1['optimizer_steps'] == 1560
    assert q1['original_failed_run_files_unchanged']
    return spec, q1


def source_loader(records, augmented):
    from torch.utils.data import DataLoader
    from data.datasets.make_dataloader import train_collate_fn
    from trifusion.aligned_data import AlignedTripletImageDataset, SharedGeometryTripletTransform
    from tools.train_msvr310_signal_oof import clean_triplet
    transform = SharedGeometryTripletTransform(size=(128,256)) if augmented else clean_triplet
    return DataLoader(AlignedTripletImageDataset(records, transform=transform), batch_size=64,
                      shuffle=False, num_workers=4, collate_fn=train_collate_fn, pin_memory=True)


def output_features(output):
    import torch
    from torch.nn.functional import normalize
    from tools.train_msvr310_trifusion_oof import output_mapping
    values = output_mapping(output)
    values.update({f'{e}_{m}_residual':output.modal_residual_embeddings[e][:,j]
                   for e in EXPERTS for j,m in enumerate(MODALITIES)})
    values['pure_bank'] = torch.cat([output.residual_embeddings[e] for e in EXPERTS],dim=1)
    values.update({f'pure_{e}':output.residual_embeddings[e] for e in EXPERTS})
    assert list(values) == list(WIDTHS)
    result = {}
    for name,width in WIDTHS.items():
        x = values[name].float()
        assert x.shape[1] == width and bool(torch.isfinite(x).all()) and bool((x.norm(dim=1)>0).all())
        result[name] = normalize(x,dim=1).cpu().numpy()
    return result


def check(args):
    spec,q1 = contract(args.contract)
    protocol = json.loads((ROOT/spec['protocol']).read_bytes())
    batches = []
    for f,prior in zip(protocol['folds'],q1['folds'],strict=True):
        rows = [protocol['records'][j] for j in f['source_record_indices']]
        assert len(rows) == [672,683,709][f['fold']]
        assert set(x['identity'] for x in rows) == set(f['source_ids'])
        assert not set(f['source_ids']) & set(f['heldout_ids'])
        assert prior['endpoints']['control']['initialization'] == prior['endpoints']['source_style']['initialization']
        for j in range(0,len(rows),64):
            cameras = {r['camera'] for r in rows[j:j+64]}
            assert len(cameras)>1
            batches.append(dict(fold=f['fold'],batch=j//64,records=len(rows[j:j+64]),cameras=sorted(cameras)))
    write_json(args.root/'check.json',dict(status='PASS_MSVR_FULL_SOURCE_CONTRACT',
        contract_sha256=sha256(args.contract),math=math_check(),source_record_memberships=2064,
        ordered_batches=batches,source_unique_records=1032,model_forwards=0,optimizer_updates=0,
        planned_record_forwards=18576,planned_feature_bytes=18576*sum(WIDTHS.values())*4,
        minimum_free_bytes=4*1024**3,available_bytes=shutil.disk_usage(args.root).free))
    assert shutil.disk_usage(args.root).free >= 4*1024**3


def run(args):
    import torch
    from tools import train_msvr310_source_style as original
    from tools.train_msvr310_signal_oof import records_for

    spec,q1 = contract(args.contract)
    checked = json.loads((args.root/'check.json').read_bytes())
    assert checked['status'] == 'PASS_MSVR_FULL_SOURCE_CONTRACT' and checked['contract_sha256']==sha256(args.contract)
    config,base,cfg,environment,protocol,b0,_metadata = original.context(ROOT/spec['original_config'])
    from tools.run_signal_preserving_v5 import _set_seed, _module_state_sha256, _training_batch
    from tools.msvr310_exact_signal_inference import exact_signal_forward
    from trifusion.source_style_v27 import make_style_plan
    assert torch.get_num_threads()==4
    started = time.perf_counter()
    summary = dict(status='RUNNING',contract_sha256=sha256(args.contract),
                   code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                   environment=environment,conditions=[],source_record_forwards=0,forward_batches=0,
                   additional_visual_passes=0,optimizer_updates=0,heldout_image_reads=0,official_image_reads=0,
                   states=list(STATES),views=list(VIEWS),widths=WIDTHS,protocols=list(PROTOCOLS))
    def save():
        write_json(args.root/'summary.json',summary)
    save()
    for fold,basefold,fixedfold in zip(protocol['folds'],b0['folds'],q1['folds'],strict=True):
        records = records_for(base,protocol,fold,True)
        record_indices = fold['source_record_indices']
        expected_pixels = {}
        baseline_arrays = {}
        for state in STATES:
            binding = fixedfold['endpoints']['control']['initialization']
            if state=='initial':
                model,actual = original.build(config,cfg,fold,basefold)
                assert actual==binding
            else:
                old = fixedfold['endpoints']['control' if state=='control_final' else 'source_style']
                model = original.reload_model(config,cfg,fold,basefold,old['checkpoint'],binding,
                                              old['training']['final_state_sha256'],sha256(ROOT/spec['original_config']))
            model.eval()
            fixed_state = _module_state_sha256(model)
            torch.cuda.reset_peak_memory_stats()
            for view in VIEWS:
                condition_started = time.perf_counter()
                directory = args.root/f'fold_{fold["fold"]}_{state}_{view}'
                directory.mkdir()
                _set_seed(42)
                model.eval()
                if view=='coupled_style':
                    model.baseline.train(True)
                model.baseline.style_enabled = view=='coupled_style'
                model.baseline.style_plan = None
                parts = {name:[] for name in WIDTHS}
                seen = []
                pixels = []
                extra = 0
                with (directory/'inputs.jsonl').open('x',encoding='utf-8') as stream:
                    for bi,raw in enumerate(source_loader(records,view!='clean')):
                        indices = record_indices[64*bi:64*(bi+1)]
                        assert list(raw[-1]) == [Path(protocol['records'][j]['paths'][0]).name for j in indices]
                        assert raw[1].tolist() == [fold['source_label_map'][str(protocol['records'][j]['identity'])] for j in indices]
                        batch,_ = _training_batch(raw)
                        plan = make_style_plan(raw[2].numpy(),fold=fold['fold'],step=bi,force_active=True)
                        if view=='coupled_style':
                            model.baseline.style_plan = plan
                        pixel = original.pixels(batch)
                        values = output_features(exact_signal_forward(model,batch))
                        stats = dict(model.baseline.last_style_stats)
                        assert stats['style_active'] == int(view=='coupled_style')
                        extra += stats['additional_visual_passes']
                        if state!='initial':
                            assert np.array_equal(values['baseline_only'],baseline_arrays[view][64*bi:64*(bi+1)])
                        for name,x in values.items():
                            parts[name].append(x)
                        seen.extend(indices);pixels.append(pixel)
                        stream.write(json.dumps(dict(batch=bi,record_indices=indices,pixel_sha256=pixel,
                            style_plan=plan if view=='coupled_style' else None,statistics=stats))+'\n')
                        stream.flush()
                        summary['source_record_forwards']+=len(indices)
                        summary['forward_batches']+=1
                assert seen==record_indices
                pixel_key = 'clean' if view=='clean' else 'augmented'
                if state=='initial' and view!='coupled_style':
                    expected_pixels[pixel_key]=pixels
                assert expected_pixels[pixel_key]==pixels
                arrays = {name:np.concatenate(xs) for name,xs in parts.items()}
                if state=='initial':
                    baseline_arrays[view]=arrays['baseline_only'].copy()
                if view=='coupled_style':
                    assert np.array_equal(arrays['baseline_only'],baseline_arrays['augmented'])
                for name,x in arrays.items():
                    assert x.shape==(len(records),WIDTHS[name]) and x.dtype==np.float32
                    np.save(directory/f'{name}.npy',x,allow_pickle=False)
                assert _module_state_sha256(model)==fixed_state
                assert all(p.grad is None for p in model.parameters())
                files={p.name:dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in directory.iterdir()}
                receipt=dict(fold=fold['fold'],state=state,view=view,records=len(records),record_indices=record_indices,
                             model_state_sha256=fixed_state,model_state_unchanged=True,gradients_absent=True,
                             raw_pixels_match_other_states=True,augmented_pixels_match_style=True,
                             source_ids=fold['source_ids'],heldout_ids=fold['heldout_ids'],files=files,
                             additional_visual_passes=extra,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
                             elapsed_seconds=time.perf_counter()-condition_started)
                write_json(directory/'receipt.json',receipt)
                summary['conditions'].append(dict(directory=directory.name,**receipt))
                summary['additional_visual_passes']+=extra
                save()
                print(json.dumps(dict(event='msvr_source_condition_complete',fold=fold['fold'],state=state,view=view,
                    records=len(records),elapsed_seconds=receipt['elapsed_seconds'],conditions=len(summary['conditions']))),flush=True)
                del arrays,parts,values
            del model
            gc.collect();torch.cuda.empty_cache()
    assert len(summary['conditions'])==27 and summary['source_record_forwards']==18576
    assert summary['forward_batches']==306 and summary['additional_visual_passes']==306
    assert len({i for f in protocol['folds'] for i in f['source_record_indices']})==1032
    contract(args.contract)
    summary.update(status='COMPLETE_FROZEN_SOURCE_EXTRACTION',all_model_states_unchanged=True,
                   all_gradients_absent=True,elapsed_seconds=time.perf_counter()-started,
                   ended_at=datetime.now().astimezone().isoformat(),disk_free_bytes=shutil.disk_usage(args.root).free)
    save()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('check','extract'),required=True)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    (check if args.mode=='check' else run)(args)
