"""Measure the failed raw-norm check on one original query batch; no scoring."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'modeling'))
from tools import run_official_three_dataset_roles as legacy
from tools.official_three_dataset_model import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=('RGBNT100','MSVR310'),required=True)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists();torch.set_num_threads(2)
    reg=json.loads(args.registration.read_text());row=next(r for r in reg['rows'] if r['dataset']==args.dataset)
    for name,digest in reg['source_sha256'].items():assert sha256(ROOT/name)==digest,name
    for name,digest in row['input_sha256'].items():assert sha256(Path(name))==digest,name
    receipt=json.loads(Path(row['metrics_path']).read_text())
    protocol=legacy.read_protocol(Path(row['protocol']),args.dataset)
    model_args=argparse.Namespace(dataset=args.dataset,method='R2',seed=42,
        signal_source=ROOT/'comparators/Signal-cd1b0a6',clip_weight=ROOT/'pertrained-model/ViT-B-16.pt',
        signal_checkpoint=Path(row['author_checkpoint']),signal_sha256=row['input_sha256'][row['author_checkpoint']],epochs=50)
    model,_,_,_=legacy.initialize(model_args,protocol)
    payload=torch.load(row['role_checkpoint'],map_location='cpu',weights_only=True)
    assert set(payload['role_state_dict'])==legacy.checkpoint_names(model,'R2')
    state=model.state_dict();state.update(payload['role_state_dict']);model.load_state_dict(state,strict=True)
    from tools.run_signal_preserving_v5 import _module_state_sha256,_training_batch
    from tools.msvr310_exact_signal_inference import exact_signal_forward
    assert _module_state_sha256(model)==receipt['model_state_sha256']
    model.eval();raw=next(iter(legacy.loader_for(protocol,legacy.records_for(protocol,'query'),training=False,method='R2')))
    batch,_=_training_batch(raw)
    with torch.no_grad():values=legacy.output_mapping(exact_signal_forward(model,batch))
    f=values['fused'].float().cpu();width=reg['baseline_width']
    g=f[:,:width];r=f[:,width:];gn=g.norm(dim=1);rn=r.norm(dim=1)
    unit=torch.nn.functional.normalize(f,dim=1)
    ug=unit[:,:width].norm(dim=1);ur=unit[:,width:].norm(dim=1)
    dg=g.double().norm(dim=1);dr=r.double().norm(dim=1)
    result=dict(status='COMPLETE_ONE_QUERY_BATCH_NORM_UNIT_DIAGNOSIS',dataset=args.dataset,
        batch_count=len(f),raw_baseline_norm_min=float(gn.min()),raw_baseline_norm_max=float(gn.max()),
        raw_bank_norm_min=float(rn.min()),raw_bank_norm_max=float(rn.max()),
        raw_float32_norm_max_difference=float((gn-rn).abs().max()),raw_float64_norm_max_difference=float((dg-dr).abs().max()),
        raw_relative_norm_max_difference=float(((dg-dr).abs()/dg).max()),
        normalized_block_norm_max_difference=float((ug-ur).abs().max()),
        normalized_baseline_energy_min=float(ug.square().min()),normalized_baseline_energy_max=float(ug.square().max()),
        normalized_bank_energy_min=float(ur.square().min()),normalized_bank_energy_max=float(ur.square().max()),
        all_blocks_nonzero=bool((gn>0).all() and (rn>0).all()),model_state_sha256=_module_state_sha256(model),
        state_equal=_module_state_sha256(model)==receipt['model_state_sha256'],all_grads_none=all(p.grad is None for p in model.parameters()),
        nn_query_batches=1,official_metric_calls=0,optimizer_updates=0,driver_sha256=sha256(Path(__file__)),
        boundary='Fault diagnosis on the original first query batch only. It cannot validate all query/gallery rows or accept any adapted metric.',
        completed_at=datetime.now().astimezone().isoformat())
    assert result['state_equal'] and result['all_grads_none']
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
