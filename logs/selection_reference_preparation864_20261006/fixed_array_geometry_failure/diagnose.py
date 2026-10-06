"""CPU-only fixed-array additive-versus-direct-sum diagnosis; no checkpoint loads."""
import argparse
import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path('/data/gaob/Re-ID/Trifusion')
sys.path.insert(0,str(ROOT))
from tools.report_row_mass_fixed_best import compare,score,write

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    seal=json.loads(args.seal.read_text())
    assert seal['schema']=='fixed-array-additive-geometry-v1'
    assert all(sha(path)==digest for path,digest in seal['input_sha256'].items())
    assert all(sha(path)==digest for path,digest in seal['source_sha256'].items())
    source=ROOT/'tools/run_official_three_dataset_roles.py'
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='distance_matrix')
    namespace={}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    distance_matrix=namespace['distance_matrix']
    assert not args.output.exists()
    args.output.mkdir()
    rows=[]
    for row in seal['rows']:
        folder=Path(row['original_folder'])
        arrays=torch.load(folder/'diagnostic_arrays.pt',map_location='cpu',weights_only=False)
        metadata=torch.load(folder/'distances.pt',map_location='cpu',weights_only=False)
        own_global=json.loads((folder/'own_global.json').read_text())
        g,c,h=(arrays[name] for name in ('g','c','h'))
        assert g.shape==c.shape==h.shape and g.shape[1]==1536
        assert all(x.device.type=='cpu' and x.dtype==torch.float32 and not x.requires_grad and torch.isfinite(x).all() for x in (g,c,h))
        gain=row['gain']
        reconstructed=g+gain*c
        raw_difference=float((reconstructed-h).abs().max())
        assert torch.allclose(reconstructed,h,atol=1e-5,rtol=1e-5)
        count=len(metadata['query_ids'])
        assert g.shape[0]==count+len(metadata['gallery_ids'])
        result_dir=args.output/(row['dataset']+'_'+row['variant'])
        result_dir.mkdir()
        original=score(metadata,row['dataset'])
        additive_features=F.normalize(h,dim=1)
        additive=distance_matrix(additive_features[:count],additive_features[count:])
        additive_scores=score({**metadata,'fused':additive},row['dataset'])
        differences={k:abs(additive_scores['metrics'][k]-original['metrics'][k]) for k in original['metrics']}
        write(result_dir/'CPU_ADDITIVE_PARITY.json',dict(raw_h_max_difference=raw_difference,
            original_metrics=original['metrics'],cpu_metrics=additive_scores['metrics'],
            difference_points=differences,tolerance_points=1e-5))
        assert max(differences.values())<1e-5,differences
        direct_sum=F.normalize(torch.cat((g,gain*c),dim=1),dim=1)
        assert direct_sum.shape==(g.shape[0],3072)
        new_distances=distance_matrix(direct_sum[:count],direct_sum[count:])
        new_scores=score({**metadata,'fused':new_distances},row['dataset'])
        torch.save({**metadata,'fused':new_distances},result_dir/'direct_sum_distances.pt')
        pairs={
            'additive__direct_sum':compare(additive_scores,new_scores,np.asarray(metadata['query_ids']),result_dir/'additive__direct_sum'),
            'own_global__direct_sum':compare(own_global,new_scores,np.asarray(metadata['query_ids']),result_dir/'own_global__direct_sum')}
        item=dict(dataset=row['dataset'],variant=row['variant'],best_epoch=row['best_epoch'],fixed_gain=gain,
                  feature_width=3072,additive_cpu_parity=differences,original_metrics=original['metrics'],
                  direct_sum_metrics=new_scores['metrics'],own_global_metrics=own_global['metrics'],pairs=pairs,
                  new_distance_sha256=sha(result_dir/'direct_sum_distances.pt'))
        write(result_dir/'RESULT.json',item)
        rows.append(item)
        write(args.output/'PROGRESS.json',dict(status='RUNNING',completed=len(rows),expected=6))
    assert len(rows)==6
    assert all(sha(path)==digest for path,digest in seal['input_sha256'].items())
    assert all(sha(path)==digest for path,digest in seal['source_sha256'].items())
    write(args.output/'SUMMARY.json',dict(schema=seal['schema'],status='COMPLETE',completed_at=datetime.now().astimezone().isoformat(),
        models=6,comparisons=12,rows=rows,boundary='Fixed saved vectors and checkpoint gain, full query/gallery, CPU-only. Direct sum doubles feature width; no learned heads, NN, training, checkpoint/gain/epoch/seed selection, semantic correspondence or SOTA claim.'))

if __name__=='__main__':
    main()
