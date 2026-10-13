"""Read saved G/C vectors on CPU; descriptive moments, no retrieval change."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def moments(x):
    assert x.shape == (836,1536) and torch.isfinite(x).all()
    x = x.double()
    mean = x.mean(0)
    total = x.square().sum(1).mean()
    common = mean.square().sum()
    variation = (x-mean).square().sum(1).mean()
    assert float(total) > 0 and float(common) > 0
    assert torch.allclose(total,common+variation,atol=1e-10,rtol=1e-12)
    cosine = (x@mean)/(x.norm(dim=1)*mean.norm())
    assert torch.isfinite(cosine).all()
    return {'samples':len(x),'mean_squared_norm':float(total),
            'squared_norm_of_sample_mean':float(common),
            'mean_squared_deviation_from_sample_mean':float(variation),
            'common_energy_fraction':float(common/total),
            'sample_mean_direction_cosine_mean':float(cosine.mean()),
            'sample_mean_direction_cosine_min':float(cosine.min()),
            'sample_mean_direction_cosine_max':float(cosine.max())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not args.output.exists()
    torch.set_num_threads(4)
    started=time.perf_counter()
    seal=json.loads(args.seal.read_text())
    assert sha(Path(__file__))==seal['source_sha256']
    assert all(sha(Path(p))==h for p,h in seal['artifact_sha256'].items())
    result={'schema':'saved-evidence-common-component-v1','evaluation_type':'descriptive_saved_feature_statistics',
            'seal_sha256':sha(args.seal),'source_sha256':sha(Path(__file__)),
            'neural_forward_calls':0,'optimizer_updates':0,'retrieval_scores_computed':0,
            'model_or_embedding_modified':False,'rows':[]}
    for row in seal['rows']:
        features=torch.load(row['path'],map_location='cpu',weights_only=True)
        assert set(features)=={'shared_global','correction','raw_fused','fused'}
        result['rows'].append({'exposure':row['exposure'],'split':row['split'],
          'feature_sha256':sha(Path(row['path'])),
          'global':moments(features['shared_global']),
          'correction':moments(features['correction'])})
    assert all(sha(Path(p))==h for p,h in seal['artifact_sha256'].items())
    result.update(elapsed_seconds=time.perf_counter()-started,
       boundary='Single fixed E8 RGBNT201 seed42. Query/gallery overlap, not independent replications. Mean is descriptive only; no test-mean calibration, transformed embedding, new distance, identity scoring, sample selection, parameter fitting or causal claim. Large common energy alone does not prove useless identity information; compare global and require direct task evidence.')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
