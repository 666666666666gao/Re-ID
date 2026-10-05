from datetime import datetime
from pathlib import Path
import hashlib,json,os,sys
assert os.environ['CUDA_VISIBLE_DEVICES']==''
root=Path('/data/gaob/Re-ID/Trifusion');sys.path.insert(0,str(root))
import torch
import torch.nn.functional as F
import numpy as np
from tools import diagnose_native_research_best as diagnosis
torch.set_num_threads(4)
sha=diagnosis.panel.base.sha
folder=root/'results/deployment_metric_fixed_best_five_20261005_850/MSVR310_semantic'
output=root/'results/deployment_metric_msvr_failed_saved_array_description_20261005_854'
assert not output.exists() and not (folder/'DIAGNOSIS.json').exists()
campaign=root/'results/deployment_metric_fixed_best_five_20261005_850/campaign.json'
log=folder.parent/'MSVR310_semantic.log'
assert json.loads(campaign.read_text())['status']=='FAILED'
assert sha(campaign)=='119f8b6b57fe14456bbc4fd239a9c43cb465ecd823cefd750b5bbdd95c07d359'
assert sha(log)=='0005c97e4806e6b2fc64950966d31b83ef5bbb1299ff464e96696b4d531ab6ca'
assert all(sha(folder/n)==r['sha256'] and (folder/n).stat().st_size==r['bytes'] for n,r in {'query_features.pt': {'bytes': 14539735, 'sha256': 'c9c42b8769ca53036797988e188aa4bb990353f7878fb1f2dbc307c95e3642d1'}, 'diagnostic_distances.pt': {'bytes': 7524463, 'sha256': '12be6b7088ab84ac4e0017c12539c7460091c612bc9084c6db5f185efbad0a60'}, 'gallery_features.pt': {'bytes': 25952305, 'sha256': '01a441bd03336221c0398243aa69d9f110e0d93491348e1a5a5982f94b872fb3'}}.items())
seal_path=root/'refine-logs/deployment_metric_fixed_best_diagnosis_v1/INPUT_SEAL.json'
assert sha(seal_path)=='a4de9faca7f502cb7a4fa01800e437919810d2cfa83bfcf0361047d831a42cb7'
seal=json.loads(seal_path.read_text())
assert len(seal['source_sha256'])==341 and len(seal['artifact_sha256'])==271
assert all(sha(root/n)==d for n,d in seal['source_sha256'].items())
assert all(sha(Path(n))==d for n,d in seal['artifact_sha256'].items())
row=next(r for r in seal['rows'] if (r['dataset'],r['variant'])==('MSVR310','semantic'))
global_row=next(r for r in seal['rows'] if (r['dataset'],r['variant'])==('MSVR310','global_only'))
old=torch.load(Path(row['run_dir'])/'official_distances.pt',map_location='cpu',weights_only=False)
independent=torch.load(Path(global_row['run_dir'])/'official_distances.pt',map_location='cpu',weights_only=False)
saved=torch.load(folder/'diagnostic_distances.pt',map_location='cpu',weights_only=False)
protocol=json.loads((diagnosis.panel.PROTOCOLS/'MSVR310.json').read_text())
metadata=diagnosis.metadata(protocol)
assert all(np.array_equal(saved[k],metadata[k]) and np.array_equal(old[k],metadata[k])
 and np.array_equal(independent[k],metadata[k]) for k in metadata)
sys.path.insert(0,str(diagnosis.panel.SOURCE))
from utils import metrics as author
assert Path(author.__file__).resolve()==(diagnosis.panel.SOURCE/'utils/metrics.py').resolve()
def score(matrix):
 r=diagnosis.scene_scores(matrix.numpy(),metadata['query_ids'],metadata['gallery_ids'],metadata['query_scenes'],metadata['gallery_scenes'])
 cmc,ap=author.eval_func_msrv(matrix.numpy(),metadata['query_ids'],metadata['gallery_ids'],metadata['query_cameras'],metadata['gallery_cameras'],metadata['query_scenes'],metadata['gallery_scenes'])
 actual={'mAP':float(ap)*100,'Rank-1':float(cmc[0])*100,'Rank-5':float(cmc[4])*100,'Rank-10':float(cmc[9])*100}
 assert all(abs(r['metrics'][k]-v)<1e-5 for k,v in actual.items())
 return r
scores={name:score(saved[name]) for name in ('global','correction','fused')}
scores['independent_global_only']=score(independent['fused'])
assert all(abs(scores['fused']['metrics'][k]-v)<1e-5 for k,v in row['metrics'].items())
assert all(abs(scores['independent_global_only']['metrics'][k]-v)<1e-5 for k,v in global_row['metrics'].items())
stats={}
for split in ('query','gallery'):
 payload=torch.load(folder/(split+'_features.pt'),map_location='cpu',weights_only=False)
 features=payload['features'];values=payload['per_sample_diagnostics']
 assert all(v.shape==(protocol['counts'][split],1536) and bool(torch.isfinite(v).all()) for v in features.values())
 assert torch.allclose(F.normalize(features['raw_fused'],dim=1),features['fused'],atol=1e-6,rtol=1e-6)
 ratio=(features['raw_fused']-features['shared_global']).norm(dim=1)/features['shared_global'].norm(dim=1)
 assert torch.allclose(ratio,values['actual_scaled_correction_to_global_norm_ratio'],atol=1e-6,rtol=1e-6)
 stats[split]={n:diagnosis.describe(v) for n,v in values.items()}
comparisons={'same_model_global_to_fused':diagnosis.paired(scores['global'],scores['fused'],metadata['query_ids']),
 'independent_global_to_same_model_global':diagnosis.paired(scores['independent_global_only'],scores['global'],metadata['query_ids']),
 'independent_global_to_fused':diagnosis.paired(scores['independent_global_only'],scores['fused'],metadata['query_ids'])}
report=dict(status='CPU_DESCRIPTION_COMPLETE_ORIGINAL_DIAGNOSIS_REMAINS_FAILED',at=datetime.now().astimezone().isoformat(),
 dataset='MSVR310',variant='semantic',checkpoint_sha256=row['checkpoint_sha256'],selected_epoch=row['best_epoch'],
 original_failed_campaign_sha256=sha(campaign),original_failed_log_sha256=sha(log),partial_inputs={'query_features.pt': {'bytes': 14539735, 'sha256': 'c9c42b8769ca53036797988e188aa4bb990353f7878fb1f2dbc307c95e3642d1'}, 'diagnostic_distances.pt': {'bytes': 7524463, 'sha256': '12be6b7088ab84ac4e0017c12539c7460091c612bc9084c6db5f185efbad0a60'}, 'gallery_features.pt': {'bytes': 25952305, 'sha256': '01a441bd03336221c0398243aa69d9f110e0d93491348e1a5a5982f94b872fb3'}},
 fused_distance_max_abs_diff_from_original=float((saved['fused']-old['fused']).abs().max()),
 scores=scores,diagnostic=stats,comparisons=comparisons,nn_forwards=0,optimizer_updates=0,
 boundary='Only saved partial arrays. Full original metadata/filter and author scorer parity, no second forward or retry. No reconstructed model-state before/after digest, accepted DIAGNOSIS receipt, changed failure, selection or seed-stability claim.')
output.mkdir();target=output/'DESCRIPTION.json';target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
assert sha(campaign)==report['original_failed_campaign_sha256'] and sha(log)==report['original_failed_log_sha256']
assert not (folder/'DIAGNOSIS.json').exists()
brief={k:v for k,v in report.items() if k not in ('scores','comparisons')}
brief['metrics']={k:v['metrics'] for k,v in scores.items()}
brief['comparisons']={k:{n:v[n] for n in ('delta_metrics','rank1_repairs','rank1_new_errors','identity_macro_delta_ap_points')} for k,v in comparisons.items()}
print(json.dumps(dict(report=brief,file={'path':str(target),'bytes':target.stat().st_size,'sha256':sha(target),'text':target.read_text()})))
