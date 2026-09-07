#!/usr/bin/env python3
"""Supplement the sealed source diagnostic using only its saved FP64 scalars."""
from pathlib import Path
import argparse,csv,hashlib,json,time
import numpy as np

METRICS=('signed_parallel_to_h_norm','perpendicular_to_h_norm','tangent_energy_fraction','angle_y_h_degrees')
ROLES=('cnn','transformer','mamba')
MODALITIES=('RGB','NI','TI')
VIEWS=('original','registered_style')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def derive(data):
    h2,c2,hc=data[...,0],data[...,1],data[...,2]
    assert np.all(h2>0) and np.all(c2>=0)
    parallel2=hc*hc/h2
    tangent2=c2-parallel2
    # Dot products were accumulated in FP64; clip only verified cancellation roundoff.
    assert np.all(tangent2>=-1e-10*(1+c2))
    tangent2=np.maximum(tangent2,0)
    fraction=np.zeros_like(c2)
    np.divide(tangent2,c2,out=fraction,where=c2>0)
    cosine=data[...,5]/np.sqrt(data[...,4]*h2)
    assert np.all(np.abs(cosine)<=1+1e-10)
    return np.stack((hc/h2,np.sqrt(tangent2/h2),fraction,
                     np.degrees(np.arccos(np.clip(cosine,-1,1)))),axis=-1)


def stats(values):
    values=values.reshape(-1,len(METRICS))
    assert len(values)>0 and np.isfinite(values).all()
    row={'observations':len(values)}
    for i,name in enumerate(METRICS):
        x=values[:,i];q=np.quantile(x,[.05,.5,.95])
        row.update({name+'_mean':float(x.mean()),name+'_min':float(x.min()),
                    name+'_p05':float(q[0]),name+'_median':float(q[1]),
                    name+'_p95':float(q[2]),name+'_max':float(x.max())})
    return row


def write_csv(path,rows):
    with path.open('x',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def check_math():
    # Zero, parallel, antiparallel and orthogonal corrections to h=(1,0).
    h=np.array([[1.,0.]]*4);c=np.array([[0.,0.],[3.,0.],[-2.,0.],[0.,3.]])
    y=h+c;y/=np.linalg.norm(y,axis=-1,keepdims=True)
    data=np.zeros((4,16))
    for index,value in [(0,(h*h).sum(-1)),(1,(c*c).sum(-1)),(2,(h*c).sum(-1)),
                        (4,(y*y).sum(-1)),(5,(y*h).sum(-1))]: data[:,index]=value
    expected=np.array([[0,0,0,0],[3,0,0,0],[-2,0,0,180],[0,3,1,np.degrees(np.arctan(3))]])
    actual=derive(data)
    assert np.allclose(actual,expected,rtol=1e-12,atol=1e-12)
    return {'status':'PASS_FOUR_EXPLICIT_GEOMETRY_CASES','maximum_error':float(np.max(np.abs(actual-expected))),
            'model_forwards':0,'optimizer_updates':0,'torch_imported':False}


def main(run):
    started=time.time()
    proof_path=run/'complete_source_scale_verification.json'
    proof=json.loads(proof_path.read_bytes())
    summary_path=run/'source_joint_scale.json';summary=json.loads(summary_path.read_bytes())
    assert proof['status']=='PASS_COMPLETE_SOURCE_JOINT_SCALE_ARRAYS_AND_GEOMETRY'
    assert proof['source_summary_sha256']==sha(summary_path)
    assert proof['checked_slots']==summary['slot_observations']==1935360
    assert summary['execution_commit']=='4158c95ca639721e584d1e7271219f0f3b557cc3'
    rows=[];identities=[];all_values={view:[] for view in VIEWS};bindings=[]
    zero_c=0
    for fold in summary['folds']:
        number=fold['fold'];array=Path(fold['statistics']['path']);receipt=Path(fold['batch_receipts']['path'])
        assert array.parent.resolve()==receipt.parent.resolve()==run.resolve()
        assert sha(array)==fold['statistics']['sha256'] and sha(receipt)==fold['batch_receipts']['sha256']
        data=np.load(array,mmap_mode='r')
        assert data.shape==((580,560,540)[number],2,64,3,3,16) and data.dtype==np.float64
        values=derive(data);zero_c+=int((data[...,1]==0).sum())
        batches=[json.loads(line) for line in receipt.read_bytes().splitlines()]
        source=fold['source_manifest']
        labels=np.array([[source[i]['identity'] for i in row['sampler_indices']] for row in batches])
        active=np.array([row['style_plan']['active'] for row in batches],dtype=bool)
        assert len(np.unique(labels))==94
        for vi,view in enumerate(VIEWS):
            all_values[view].append(values[:,vi].reshape(-1,len(METRICS)))
            for ei,role in enumerate(ROLES):
                for mi,modality in enumerate(MODALITIES):
                    base={'fold':number,'view':view,'role':role,'modality':modality}
                    for stratum,mask in [('all',np.ones(len(batches),dtype=bool)),('plan_active',active),('plan_inactive',~active)]:
                        rows.append({**base,'stratum':stratum,**stats(values[mask,vi,:,ei,mi])})
                    for identity in np.unique(labels):
                        original={r['file'].split('_',1)[0] for r in source if r['identity']==int(identity)}
                        assert len(original)==1
                        identities.append({**base,'encoded_identity':int(identity),'original_identity':original.pop(),
                                           **stats(values[:,vi,:,ei,mi][labels==identity])})
        bindings.append({'fold':number,'array_sha256':sha(array),'receipt_sha256':sha(receipt)})
    assert len(rows)==162 and len(identities)==5076
    overall={view:stats(np.concatenate(chunks)) for view,chunks in all_values.items()}
    assert sum(row['observations'] for row in overall.values())==1935360
    cell_path=run/'supplement_tangent_all_strata.csv';id_path=run/'supplement_tangent_all_source_identities.csv'
    write_csv(cell_path,rows);write_csv(id_path,identities)
    result={'status':'COMPLETE_SOURCE_TANGENT_SCALAR_SUPPLEMENT','source_summary_sha256':sha(summary_path),
            'complete_verification_sha256':sha(proof_path),'script_sha256':sha(__file__),
            'source_execution_commit':summary['execution_commit'],'checked_slots':1935360,'zero_c':zero_c,
            'metrics':list(METRICS),'overall_by_view':overall,'all_cells':rows,'all_identity_rows':identities,
            'fold_bindings':bindings,'cells_csv_sha256':sha(cell_path),'identity_csv_sha256':sha(id_path),
            'math_check':check_math(),'new_model_forwards':0,'new_optimizer_updates':0,'image_reads':0,
            'boundary':'Post-registration user-requested scalar supplement. Original diagnostic contract is unchanged. '
                       'Angles measure numeric direction, not identity-information loss or causality. '
                       'No initial-versus-final role comparison. Zero-c tangent fraction is defined as zero and counted separately.',
            'elapsed_seconds':time.time()-started}
    output=run/'supplement_tangent_geometry.json'
    with output.open('x',encoding='utf-8') as handle: json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['all_cells','all_identity_rows']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path);parser.add_argument('--check-math',action='store_true')
    args=parser.parse_args()
    if args.check_math: print(json.dumps(check_math()))
    else: assert args.run_dir is not None;main(args.run_dir)
