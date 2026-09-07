"""Complete text-only aggregation; does not reconstruct parameter gradients."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def analyze(run, output):
    summary=json.loads((run/'summary.json').read_bytes())
    cpu=json.loads((run/'cpu_verification.json').read_bytes())
    assert summary['status']=='PASS_COMPLETE_FIXED_STATE_PROBE'
    assert cpu['status']=='PASS_COMPLETE_FIXED_STATE_PROBE_CPU'
    assert cpu['summary_sha256']==hashlib.sha256((run/'summary.json').read_bytes()).hexdigest()
    rows=[];states=[];count=0;max_closure=0.
    for f in summary['folds']:
        for name,receipt in f['states'].items():
            steps=[json.loads(s) for s in (run/f"fold_{f['fold']}_{name}"/'steps.jsonl').read_text().splitlines()]
            assert len(steps)==receipt['batches']
            count+=len(steps)
            own=[]
            for r in steps:
                for role,v in r['roles'].items():
                    uv=v['current_vs_history'];both=v['current_vs_both'];total=v['total_vs_both']
                    u,h=uv['first_norm'],uv['second_norm']
                    dot=u*h*uv['cosine'] if uv['cosine'] is not None else 0.
                    expected=u*u+h*h+2*dot
                    closure=abs(both['second_norm']**2-expected)/max(u*u+h*h,1e-30)
                    assert closure<1e-5
                    assert abs(both['difference_norm']-h)<1e-8+1e-5*max(u,h)
                    max_closure=max(max_closure,closure)
                    row=dict(fold=f['fold'],state=name,step=r['step'],epoch=r['epoch'],role=role,
                             current_norm=u,history_norm=h,history_to_current_ratio=h/u if u>0 else None,
                             current_history_cosine=uv['cosine'],current_both_cosine=both['cosine'],
                             task_total_both_cosine=total['cosine'],
                             history_noise=v['history_repeat_noise']['difference_norm'],
                             history_above_repeat_noise=h>v['history_repeat_noise']['difference_norm'],
                             history_nonzero_record_exposures=r['history_nonzero_gradient_records'])
                    rows.append(row);own.append(row)
            saved=next(s for s in cpu['states'] if s['fold']==f['fold'] and s['state']==name)
            aggregate={}
            for role in ('cnn','transformer','mamba'):
                points=[r for r in own if r['role']==role]
                norms=[r['history_norm'] for r in points]
                assert norms==saved['roles'][role]['history_parameter_norm']
                assert [r['current_norm'] for r in points]==saved['roles'][role]['current_parameter_norm']
                a={}
                for field in ('history_to_current_ratio','current_history_cosine','current_both_cosine','task_total_both_cosine'):
                    values=[r[field] for r in points if r[field] is not None]
                    a[field]=dict(defined=len(values),undefined=len(points)-len(values),
                                  mean=float(np.mean(values)) if values else None,
                                  minimum=min(values) if values else None,maximum=max(values) if values else None,
                                  negative=sum(x<0 for x in values),
                                  quantiles=np.quantile(values,[0,.25,.5,.75,1]).tolist() if values else None)
                a['history_above_repeat_noise']=sum(r['history_above_repeat_noise'] for r in points)
                aggregate[role]=a
            states.append(dict(fold=f['fold'],state=name,batches=len(steps),roles=aggregate,
                               extra_role_record_forwards=receipt['extra_role_record_forwards'],
                               chain_rule_relative_error=receipt['candidate_gradient_chain_rule']['relative_l2_error'],
                               peak_allocated_mib=receipt['peak_allocated_mib']))
    assert count==cpu['batches'] and len(states)==9
    output.mkdir()
    result=dict(status='PASS_COMPLETE_TEXT_REAGGREGATION',summary_sha256=cpu['summary_sha256'],
                batches=count,role_history_rows=len(rows),states=states,max_gradient_norm_identity_relative_error=max_closure,
                scope='All text rows and norm identities; recorded runtime gradients, not independent model backward.',
                optimizer_updates=0,heldout_record_forwards=0,official_image_reads=0)
    (output/'complete_text_reaggregation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    with (output/'all_role_history_steps.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(status=result['status'],batches=count,role_history_rows=len(rows),maximum_norm_identity_error=max_closure)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();analyze(args.run_dir,args.output_dir)
