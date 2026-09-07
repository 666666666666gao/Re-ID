"""Read-only complete mining arithmetic; no model, images, or training changes."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import csv
import time

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decompose(hp, hn, up, un):
    base = np.maximum(hp-hn+.3, 0)
    positive = np.maximum(up-hn+.3, 0)
    negative = np.maximum(hp-un+.3, 0)
    both = np.maximum(up-un+.3, 0)
    p = .5*((positive-base)+(both-negative))
    n = .5*((negative-base)+(both-positive))
    assert np.all(p >= -1e-12) and np.all(n >= -1e-12)
    assert np.allclose(p+n, both-base, rtol=0, atol=1e-12)
    return base, positive, negative, both, p, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    # Fixture includes inactive, one-sided, and jointly activated hinges.
    expected = decompose(np.array([.2,.2,.2]),np.array([.8,.8,.8]),
                         np.array([.2,.7,.4]),np.array([.8,.8,.6]))
    assert np.allclose(expected[4]+expected[5], [0,.2,.1])
    assert np.allclose(expected[4], [0,.2,.05])
    summary = json.loads((args.run/'q1/summary.json').read_bytes())
    cpu = json.loads((args.run/'q1_cpu.json').read_bytes())
    assert summary['status'] == 'Q1_FAIL'
    assert cpu['status'] == 'PASS_COMPLETE_INSTANCE_MEMORY_Q1'
    assert sha(args.run/'q1/summary.json') == cpu['summary_sha256']
    assert summary['config_sha256'] == '40f44b0e6c12771c53a283a5b65ca6a45556e0293b7d48decf5fd518ad771a3d'
    pipeline = json.loads((args.run/'pipeline.json').read_bytes())
    assert pipeline['code_commit'] == '104506b72193b6cdbcd237044be97c2410fd7a45'
    assert summary['project_commit'] == '9da45c589ef56b05eabec8c73ff32533e32556cd'
    assert summary['runner_sha256'] == 'c16cdbff37b46decb279aa013e4a53d9273851a59160cda99fbdf0f16cf39f3b'
    args.output.mkdir()
    rows, endpoints, inputs = [], [], {}
    elements = 0
    for fold in summary['folds']:
        for end in ['control','instance_memory']:
            directory = args.run/f"q1/fold_{fold['fold']}_{end}"
            tr = json.loads((directory/'training.json').read_bytes())
            assert tr == fold['endpoints'][end]['training']
            for name, proof in tr['audit_files'].items():
                assert (directory/name).stat().st_size == proof['bytes']
                assert sha(directory/name) == proof['sha256']
                inputs[str(directory/name)] = proof
            audits = [json.loads(x) for x in (directory/'memory_steps.jsonl').read_text().splitlines()]
            assert len(audits) == 260
            per_end = []
            with (directory/'memory_distances.f32').open('rb') as f:
                for step, a in enumerate(audits,1):
                    assert a['step'] == step and f.tell() == a['distance_offset_bytes']
                    m = len(a['memory'])
                    raw = np.fromfile(f, dtype=np.float32, count=a['distance_float_count'])
                    assert len(raw) == 64*(64+m) and np.isfinite(raw).all()
                    elements += len(raw)
                    dc, dm = raw[:4096].reshape(64,64).astype(np.float64),raw[4096:].reshape(64,m).astype(np.float64)
                    ids = np.array(a['identities']); scenes = np.array(a['scenes'])
                    pos = (ids[:,None] == ids[None,:]) & ~np.eye(64,dtype=bool)
                    neg = ids[:,None] != ids[None,:]
                    hp = np.where(pos, dc,-np.inf).max(1)
                    hn = np.where(neg, dc, np.inf).min(1)
                    p_age = [0]*9; n_age = [0]*9
                    p_ties = n_ties = p_cross = n_same = 0
                    p_win = np.zeros(64,dtype=bool); n_win = np.zeros(64,dtype=bool)
                    up, un = hp, hn
                    if m:
                        mids=np.array([r['identity'] for r in a['memory']])
                        age=np.array([r['age'] for r in a['memory']])
                        msc=np.array([r['scene'] for r in a['memory']])
                        mp=ids[:,None] == mids[None,:]
                        pd=np.where(mp,dm,-np.inf);nd=np.where(~mp,dm,np.inf)
                        ip=pd.argmax(1);inn=nd.argmin(1)
                        ph=pd[np.arange(64),ip];nh=nd[np.arange(64),inn]
                        p_win=ph>hp;n_win=nh<hn
                        up=np.maximum(hp,ph);un=np.minimum(hn,nh)
                        for i in np.flatnonzero(p_win):
                            p_age[int(age[ip[i]])] += 1
                            p_ties += int(np.count_nonzero(pd[i]==ph[i])>1)
                            p_cross += int(scenes[i] != msc[ip[i]])
                        for i in np.flatnonzero(n_win):
                            n_age[int(age[inn[i]])] += 1
                            n_ties += int(np.count_nonzero(nd[i]==nh[i])>1)
                            n_same += int(scenes[i] == msc[inn[i]])
                    base, ponly, nonly, both, ppart, npart = decompose(hp,hn,up,un)
                    assert abs(base.mean()-a['statistics']['current_triplet'])<2e-6
                    assert abs(both.mean()-a['statistics']['expanded_triplet'])<2e-6
                    assert int(p_win.sum())==a['statistics']['harder_positive_anchors']
                    assert int(n_win.sum())==a['statistics']['harder_negative_anchors']
                    assert tr['steps'][step-1]['step'] == step
                    row=dict(fold=fold['fold'],endpoint=end,step=step,epoch=tr['steps'][step-1]['epoch'],history_count=m,
                             anchors=64,base_sum=float(base.sum()),positive_only_sum=float(ponly.sum()),
                             negative_only_sum=float(nonly.sum()),expanded_sum=float(both.sum()),
                             positive_shapley_sum=float(ppart.sum()),negative_shapley_sum=float(npart.sum()),
                             positive_wins=int(p_win.sum()),negative_wins=int(n_win.sum()),
                             positive_only_wins=int((p_win & ~n_win).sum()),negative_only_wins=int((n_win & ~p_win).sum()),
                             both_win=int((p_win & n_win).sum()),neither_win=int((~p_win & ~n_win).sum()),
                             expanded_wrong=int((up>=un).sum()),base_wrong=int((hp>=hn).sum()),
                             positive_winning_cross_scene=p_cross,negative_winning_same_scene=n_same,
                             positive_ambiguous_ties=p_ties,negative_ambiguous_ties=n_ties)
                    for age in range(1,9):
                        row['positive_win_age_'+str(age)] = p_age[age]
                        row['negative_win_age_'+str(age)] = n_age[age]
                    rows.append(row);per_end.append(row)
                assert f.read(1)==b''
            def aggregate(selected):
                keys=[k for k in selected[0] if k not in {'fold','endpoint','step','epoch'}]
                total={k:sum(r[k] for r in selected) for k in keys}
                delta=total['expanded_sum']-total['base_sum']
                total.update(steps=len(selected),negative_share_of_added_hinge=total['negative_shapley_sum']/delta if delta>0 else None,
                             positive_share_of_added_hinge=total['positive_shapley_sum']/delta if delta>0 else None)
                return total
            endpoints.append(dict(fold=fold['fold'],endpoint=end,all_steps=aggregate(per_end),
                                  history_steps=aggregate([r for r in per_end if r['history_count']]),
                                  epochs=[dict(epoch=e,**aggregate([r for r in per_end if r['epoch']==e])) for e in range(1,21)]))
    assert len(rows)==1560 and elements==29125376
    with (args.output/'all1560_step_mining.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report=dict(status='PASS_COMPLETE_MINING_DECOMPOSITION',checked_at=datetime.now().astimezone().isoformat(),
                run_dir=str(args.run),source_code_sha256=sha(__file__),summary_sha256=sha(args.run/'q1/summary.json'),
                wrapper_execution_commit=pipeline['code_commit'],q1_observed_documentation_head=summary['project_commit'],
                q1_runner_sha256=summary['runner_sha256'],
                cpu_sha256=sha(args.run/'q1_cpu.json'),all_input_file_hashes=inputs,steps=len(rows),anchor_exposures=64*len(rows),
                distance_elements=elements,endpoints=endpoints,optimizer_updates=0,model_forwards=0,image_reads=0,
                official_reads=0,new_checkpoint_count=0,elapsed_seconds=time.perf_counter()-started,
                interpretation='Posthoc source-distance arithmetic only. Positive/negative two-player Shapley splits added hinge, not causal training effects. Age uses first argmin/argmax among exact ties and tie counts are reported. No fresh historical re-encoding or parameter gradients are calculated; no age or loss intervention is tested.',
                csv_sha256=sha(args.output/'all1560_step_mining.csv'))
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in {'endpoints','all_input_file_hashes'}}))
    print(json.dumps([dict(fold=e['fold'],endpoint=e['endpoint'],negative_share=e['history_steps']['negative_share_of_added_hinge'],
                           positive_wins=e['history_steps']['positive_wins'],negative_wins=e['history_steps']['negative_wins'],
                           wrong_before=e['history_steps']['base_wrong'],wrong_expanded=e['history_steps']['expanded_wrong'],
                           age_negative=[e['history_steps']['negative_win_age_'+str(a)] for a in range(1,9)]) for e in endpoints]))


if __name__=='__main__':
    main()
