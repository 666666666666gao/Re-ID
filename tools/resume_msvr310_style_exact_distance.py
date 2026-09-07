#!/usr/bin/env python3
"""Resume the fixed style experiment with the measured B0 CPU distance dispatch."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import time

import torch

from tools import train_msvr310_source_style as original
from tools.train_msvr310_signal_oof import sha256,write_json,records_for

ROOT=Path(__file__).resolve().parents[1]


def contract(args):
    value=json.loads(args.contract.read_bytes())
    assert value['training_cpu_threads']==4 and value['distance_cpu_threads']==56
    for name,digest in value['project_file_sha256'].items():assert sha256(ROOT/name)==digest,name
    for name,digest in value['original_run_file_sha256'].items():assert sha256(Path(value['original_run'])/name)==digest,name
    config=ROOT/value['original_config'];assert sha256(config)==value['original_config_sha256']
    return value,config


def check(args):
    plan,config_path=contract(args)
    _config,_base,_cfg,_env,_protocol,baseline,_meta=original.context(config_path)
    assert torch.get_num_threads()==4 and not torch.cuda.is_initialized()
    rows=[]
    for threads in (4,56):
        torch.set_num_threads(threads)
        for f,b0 in enumerate(baseline['folds']):
            data=torch.load(Path(b0['checkpoint']).parent/'retrieval_arrays.pt',map_location='cpu',weights_only=True)
            unit=torch.nn.functional.normalize(data['features'].float(),dim=1);q=unit[data['query_gallery_positions']]
            matrix=q.square().sum(1,keepdim=True)+unit.square().sum(1)[None]
            matrix.addmm_(q,unit.T,beta=1,alpha=-2)
            error=matrix-data['distances']
            equal=torch.equal(matrix,data['distances'])
            assert equal==(threads==56)
            rows.append(dict(fold=f,cpu_threads=threads,bitwise_equal=equal,
                             differing_entries=int((error!=0).sum()),maximum_error=float(error.abs().max())))
    torch.set_num_threads(4)
    from tools.verify_msvr310_source_style import state_sha
    prior=Path(plan['original_run'])
    saved=torch.load(prior/'q1/fold_0_control/roles_epoch20.pth',map_location='cpu',weights_only=True)
    base_state=torch.load(baseline['folds'][0]['checkpoint'],map_location='cpu',weights_only=True)['model_state_dict']
    state={key:base_state[source] for key,source in saved['baseline_aliases'].items()}
    state.update(saved['role_state_dict'])
    training=json.loads((prior/'q1/fold_0_control/training.json').read_bytes())
    assert state_sha(state)==training['final_state_sha256']
    assert saved['config_sha256']==sha256(config_path)
    assert not torch.cuda.is_initialized()
    out=args.root/'distance_preflight.json';assert not out.exists()
    write_json(out,dict(status='PASS_EXACT_B0_CPU_DISTANCE_DISPATCH',contract_sha256=sha256(args.contract),
                        checked_at=datetime.now().astimezone().isoformat(),all_three_fold_checks=rows,
                        reused_checkpoint_full_state_sha256=training['final_state_sha256'],
                        model_forwards=0,optimizer_updates=0,checkpoint_writes=0,ranking_metric_evaluations=0))


def run(args):
    from tools.run_signal_preserving_v5 import _module_state_sha256
    started=time.time();plan,config_path=contract(args)
    assert torch.get_num_threads()==4 and torch.cuda.is_available()
    proof=json.loads((args.root/'distance_preflight.json').read_bytes())
    assert proof['status']=='PASS_EXACT_B0_CPU_DISTANCE_DISPATCH' and proof['contract_sha256']==sha256(args.contract)
    config,base,cfg,environment,protocol,baseline,metadata=original.context(config_path)
    prior=Path(plan['original_run']);pipeline=json.loads((prior/'pipeline.json').read_bytes())
    assert pipeline['status']=='STOPPED_AT_Q1' and pipeline['stages'][-1]['exit_code']==1
    m0=json.loads((prior/'m0/summary.json').read_bytes());m0_cpu=json.loads((prior/'m0_cpu.json').read_bytes())
    assert m0['status']=='PASS_ENGINEERING_ONLY' and m0_cpu['status']=='PASS_COMPLETE_MSVR_STYLE_M0'
    assert m0['config_sha256']==sha256(config_path) and m0_cpu['summary_sha256']==sha256(prior/'m0/summary.json')
    partial=json.loads((prior/'q1/summary.json').read_bytes())
    assert partial['status']=='RUNNING' and len(partial['folds'])==1
    assert list(partial['folds'][0]['endpoints'])==['control']
    old=partial['folds'][0]['endpoints']['control']
    assert old['training']==json.loads((prior/'q1/fold_0_control/training.json').read_bytes())
    assert old['training']['optimizer_steps']==260 and old['training']['epochs']==20
    directory=args.root/'q1';assert not directory.exists();directory.mkdir()
    summary=dict(status='RUNNING',mode='q1',seed=42,config_sha256=sha256(config_path),
                 repair_contract_sha256=sha256(args.contract),runner_sha256=sha256(__file__),
                 project_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                 environment=environment,folds=[],overfit={},official_image_reads=0,rgbnt201_dev_image_reads=0,
                 heldout_record_forwards=0,prior_failed_heldout_record_forwards=360,
                 m0_weights_reused=False,baseline_retrained=False,original_training_updates_reused=260,
                 new_optimizer_steps=0,distance_cpu_threads=56,training_cpu_threads=4,
                 m0_receipt_sha256=sha256(prior/'m0/summary.json'),m0_verification_sha256=sha256(prior/'m0_cpu.json'))
    def save():
        summary['elapsed_seconds']=time.time()-started;write_json(directory/'summary.json',summary)
    save()
    for fold,b0,md in zip(protocol['folds'],baseline['folds'],metadata['folds'],strict=True):
        records=records_for(base,protocol,fold,True)
        result=dict(fold=fold['fold'],endpoints={});summary['folds'].append(result);paired=[]
        for end in original.ENDPOINTS:
            local=directory/f"fold_{fold['fold']}_{end}";local.mkdir()
            model,binding=original.build(config,cfg,fold,b0)
            assert binding==m0['folds'][fold['fold']]['endpoints'][end]['initialization']
            reused=fold['fold']==0 and end=='control'
            if reused:
                assert binding==old['initialization']
                training=json.loads((prior/'q1/fold_0_control/training.json').read_bytes())
                path=Path(training['style_steps_path'])
                assert sha256(path)==training['style_steps_sha256']
                shutil.copyfile(path,local/'style_steps.jsonl')
                styles=[json.loads(line) for line in path.read_text().splitlines()]
                training['style_steps_path']=str(local/'style_steps.jsonl')
                write_json(local/'training.json',training)
                checkpoint=prior/'q1/fold_0_control/roles_epoch20.pth'
            else:
                assert torch.get_num_threads()==4
                training,styles=original.fit(model,records,fold,config,md,endpoint=end,mode='comparison',directory=local)
                summary['new_optimizer_steps']+=training['optimizer_steps']
                checkpoint=local/'roles_epoch20.pth'
                original.checkpoint(model,binding,fold,sha256(config_path),checkpoint)
            checks=original.engineering_checks(training);checks['fixed_training_length']=training['optimizer_steps']==260
            row=dict(fold=fold['fold'],initialization=binding,preflight=None,training=training,engineering_checks=checks,
                     training_reused_from_original_run=reused,training_execution_commit=partial['project_commit'] if reused else summary['project_commit'])
            result['endpoints'][end]=row;save();assert all(checks.values()),checks
            checkpoint_hash=sha256(checkpoint)
            del model;torch.cuda.empty_cache()
            model=original.reload_model(config,cfg,fold,b0,checkpoint,binding,training['final_state_sha256'],sha256(config_path))
            row.update(checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_hash,strict_reload_state_sha256=_module_state_sha256(model))
            gallery=records_for(base,protocol,fold,False)
            assert torch.get_num_threads()==4
            features=original.extract(model,gallery)
            torch.set_num_threads(56)
            row['retrieval']=original.evaluate(features,protocol,fold,local,b0)
            torch.set_num_threads(4)
            row['retrieval']['distance_cpu_threads']=56
            summary['heldout_record_forwards']+=len(gallery)
            assert _module_state_sha256(model)==training['final_state_sha256'] and sha256(checkpoint)==checkpoint_hash
            paired.append([{k:s[k] for k in ('plan','pixel_sha256')} for s in styles])
            write_json(local/'receipt.json',row);save()
            del model;torch.cuda.empty_cache()
        assert paired[0]==paired[1]
        assert result['endpoints']['control']['initialization']==result['endpoints']['source_style']['initialization']
        result['all_paired_inputs_and_style_plans_exact']=True;save()
    comparison=original.paired_summary(summary['folds'])
    assert summary['new_optimizer_steps']==1300 and summary['heldout_record_forwards']==2064
    contract(args)
    summary.update(status='Q1_PASS' if comparison['next_phase_qualified'] else 'Q1_FAIL',comparison=comparison,
                   optimizer_steps=1560,original_failed_run_files_unchanged=True,
                   completed_at=datetime.now().astimezone().isoformat())
    save();print(json.dumps(dict(status=summary['status'],new_optimizer_steps=1300,reused_optimizer_steps=260)),flush=True)


def verify(args):
    plan,config_path=contract(args)
    from tools.verify_msvr310_source_style import verify as full_verify
    torch.set_num_threads(56)
    full_verify(argparse.Namespace(config=config_path,run_dir=args.root/'q1',output=args.root/'q1_cpu.json'))
    summary=json.loads((args.root/'q1/summary.json').read_bytes())
    assert summary['repair_contract_sha256']==sha256(args.contract)
    assert summary['original_training_updates_reused']==260 and summary['new_optimizer_steps']==1300
    assert summary['training_cpu_threads']==4 and summary['distance_cpu_threads']==56
    assert sum(r['training_reused_from_original_run'] for f in summary['folds'] for r in f['endpoints'].values())==1
    original_training=json.loads((Path(plan['original_run'])/'q1/fold_0_control/training.json').read_bytes())
    reused=summary['folds'][0]['endpoints']['control']['training']
    assert {k:v for k,v in original_training.items() if k!='style_steps_path'}=={k:v for k,v in reused.items() if k!='style_steps_path'}
    contract(args)
    write_json(args.root/'resume_verification.json',dict(status='PASS_FIXED_ORIGINAL_END_REUSE_AND_EXACT_CPU_DISTANCE',
        verifier_sha256=sha256(__file__),repair_contract_sha256=sha256(args.contract),
        original_training_updates_reused=260,new_optimizer_steps=1300,original_files_unchanged=True,
        complete_cpu_verification_sha256=sha256(args.root/'q1_cpu.json'),summary_sha256=sha256(args.root/'q1/summary.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('check','q1','cpu'),required=True)
    p.add_argument('--contract',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();args.root=args.root.resolve();args.contract=args.contract.resolve()
    {'check':check,'q1':run,'cpu':verify}[args.mode](args)
