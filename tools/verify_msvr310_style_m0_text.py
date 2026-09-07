#!/usr/bin/env python3
"""Independent local verification of every M0 receipt, loss and paired input."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

ENDS=('control','source_style')
ROLES=('cnn','transformer','mamba')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    directory=args.directory;root=Path(__file__).resolve().parents[1]
    def read(name):return json.loads((directory/name).read_bytes())
    summary=read('m0_summary.json');cpu=read('m0_cpu.json');intake=read('intake_proof.json')
    config=json.loads(args.config.read_bytes())
    metadata=json.loads((root/config['SOURCE_METADATA']['PATH']).read_bytes())
    protocol=json.loads((root/config['DATA']['PROTOCOL']).read_bytes())
    assert summary['status']=='PASS_ENGINEERING_ONLY' and summary['optimizer_steps']==248
    assert summary['config_sha256']==sha(args.config)
    assert summary['official_image_reads']==summary['rgbnt201_dev_image_reads']==summary['heldout_record_forwards']==0
    assert cpu['status']=='PASS_COMPLETE_MSVR_STYLE_M0' and cpu['checked_training_steps']==248
    assert cpu['summary_sha256']==sha(directory/'m0_summary.json')
    for remote,item in intake['files'].items():
        local=directory/remote.replace('/','_')
        assert local.stat().st_size==item['bytes'] and sha(local)==item['sha256']
    total=0;loss_error=0.;table=[];styles_by_run={}
    def check(row,stem,fold,end,overfit):
        nonlocal total,loss_error
        training=row['training'];initial=row['initialization']
        assert training==read(stem+'_training.json')
        assert training['initial_state_sha256']==initial['initial_state_sha256']!=training['final_state_sha256']
        assert training['signal_state_before_sha256']==training['signal_state_after_sha256']==initial['signal_state_sha256']
        assert training['frozen_state_before_sha256']==training['frozen_state_after_sha256']
        assert training['trainable_tensors']==training['nonzero_gradient_tensors']==203
        assert not training['missing_nonzero_gradients'] and training['overflow_events']==0
        assert initial['source_ids']==fold['source_ids'] and initial['heldout_ids']==fold['heldout_ids']
        assert not set(initial['source_ids'])&set(initial['heldout_ids'])
        assert all(row['checks' if overfit else 'engineering_checks'].values())
        path=directory/(stem+'_style_steps.jsonl')
        assert sha(path)==training['style_steps_sha256']
        styles=[json.loads(s) for s in path.read_text().splitlines()]
        assert len(styles)==len(training['steps'])==training['optimizer_steps']==(100 if overfit else 8)
        md=metadata['folds'][fold['fold']]['batches'];weight=config['LOSS']
        for k,(step,style) in enumerate(zip(training['steps'],styles,strict=True)):
            expected=md[0 if overfit else k]
            assert step['sampled_record_indices']==expected['record_indices']
            assert set(step['sampled_record_indices'])<=set(fold['source_record_indices'])
            plan=expected['style_plan'] if not overfit else {**expected['style_plan'],'active':True,'forced_active':True}
            assert style['plan']==plan and step['step']==style['step']==k+1
            assert style['statistics']['style_active']==int(end=='source_style' and plan['active'])
            assert style['statistics']['additional_visual_passes']==3
            assert set(style['pixel_sha256'])=={'RGB','NI','TI'}
            assert step['amp_scale_after']>=step['amp_scale_before']
            values=step['components']
            assert len(values)==14 and all(math.isfinite(v) for v in values.values())
            calculated=weight['ID_FUSED']*values['id_fused']+weight['TRIPLET_FUSED']*values['triplet_fused']
            for role in ROLES:
                calculated+=weight['ID_BRANCH']*values['id_'+role]+weight['TRIPLET_BRANCH']*values['triplet_'+role]
                calculated+=weight['ID_RESIDUAL']*values['id_residual_'+role]+weight['TRIPLET_RESIDUAL']*values['triplet_residual_'+role]
            loss_error=max(loss_error,abs(calculated-step['loss']));total+=1
        for epoch in training['history']:
            steps=[s for s in training['steps'] if s['epoch']==epoch['epoch']]
            assert len(steps)==epoch['optimizer_steps']
            assert abs(statistics.fmean(s['loss'] for s in steps)-epoch['mean_loss'])<1e-12
        styles_by_run[stem]=[{k:s[k] for k in ('plan','pixel_sha256')} for s in styles]
        ratio=None
        if overfit:
            assert all(s==styles_by_run[stem][0] for s in styles_by_run[stem])
            count=len(fold['source_ids']);smoothing=weight['LABEL_SMOOTHING']
            p=1-smoothing+smoothing/count;q=smoothing/count
            floor=(-p*math.log(p)-(count-1)*q*math.log(q))*(weight['ID_FUSED']+3*weight['ID_BRANCH']+3*weight['ID_RESIDUAL'])
            assert abs(floor-row['gate']['minimum_loss'])<1e-14
            ratio=(training['steps'][-1]['loss']-floor)/(training['steps'][0]['loss']-floor)
            assert abs(ratio-row['gate']['loss_ratio'])<1e-12 and ratio<=.1
        else:
            assert row==read(stem+'_receipt.json')
            assert row['strict_reload_state_sha256']==training['final_state_sha256']
            assert row['strict_reload_all_outputs_bitwise_equal']
            assert row['preflight']['standalone_source_signal_bitwise_equal'] and row['preflight']['original_state_unchanged']
            assert row['preflight']['heldout_image_reads']==0
            assert [s['style_active'] for s in row['preflight']['field_cases']]==[0,1,0]
            assert cpu['files'][row['checkpoint']]['sha256']==row['checkpoint_sha256']
        table.append(dict(run=stem,fold=fold['fold'],endpoint=end,updates=len(styles),live_gradients=203,
                          trainable_parameters=initial['trainable_parameters'],total_parameters=initial['total_parameters'],
                          active_style_steps=training['active_style_steps'],overfit_ratio=ratio,
                          training_seconds=sum(e['elapsed_seconds'] for e in training['history']),peak_reserved_mib=training['peak_reserved_mib']))
    for f,fold in zip(summary['folds'],protocol['folds'],strict=True):
        for end in ENDS:check(f['endpoints'][end],f"m0_fold_{fold['fold']}_{end}",fold,end,False)
        assert f['endpoints']['control']['initialization']==f['endpoints']['source_style']['initialization']
        assert styles_by_run[f"m0_fold_{fold['fold']}_control"]==styles_by_run[f"m0_fold_{fold['fold']}_source_style"]
    for end in ENDS:check(summary['overfit'][end],'m0_overfit_'+end,protocol['folds'][0],end,True)
    assert styles_by_run['m0_overfit_control']==styles_by_run['m0_overfit_source_style'] and total==248
    assert abs(loss_error-cpu['maximum_saved_loss_double_reassembly_error'])<1e-12
    result=dict(status='PASS_LOCAL_COMPLETE_MSVR_STYLE_M0_TEXT',checked_at=datetime.now().astimezone().isoformat(),
                checked_updates=total,all_runs=table,maximum_loss_double_reassembly_error=loss_error,
                files_received=len(intake['files']),received_bytes=sum(r['bytes'] for r in intake['files'].values()),
                summary_sha256=sha(directory/'m0_summary.json'),remote_cpu_sha256=sha(directory/'m0_cpu.json'),script_sha256=sha(__file__),
                scope='Every saved M0 input/plan/loss/receipt verified locally. Checkpoint tensors verified remotely, no local model execution. Engineering only, no retrieval claim.')
    assert not args.output.exists();args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
