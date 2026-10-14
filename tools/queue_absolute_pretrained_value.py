"""One registered RGBNT201 experiment; prepare, real M0, fresh50, strict."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import queue_pretrained_value_delta as previous
from tools.run_absolute_pretrained_value import SCHEMA

base=previous.base
SCOPE=ROOT/'refine-logs/absolute_pretrained_value_20261014_958/SOURCE_SCOPE.json'
CONTROL=ROOT/'logs/pretrained_value_training951/pretrained_value_training951_uniform_reference/ACCEPTED_RESULT.json'


def source_map():
    value=json.loads(SCOPE.read_text())['source_sha256']
    assert all(base.sha(ROOT/name)==digest for name,digest in value.items())
    return value


def coordinate(campaign):
    assert str(ROOT)=='/data/gaob/Re-ID/Trifusion' and os.environ['CUDA_VISIBLE_DEVICES']=='0,1'
    assert not campaign.exists()
    assert shutil.disk_usage(ROOT).free>=2*1024**3+3*384*1024**2
    base.SCHEMA=SCHEMA
    base.source_map=source_map
    base.output_dir=previous.previous.output_dir
    control=json.loads(CONTROL.read_text())
    assert control['status']=='VERIFIED_COMPLETE' and control['value_reference']=='uniform_reference'
    sources=source_map()
    campaign.mkdir()
    child=campaign/(campaign.name+'_absolute_reading');child.mkdir()
    state=dict(status='RUNNING',controller_pid=os.getpid(),started_at=base.queue.stamp(),
               arms=[],expected=1,accepted=0)
    arm=dict(status='RUNNING',value_reference='uniform_reference',value_reading='absolute',
             controller_pid=os.getpid(),steps=[],campaign=str(child),started_at=base.queue.stamp())
    state['arms'].append(arm)
    manifest=dict(schema=SCHEMA,seed=42,epochs=50,dataset='RGBNT201',variant='semantic',
        value_reference='uniform_reference',value_reading='absolute',source_sha256=sources,
        initialization_sha256={},physical_gpus=[0,1],max_parallel_jobs=1,poll_seconds=240,
        policy=previous.POLICY,control_receipt_sha256=base.sha(CONTROL))
    base.queue.write(campaign/'SOURCE_SEAL.json',sources)
    base.queue.write(child/'manifest.json',manifest)
    base.queue.write(child/'campaign.json',arm)
    base.queue.write(campaign/'campaign.json',state)
    for mode in ('prepare','m0','train','evaluate'):
        output=previous.previous.output_dir(child,'m0' if mode=='m0' else 'full','RGBNT201','semantic')
        assert mode=='evaluate' or not output.exists()
        command=previous.command(mode,'uniform_reference',child,output)
        command[2]=str(ROOT/'tools/run_absolute_pretrained_value.py')
        row=dict(mode=mode,command=command,output_dir=str(output));arm['steps'].append(row)
        code=previous.previous.run_logged(child,arm,row,mode+'.log')
        if code:
            state.update(status='FAILED',failed_stage=mode,completed_at=base.queue.stamp())
            base.queue.write(campaign/'campaign.json',state);return code
        if mode=='prepare':
            binding=base.expected_binding(child,'RGBNT201','semantic')
            original=control['initializer']
            for key in ('initial_model_state_sha256','shared_initializer_sha256','visual_initial_sha256',
                        'camera_initial_sha256','trainable_parameters','trainable_parameter_tensors',
                        'batch_size','num_instances','head_names','cfg_yaml','objective_gradient_policy'):
                assert binding[key]==original[key],key
            path=child/'initialization/RGBNT201_semantic.json'
            manifest['initialization_sha256'][str(path)]=base.sha(path)
            base.queue.write(child/'manifest.json',manifest)
        elif mode=='m0':
            accepted=previous.previous.verify_m0(child,'RGBNT201','semantic')
            assert accepted['production_m0_diagnostics']['pretrained_value_role_global_isolated']
            base.queue.write(child/'M0_ACCEPTED.json',accepted)
        base.queue.write(campaign/'campaign.json',state)
    accepted=base.verify(child,'RGBNT201','semantic')
    accepted.update(value_reference='uniform_reference',value_reading='absolute')
    base.queue.write(child/'ACCEPTED_RESULT.json',accepted)
    arm.update(status='COMPLETE',result=accepted,completed_at=base.queue.stamp())
    base.queue.write(child/'campaign.json',arm)
    state.update(status='COMPLETE',accepted=1,results=[accepted],completed_at=base.queue.stamp())
    base.queue.write(campaign/'campaign.json',state)
    base.queue.write(campaign/'ACCEPTED_MATRIX.json',dict(expected=1,accepted=1,rows=[accepted]))
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    args=parser.parse_args();sys.exit(coordinate(args.campaign.resolve()))
