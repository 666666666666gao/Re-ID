#!/usr/bin/env python3
"""CPU contract, complete source-plan and vehicle statistic-formula checks."""
import argparse
import ast
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch

from tools.train_msvr310_source_style import context,ROOT
from tools.train_msvr310_signal_oof import sha256,write_json


def main(args):
    config,_base,_cfg,environment,protocol,_baseline,metadata=context(args.config)
    from trifusion.source_style_v27 import make_style_plan,mix_patch_statistics
    assert not torch.cuda.is_initialized()
    original=ast.parse((ROOT/'modeling/trifusion/source_style_v27.py').read_text())
    new=ast.parse((ROOT/'modeling/trifusion/source_style_msvr.py').read_text())
    old_class=next(n for n in original.body if isinstance(n,ast.ClassDef))
    new_class=next(n for n in new.body if isinstance(n,ast.ClassDef))
    old_class.name=new_class.name
    class VehicleShape(ast.NodeTransformer):
        def visit_Tuple(self,node):
            if all(isinstance(e,ast.Constant) for e in node.elts) and [e.value for e in node.elts]==[768,16,8]:
                node.elts[1].value=8;node.elts[2].value=16
            return self.generic_visit(node)
    assert ast.dump(VehicleShape().visit(old_class))==ast.dump(new_class)
    batches=0;active=[]
    for fold,md in zip(protocol['folds'],metadata['folds'],strict=True):
        count=0;seen=set()
        for step,row in enumerate(md['batches']):
            ids=row['record_indices'];assert set(ids)<=set(fold['source_record_indices'])
            seen.update(ids)
            cameras=np.array([protocol['records'][i]['camera'] for i in ids])
            plan=make_style_plan(cameras,fold=fold['fold'],step=step)
            assert plan==row['style_plan'];count+=int(plan['active']);batches+=1
        assert seen==set(fold['source_record_indices'])
        active.append(count)
    assert batches==780 and active==[129,127,137]
    # Actual vehicle spatial axes, including different means/stds and a flat channel.
    value=torch.arange(8*3*8*16,dtype=torch.float32).reshape(8,3,8,16)/1024
    value[0,0]=3
    plan=make_style_plan(np.arange(8),fold=0,step=0,force_active=True)
    mixed,_=mix_patch_statistics(value,plan)
    x=value.numpy();mean=x.mean((2,3),keepdims=True);std=np.sqrt(x.var((2,3),keepdims=True)+1e-6)
    weight=np.array(plan['coefficients'],dtype=np.float32)[:,None,None,None];donor=plan['donors']
    expected=(x-mean)/std*(weight*std+(1-weight)*std[donor])+weight*mean+(1-weight)*mean[donor]
    error=float(np.max(np.abs(mixed.numpy()-expected)))
    assert error<2e-6 and np.isfinite(mixed.numpy()).all()
    assert not torch.cuda.is_initialized()
    result=dict(status='PASS_MSVR_STYLE_CPU_CONTRACT',config_sha256=sha256(args.config),
                checked_at=datetime.now().astimezone().isoformat(),environment=environment,
                complete_source_batches=batches,active_batches=active,source_record_exposure_complete=True,
                vehicle_class_only_name_and_stem_shape_changed=True,numpy_formula_max_error=error,
                model_forwards=0,optimizer_updates=0,heldout_image_reads=0,
                project_source_files=len(config['project_source_file_sha256']))
    assert not args.output.exists();write_json(args.output,result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
