"""CPU construction/state loading only; no neural forward, optimizer, or scorer."""
import contextlib
import hashlib
import io
import json
import os
import pathlib
import random
import sys
import numpy as np
import torch

torch.set_num_threads(2)
assert not torch.cuda.is_available()
ROOT = pathlib.Path('/data/gaob/Re-ID/Trifusion')
SCRATCH = pathlib.Path('/data/gaob/Re-ID/prompt_role_state_audit_20260930_tmp').resolve()
dataset = sys.argv[1]
source = ROOT/'comparators/Signal-cd1b0a6'

def allowed_write(path):
    if isinstance(path,int):
        path=os.readlink(f'/proc/self/fd/{path}')
    resolved=pathlib.Path(path).resolve()
    return resolved==SCRATCH or SCRATCH in resolved.parents

def deny_writes(event, args):
    if event == 'open':
        path, mode, flags = args
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND) and not allowed_write(path):
            raise PermissionError(f'Reviewer CPU construction forbids writing {path}')
    if event in ('os.mkdir','os.remove','os.rmdir') and not allowed_write(args[0]):
        raise PermissionError(f'Reviewer CPU construction forbids {event}')
    if event=='os.rename' and not (allowed_write(args[0]) and allowed_write(args[1])):
        raise PermissionError('Reviewer CPU construction forbids rename outside scratch')

sys.addaudithook(deny_writes)
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'modeling'))
sys.path.insert(0,str(source))

# The production constructors call Module.cuda() and CLIP Module.to('cuda').
# Redirect only those module transfers to CPU for state construction. No forward
# runs; actual production Mamba parameterization remains unchanged.
original_to = torch.nn.Module.to
def cpu_to(self, *args, **kwargs):
    if args and args[0] == 'cuda':
        args = ('cpu', *args[1:])
    return original_to(self,*args,**kwargs)
torch.nn.Module.to = cpu_to
torch.nn.Module.cuda = lambda self, device=None: self.cpu()

def state_hash(module):
    digest=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

protocol = json.loads((ROOT/f'logs/official_three_dataset_protocols_20260923/{dataset}.json').read_text())
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    from config import cfg
    cfg.merge_from_file(str(source/'configs'/dataset/'Signal.yml'))
    cfg.defrost()
    cfg.MODEL.PRETRAIN_PATH_T = str(ROOT/'pertrained-model/ViT-B-16.pt')
    cfg.SOLVER.SEED = 42
    cfg.MODEL.USE_A = False
    cfg.MODEL.USE_B = False
    cfg.freeze()
    from modeling import make_frame
    signal = make_frame(cfg,num_class=len(protocol['train_label_map']),
                        camera_num=4 if dataset=='RGBNT201' else 8,view_num=0)
    tr = json.loads((ROOT/f'trained-model/prompt_role_state_20260930_prompt_reset_roles_{dataset}_seed42_full/training.json').read_text())
    base_state = torch.load(tr['initializer']['author_checkpoint'],map_location='cpu',weights_only=True)
    baseline_load = signal.load_state_dict(base_state,strict=True)
    assert not baseline_load.missing_keys and not baseline_load.unexpected_keys
    assert state_hash(signal) == tr['initializer']['signal_state_sha256']
    del base_state
    from trifusion.prompt_role_state import PromptRoleTriFusion
    from trifusion.experts.mamba import production_mamba_factory
    rows = []
    for mode in ('reset','carry'):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        model = PromptRoleTriFusion(signal,num_classes=len(protocol['train_label_map']),
            grid=(16,8) if dataset=='RGBNT201' else (8,16),width=128,m1=True,m2=True,m3=False,
            mamba_factory=production_mamba_factory,query_mode='context',auxiliary_target='none',prompt_mode=mode)
        run = ROOT/f'trained-model/prompt_role_state_20260930_prompt_{mode}_roles_{dataset}_seed42_full'
        tr = json.loads((run/'training.json').read_text())
        initial_hash = state_hash(model)
        assert initial_hash == tr['initializer']['initial_model_state_sha256']
        named = {n:p for n,p in model.named_parameters() if p.requires_grad}
        assert sum(p.numel() for p in named.values()) == tr['initializer']['trainable_parameters']
        assert all(not p.requires_grad for p in signal.parameters())
        results = []
        for stage,path in (('m0',run.parent/(run.name.replace('_full','_m0'))/'m0_reload_probe.pth'),('full',run/'best_map.pth')):
            payload = torch.load(path,map_location='cpu',weights_only=True)
            state = model.state_dict()
            expected = {n for n in state if not n.startswith(('backbone.signal.','teacher.'))}
            assert set(payload['state']) == expected
            state.update(payload['state'])
            loaded = model.load_state_dict(state,strict=True)
            assert not loaded.missing_keys and not loaded.unexpected_keys
            assert all(torch.equal(model.state_dict()[n],v) for n,v in payload['state'].items())
            assert state_hash(signal) == tr['initializer']['signal_state_sha256']
            results.append({'stage':stage,'strict_load':True,'missing_keys':[],'unexpected_keys':[],
                            'saved_tensor_keys':len(expected),'tensors_equal_after_load':True,'frozen_signal_unchanged':True})
        rows.append({'dataset':dataset,'mode':mode,'initial_model_state_sha256_recomputed':initial_hash,
                     'initial_hash_matches_recorded':True,'trainable_parameter_tensors':len(named),
                     'trainable_elements':sum(p.numel() for p in named.values()),'loads':results})
        del model
print(json.dumps({'dataset':dataset,'device':'cpu','cuda_available':torch.cuda.is_available(),
    'scope':'real model constructors and strict state reload; no neural forward; CUDA module transfers redirected to CPU',
    'rows':rows,'constructor_stdout':captured.getvalue()},indent=2))
