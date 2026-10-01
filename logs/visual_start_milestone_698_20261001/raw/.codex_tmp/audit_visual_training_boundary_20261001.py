"""Source and saved-shape audit only; no model construction or training."""
from datetime import datetime
import hashlib
import json
from math import prod
from pathlib import Path
import re

ROOT = Path('/data/gaob/Re-ID/Trifusion')
CAMPAIGN = ROOT / 'logs/visual_start_roles_20261001_v1'
DEST = ROOT / '.codex_tmp/visual_training_boundary_20261001'
assert not DEST.exists()
manifest_path = CAMPAIGN / 'manifest.json'
manifest = json.loads(manifest_path.read_text())
assert len(manifest['source_sha256']) == 221
assert all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
           for path, digest in manifest['source_sha256'].items())
sources = (
    'modeling/trifusion/correspondence_roles.py',
    'tools/run_visual_start_roles.py',
    'tools/run_role_global_tokens.py',
    'tools/run_correspondence_context_identity.py',
    'tools/run_correspondence_roles.py',
    'tools/official_three_dataset_data.py',
    'tools/run_signal_preserving_v5.py',
    'comparators/Signal-cd1b0a6/config/defaults.py',
    'comparators/Signal-cd1b0a6/modeling/meta_arch.py',
    'comparators/Signal-cd1b0a6/solver/make_optimizer.py',
    *(f'comparators/Signal-cd1b0a6/configs/{name}/Signal.yml'
      for name in ('RGBNT201', 'RGBNT100', 'MSVR310')),
)
bindings = {name: {'sha256': manifest['source_sha256'][name]}
            for name in sources}
text = {name: (ROOT / name).read_text() for name in sources}
model = text[sources[0]]
trainer = text['tools/run_correspondence_context_identity.py']
checkpoint = text['tools/run_correspondence_roles.py']
assert 'for parameter in signal.parameters():\n            parameter.requires_grad_(False)' in model
assert 'return output + sum(deltas) / 3' in model
assert 'if parameter.requires_grad' in trainer
assert 'torch.optim.AdamW(parameters, lr=config["OPTIMIZATION"]["NEW_MODULE_LR"]' in trainer
assert 'frozen_signal_sha256 == _module_state_sha256(model.backbone.signal)' in trainer
assert 'not name.startswith("backbone.signal.")' in checkpoint
assert '_C.MODEL.FROZEN = False' in text['comparators/Signal-cd1b0a6/config/defaults.py']
optimizer = text['comparators/Signal-cd1b0a6/solver/make_optimizer.py']
assert 'if not cfg.MODEL.FROZEN:' in optimizer and 'lr = 0.000005' in optimizer
inputs_path = Path(manifest['inputs_path'])
assert hashlib.sha256(inputs_path.read_bytes()).hexdigest() == manifest['inputs_sha256']
inputs = json.loads(inputs_path.read_text())
witness_path = Path(manifest['initialization_witness_path'])
assert hashlib.sha256(witness_path.read_bytes()).hexdigest() == manifest['initialization_witness_sha256']
witness = json.loads(witness_path.read_text())
rows = []
for name in ('RGBNT201', 'RGBNT100', 'MSVR310'):
    meta = inputs['datasets'][name]
    elements = sum(prod(item['shape']) for item in meta['public_visual_tensors'])
    assert len(meta['public_visual_tensors']) == 152 and elements == 86140416
    row = next(item for item in witness['rows'] if item['dataset'] == name)
    config = text[f'comparators/Signal-cd1b0a6/configs/{name}/Signal.yml']
    rows.append({'dataset': name, 'frozen_visual_saved_state_tensors': 152,
                 'frozen_visual_saved_state_elements': elements,
                 'current_trainable_parameters': row['trainable_parameters'],
                 'native_signal_config': {field: re.search(r'^\s*' + field + r':\s*([^\n]+)',
                                                          config, re.MULTILINE).group(1).strip()
                                          for field in ('BASE_LR', 'MAX_EPOCHS', 'IMS_PER_BATCH',
                                                        'NUM_INSTANCE', 'WARMUP_ITERS', 'OPTIMIZER_NAME')}})
record = {
    'schema': 'trifusion-visual-training-boundary-source-audit-v1',
    'audited_at': datetime.now().astimezone().isoformat(),
    'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    'runtime_source_count_verified': 221, 'source_bindings': bindings,
    'inputs_sha256': manifest['inputs_sha256'],
    'initialization_witness_sha256': manifest['initialization_witness_sha256'],
    'rows': rows,
    'established_facts': [
        'Current CrossLayerAdaptedCLIP explicitly freezes every Signal parameter. Public/reid visual conditions change frozen weights, not trainability.',
        'M1 hook outputs alter the downstream shared input. There is no whole-forward no_grad cutoff; gradients can propagate through frozen later blocks to adapters. Shared-global is therefore trainable through adapters, not an independent fixed baseline.',
        'Current optimizer includes requires_grad parameters only in one AdamW group; the training loop sets NEW_MODULE_LR times the same five-epoch warmup/cosine multiplier.',
        'Current loader explicitly uses training batch64/K8. Native Signal configurations differ on RGBNT100 and MSVR310; matching50 epochs alone does not equalize their schedule or sampling.',
        'Native Signal defaults FROZEN=False and only invokes adapter-only freezing when that flag is true. Its optimizer assigns CLIP base, non-adapter parameters5e-6 when not frozen. This is source behavior, not an audit of every historical publication checkpoint training run.',
        'Current checkpoint_state excludes all backbone.signal.* tensors and current training asserts the whole Signal hash unchanged. Simply switching requires_grad=True would violate both the training invariant and checkpoint/reload contract.',
    ],
    'requirements_for_any_future_visual_update_control': [
        'Keep the present registered six-end campaign unchanged and complete its paired analysis first.',
        'An independently registered visual-update study must define its exact trainable visual tensors and optimizer groups, save and strictly reload changed visual state, and check only genuinely frozen tensors for immutability.',
        'Separate ordinary visual fine-tuning/global-only effects from role contribution with matched initialization and training boundaries. Do not label visual fine-tuning a new algorithmic contribution.',
    ],
    'limits': [
        '86140416 is a saved visual-state element count derived from pinned tensor shapes, not measured gradient support, optimizer updates, FLOPs or a proven capacity bottleneck.',
        'Source/training-boundary differences are established. Their performance causality is not: this audit executes no model, optimization, evaluation or new experiment.',
        'No current source, gate, seed, loader, checkpoint, environment or training process was changed.',
    ],
}
DEST.mkdir()
(DEST / 'SOURCE_AUDIT.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({'audit': str(DEST / 'SOURCE_AUDIT.json'), 'audited_at': record['audited_at'],
                  'sha256': hashlib.sha256((DEST / 'SOURCE_AUDIT.json').read_bytes()).hexdigest(),
                  'selected_sources': len(bindings), 'runtime_sources_verified': 221,
                  'rows': rows}, indent=2))
