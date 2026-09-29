from pathlib import Path
import ast
from datetime import datetime
import glob
import hashlib
import json
import os
import re
import torch

root = Path('/data/gaob/Re-ID/Trifusion')
source = root / 'comparators/Signal-cd1b0a6'
source_files = {'RGBNT201': 'RGBNT201.py', 'RGBNT100': 'RGBNT100.py', 'MSVR310': 'msvr310.py'}
weights = {
 'RGBNT201': ('RGBNT201_PlainBaseline_50.pth', '789e5e14aacd74ad122aad701389eb216ca5b4fda92687e27351a513023b4407'),
 'RGBNT100': ('RGBNT100_PlainBaseline_30.pth', '299a28bfb3e3180eeae0736cf8638cd162525dce0f2192a940b62b97f6e67dcd'),
 'MSVR310': ('MSVR310_PlainBaseline_50.pth', '69c5e71b75036d7216ece3ff84450f0052f5e70dfaba46bf73f3e1d40992bb37'),
}
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
rows = []
for dataset in source_files:
    protocol_path = root / f'logs/official_three_dataset_protocols_20260923/{dataset}.json'
    protocol = json.loads(protocol_path.read_text())
    data_root = Path(protocol['dataset_root'])
    train = protocol['records']['train']
    first = data_root / train[0]['paths'][0]
    depth = {'RGBNT201': 1, 'RGBNT100': 0, 'MSVR310': 2}[dataset]
    train_dir = first.parents[depth]
    dataset_file = source / 'data/datasets' / source_files[dataset]
    syntax = ast.parse(dataset_file.read_text())
    cls, = [node for node in syntax.body if isinstance(node, ast.ClassDef)]
    method, = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_process_dir']
    env = {'glob': glob, 'osp': os.path, 'os': os, 're': re}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(dataset_file), 'exec'), env)
    author = env['_process_dir'](None, str(train_dir), relabel=True)
    by_path = {}
    for images, label, camera, view in author:
        key = images if isinstance(images, str) else images[0]
        assert key not in by_path
        by_path[key] = (label, camera, view)
    assert len(by_path) == len(train)
    mismatches = []
    author_map = {}
    for row in train:
        key = str(data_root / row['paths'][0])
        label, camera, view = by_path[key]
        assert camera == row['camera'] and view == row['view']
        identity = str(row['identity'])
        if identity in author_map:
            assert author_map[identity] == label
        author_map[identity] = label
        if label != row['label']:
            mismatches.append({'identity': identity, 'author_label': label, 'protocol_label': row['label']})
    file_name, expected_sha = weights[dataset]
    checkpoint = root / 'pertrained-model' / file_name
    actual_sha = sha(checkpoint)
    assert actual_sha == expected_sha
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    suffixes = [''] if dataset == 'RGBNT201' else ['_r', '_n', '_t']
    feature_width = 1536 if dataset == 'RGBNT201' else 512
    classifier_keys = [f'classifier{suffix}.weight' for suffix in suffixes]
    assert {key for key in state if key.startswith('classifier')} == set(classifier_keys)
    assert all(list(state[key].shape) == [len(protocol['train_label_map']), feature_width] for key in classifier_keys)
    head_keys = classifier_keys + [f'bottleneck{suffix}.{name}' for suffix in suffixes for name in ('weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked')]
    row = {'dataset': dataset, 'protocol_sha256': sha(protocol_path), 'dataset_source_sha256': sha(dataset_file),
           'author_current_training_directory': str(train_dir), 'training_records': len(train),
           'author_current_label_map': author_map, 'protocol_label_map': protocol['train_label_map'],
           'mismatched_label_records': len(mismatches), 'mismatched_label_identities': len({item['identity'] for item in mismatches}),
           'checkpoint_sha256': actual_sha, 'classifier_shapes': {key: list(state[key].shape) for key in classifier_keys}, 'author_head_layout': 'concat' if dataset == 'RGBNT201' else 'three_per_modality_heads',
           'head_state': {key: {'shape': list(state[key].shape), 'dtype': str(state[key].dtype),
                                'l2_norm': float(state[key].double().norm())} for key in head_keys}}
    rows.append(row)
    del state
report = {'status': 'CPU_SOURCE_LABEL_AND_STORED_HEAD_AUDIT_COMPLETE', 'at': datetime.now().astimezone().isoformat(),
          'rows': rows, 'make_model_source_sha256': sha(source / 'modeling/make_model.py'),
          'role_model_source_sha256': sha(root / 'modeling/trifusion/correspondence_roles.py'),
          'role_runner_source_sha256': sha(root / 'tools/run_correspondence_roles.py'),
          'code_facts': {'author_id_input': {'RGBNT201': 'raw RGB/NIR/TIR global concat', 'RGBNT100': 'three separate raw modality globals', 'MSVR310': 'three separate raw modality globals'},
                         'correspondence_id_input': 'L2 normalized fused before newly initialized neck/classifier',
                         'inference': 'pre-neck embedding; no classifier or BN running-stat use'},
          'boundary': 'No model construction, image decoding, forward, gradient, GPU, official re-evaluation, checkpoint selection or training change. Exact current author _process_dir AST executed on all training paths. This reconstructs current source labels, not proof of historical training mapping or an explanation of retrieval loss. Stored head shape compatibility does not establish normalized-input behavior compatibility.'}
out = root / '.git/correspondence_head_source_audit_646_20260929.json'
assert not out.exists()
out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'status': report['status'], 'at': report['at'], 'rows': [{key: row[key] for key in ('dataset', 'training_records', 'mismatched_label_records', 'mismatched_label_identities', 'classifier_shapes')} for row in rows]}))
