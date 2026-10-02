"""Witness the common initialization of all nine shared/private evidence flows."""
import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_shared_private_evidence as run
from tools.queue_correspondence_roles import BASELINES, PROTOCOLS, SOURCE, WEIGHTS

DATASETS = ('RGBNT201', 'RGBNT100', 'MSVR310')
CONDITIONS = {flow: ('low_lr', 'global_only' if flow == 'global_only' else 'roles')
              for flow in run.FLOWS}


def options(dataset, variant):
    update, readout = CONDITIONS[variant]
    filename, digest = BASELINES[dataset]
    return argparse.Namespace(dataset=dataset, visual_update=update, readout=readout,
        mode='m0', signal_source=SOURCE, clip_weight=WEIGHTS / 'ViT-B-16.pt',
        baseline_checkpoint=WEIGHTS / filename, baseline_sha256=digest,
        protocol=PROTOCOLS / f'{dataset}.json', seed=42, epochs=50,
        output_dir=ROOT / 'trained-model/shared_private_initialization_unused')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    preflight = json.loads(args.preflight.read_text())
    assert preflight['status'] == 'COMPLETE'
    sources = preflight['source_sha256']
    assert all(run.control.runner.sha256(ROOT / path) == digest for path, digest in sources.items())
    rows = []
    for dataset in DATASETS:
        protocol = run.control.runner.read_protocol(PROTOCOLS / f'{dataset}.json', dataset)
        reference = None
        common_digest = None
        for variant, (update, readout) in CONDITIONS.items():
            run.activate(variant)
            model, _, _, binding = run.control.build(options(dataset, variant), protocol)
            common = {f'{module}.{name}': value.detach().cpu().clone()
                      for module in ('backbone', 'neck', 'classifier')
                      for name, value in getattr(model, module).state_dict().items()
                      if not (module == 'backbone' and name.startswith('private_adapters.'))}
            if reference is None:
                reference, common_digest = common, binding['common_initializer_sha256']
            assert set(common) == set(reference)
            assert all(torch.equal(common[key], reference[key]) for key in common)
            assert binding['common_initializer_sha256'] == common_digest
            visual = model.backbone.signal.clip_vision_encoder.base
            assert all(p.requires_grad == (update == 'low_lr') for p in visual.parameters())
            assert all(p.dtype == torch.float32 for p in visual.parameters())
            assert isinstance(model, run.control.SharedGlobalOnly) == (readout == 'global_only')
            rows.append({'dataset': dataset, 'variant': variant, 'binding': binding,
                         'common_states_bitwise_equal': True,
                         'visual_parameter_tensors': len(list(visual.parameters())),
                         'trainable_tensors': sum(p.requires_grad for p in model.parameters())})
            del common, model
            gc.collect()
            torch.cuda.empty_cache()
        paired = [row for row in rows if row['dataset'] == dataset and row['variant'] != 'global_only']
        assert len(paired) == 2
        assert paired[0]['binding']['initial_model_state_sha256'] == paired[1]['binding']['initial_model_state_sha256']
        assert paired[0]['binding']['trainable_parameters'] == paired[1]['binding']['trainable_parameters']
        del reference
    assert all(run.control.runner.sha256(ROOT / path) == digest for path, digest in sources.items())
    result = {'status': 'MATCHED_COMMON_INITIALIZATION_PASS', 'rows': rows,
              'source_snapshot_sha256': sources, 'preflight_sha256': run.control.runner.sha256(args.preflight),
              'completed_at': datetime.now().astimezone().isoformat(),
              'source_sha256': run.control.runner.sha256(Path(__file__)),
              'entry_sha256': run.control.runner.sha256(ROOT / 'tools/run_shared_private_evidence.py'),
              'scope': 'Actual nine model constructions; bitwise shared backbone excluding private_adapters, neck and classifier equality; coupled/separated full initial state and trainable parameter count equality. No forward, optimizer or retrieval evaluation.'}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'endpoints': len(rows), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
