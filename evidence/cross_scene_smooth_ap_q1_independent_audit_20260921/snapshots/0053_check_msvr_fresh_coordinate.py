"""CPU-only contract, full sampler admission replay and exact synthetic math."""
import argparse
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import torch

from tools.train_msvr_fresh_coordinate import context
from tools.msvr_instance_memory import InstanceMemory, check_math
from tools.train_msvr310_signal_oof import sha256, write_json


def run(args):
    spec, (_config, _base, _cfg, environment, protocol, _b0, metadata) = context(args.config)
    math = check_math()
    folds = []
    for fold, md in zip(protocol['folds'], metadata['folds'], strict=True):
        source = [protocol['records'][i] for i in fold['source_record_indices']]
        memory = InstanceMemory(source)
        expected = OrderedDict()
        dummy = torch.zeros(64, 2)
        counts = [];steps = []
        for step, row in enumerate(md['batches']):
            indices = row['record_indices']
            assert len(indices) == 64 and set(indices) <= set(fold['source_record_indices'])
            values, actual = memory.read(step, indices, dummy)
            for i in [i for i, age in expected.items() if step-age > 8]:
                del expected[i]
            wanted = [dict(record_index=i, identity=protocol['records'][i]['identity'],
                           scene=protocol['records'][i]['scene'], age=step-age, stored_step=age)
                      for i, age in expected.items() if i not in indices]
            assert actual == wanted and values.shape == (len(wanted), 2)
            counts.append(len(actual))
            steps.append(dict(step=step+1, historical_records=len(actual), ages=[r['age'] for r in actual]))
            if step >= 65:
                memory.update(step, indices, dummy)
                for i in indices:
                    expected.pop(i, None);expected[i] = step
                while len(expected) > 512:
                    expected.popitem(last=False)
        assert len(counts) == 260 and sum(counts[:66]) == 0 and max(counts) <= 512
        assert sum(counts) > 0
        folds.append(dict(fold=fold['fold'], steps=steps, total_historical_candidate_records=sum(counts),
                          maximum_historical_records=max(counts), active_steps=sum(x > 0 for x in counts)))
    assert not torch.cuda.is_initialized()
    result = dict(status='PASS_FRESH_COORDINATE_CPU_CONTRACT', config_sha256=sha256(args.config),
                  verified_at=datetime.now().astimezone().isoformat(), math=math, folds=folds,
                  full_source_batches=780, source_only_identity_filter=True, model_forwards=0,
                  optimizer_updates=0, heldout_image_reads=0, environment=environment)
    assert not args.output.exists();write_json(args.output, result)
    print({k: v for k, v in result.items() if k != 'folds'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
