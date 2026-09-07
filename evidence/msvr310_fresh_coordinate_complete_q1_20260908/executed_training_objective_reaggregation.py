from pathlib import Path
import argparse
import csv
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
p = args.input_dir
s = json.loads((p/'q1/summary.json').read_bytes())
assert s['status'] in ('Q1_PASS', 'Q1_FAIL') and s['optimizer_steps'] == 1560
result, epochs = [], []
for fold in s['folds']:
    for end, r in fold['endpoints'].items():
        tr = r['training']
        rows = [json.loads(line) for line in (p/f"q1/fold_{fold['fold']}_{end}/memory_steps.jsonl").read_text().splitlines()]
        assert len(rows) == len(tr['steps']) == 260
        windows = {}
        for name, indices in [('all', list(range(260))), ('history', [i for i, x in enumerate(rows) if x['memory']]), ('last65', list(range(195, 260)))]:
            components = {k: float(np.mean([tr['steps'][i]['components'][k] for i in indices])) for k in tr['steps'][0]['components']}
            windows[name] = dict(updates=len(indices), mean_total_loss=float(np.mean([tr['steps'][i]['loss'] for i in indices])), components=components,
                                 batch_triplet_mean=float(np.mean([rows[i]['original_triplet'] for i in indices])),
                                 stale_expanded_mean=float(np.mean([rows[i]['stale_statistics']['expanded_triplet'] for i in indices])),
                                 fresh_expanded_mean=float(np.mean([rows[i]['fresh_statistics']['expanded_triplet'] for i in indices])),
                                 stale_wrong_order_anchor_exposures=sum(rows[i]['stale_statistics']['expanded_wrong_order_anchors'] for i in indices),
                                 fresh_wrong_order_anchor_exposures=sum(rows[i]['fresh_statistics']['expanded_wrong_order_anchors'] for i in indices),
                                 anchor_exposures=64*len(indices))
        for ep in range(1, 21):
            selected = [i for i, x in enumerate(tr['steps']) if x['epoch'] == ep]
            assert len(selected) == 13
            er = dict(fold=fold['fold'], endpoint=end, epoch=ep, updates=13, mean_total_loss=float(np.mean([tr['steps'][i]['loss'] for i in selected])))
            er.update({k: float(np.mean([tr['steps'][i]['components'][k] for i in selected])) for k in tr['steps'][0]['components']})
            er.update(batch_triplet=float(np.mean([rows[i]['original_triplet'] for i in selected])),
                      stale_expanded=float(np.mean([rows[i]['stale_statistics']['expanded_triplet'] for i in selected])),
                      fresh_expanded=float(np.mean([rows[i]['fresh_statistics']['expanded_triplet'] for i in selected])))
            epochs.append(er)
        result.append(dict(fold=fold['fold'], endpoint=end, windows=windows, extra_fresh_record_forwards=tr['extra_fresh_role_record_forwards'],
                           extra_fixed_drift_record_forwards=tr['extra_probe_record_forwards'],
                           elapsed_epochs_seconds=sum(x['elapsed_seconds'] for x in tr['history'])))
assert len(epochs) == 120
args.output_dir.mkdir()
output = dict(scope='All1560 recorded training updates; losses and wrong-order anchor exposures are runtime training diagnostics, not retrieval AP or classification accuracy.', endpoints=result)
(args.output_dir/'complete_training_loss_and_history.json').write_bytes((json.dumps(output, indent=2)+'\n').encode())
with (args.output_dir/'all120_epoch_training_objectives.csv').open('x', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(epochs[0]))
    writer.writeheader()
    writer.writerows(epochs)
print(json.dumps(dict(status='COMPLETE_TRAINING_OBJECTIVE_REAGGREGATION', epochs=len(epochs), updates=1560)))
