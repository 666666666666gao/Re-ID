"""Describe role-proposed source relations using sealed census arrays only."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROLES = ('cnn', 'transformer', 'mamba')
OUTPUTS = ('fused',) + ROLES
WIDTHS = {'fused': 7680, **{r: 4608 for r in ROLES}}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    started = time.perf_counter()
    spec = json.loads(args.contract.read_bytes())
    assert spec['schema'] == 'msvr310-role-relation-coverage-v1'
    assert spec['optimizer_updates'] == 0 and spec['model_forwards'] == 0
    assert spec['outputs'] == list(OUTPUTS) and spec['margin'] == .3
    for path, expected in spec['input_sha256'].items():
        assert sha(Path(path)) == expected, path
    source = Path(spec['source_root'])
    summary = json.loads((source/'summary.json').read_bytes())
    protocol = json.loads(Path(spec['protocol']).read_bytes())
    assert len(summary['conditions']) == 27
    assert summary['status'] == 'COMPLETE_FROZEN_SOURCE_EXTRACTION'
    assert not args.output.exists()
    args.output.mkdir()
    receipts = {}
    arrays = {}
    files = {}
    for condition in summary['conditions']:
        name = condition['directory']
        receipt_path = source/name/'receipt.json'
        receipt = json.loads(receipt_path.read_bytes())
        assert receipt == {k:v for k,v in condition.items() if k != 'directory'}
        receipts[name] = receipt
        files[str(receipt_path)] = dict(bytes=receipt_path.stat().st_size, sha256=sha(receipt_path))
        for output in OUTPUTS:
            path = source/name/f'{output}.npy'
            expected = receipt['files'][path.name]
            assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256']
            x = np.load(path, allow_pickle=False).astype(np.float64)
            assert x.shape == (receipt['records'], WIDTHS[output]) and np.isfinite(x).all()
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            assert (norms > 0).all()
            arrays[name, output] = x/norms
            files[str(path)] = expected
    condition_rows = []
    with (args.output/'all_query_relations.jsonl').open('x', encoding='utf-8') as stream:
        for condition in summary['conditions']:
            name = condition['directory']
            clean = f"fold_{condition['fold']}_{condition['state']}_clean"
            indices = receipts[name]['record_indices']
            fold = protocol['folds'][condition['fold']]
            assert indices == receipts[clean]['record_indices'] == fold['source_record_indices']
            ids = np.array([protocol['records'][i]['identity'] for i in indices])
            scenes = np.array([protocol['records'][i]['scene'] for i in indices])
            assert set(ids) == set(fold['source_ids']) and not set(ids) & set(fold['heldout_ids'])
            distances = {o:np.sqrt(np.maximum(0.,2-2*(arrays[name,o]@arrays[clean,o].T))) for o in OUTPUTS}
            for mode in ('identity_exclude_record','cross_scene'):
                counts = Counter()
                for q in range(len(indices)):
                    positive = (ids == ids[q])
                    positive[q] = False
                    if mode == 'cross_scene':
                        positive &= scenes != scenes[q]
                    negative = ids != ids[q]
                    pp, nn = np.flatnonzero(positive), np.flatnonzero(negative)
                    row = dict(condition=name, protocol=mode, query_position=q, record_index=indices[q], identity=int(ids[q]), scene=int(scenes[q]), positive_count=len(pp), negative_count=len(nn), eligible=bool(len(pp)))
                    counts['all_query_memberships'] += 1
                    if len(pp):
                        counts['eligible_query_memberships'] += 1
                        hp = {o:int(pp[np.argmax(distances[o][q,pp])]) for o in OUTPUTS}
                        hn = {o:int(nn[np.argmin(distances[o][q,nn])]) for o in OUTPUTS}
                        df = distances['fused'][q]
                        full = max(0.,float(df[hp['fused']]-df[hn['fused']]+.3))
                        proposed_p = sorted({hp[o] for o in ROLES})
                        proposed_n = sorted({hn[o] for o in ROLES})
                        subset = max(0.,float(df[proposed_p].max()-df[proposed_n].min()+.3))
                        # These are a subset of the same full source candidates.
                        assert subset <= full + 1e-12
                        extra_n = [j for j in proposed_n if j != hn['fused']]
                        extra_p = [j for j in proposed_p if j != hp['fused']]
                        supported_n = [j for j in extra_n if df[hp['fused']]-df[j]+.3 > 0]
                        wrong_n = [j for j in extra_n if df[hp['fused']] >= df[j]]
                        unique_role = {o:hn[o] not in {hn[r] for r in ROLES if r != o} for o in ROLES}
                        row.update(hardest_positive={o:indices[j] for o,j in hp.items()}, nearest_negative={o:indices[j] for o,j in hn.items()}, role_positive_union=[indices[j] for j in proposed_p], role_negative_union=[indices[j] for j in proposed_n], full_fused_hinge=full, role_subset_fused_hinge=subset, extra_negative_records=[indices[j] for j in extra_n], extra_positive_records=[indices[j] for j in extra_p], extra_negative_fused_hinge_active=[indices[j] for j in supported_n], extra_negative_fused_nonpositive_margin=[indices[j] for j in wrong_n], role_unique_negative=unique_role, negative_union_identities=len(set(ids[proposed_n].tolist())), negative_union_scenes=len(set(scenes[proposed_n].tolist())), contains_fused_negative=hn['fused'] in proposed_n, contains_fused_positive=hp['fused'] in proposed_p, fused_positive_extremum_ties=int(np.count_nonzero(df[pp] == df[hp['fused']])), fused_negative_extremum_ties=int(np.count_nonzero(df[nn] == df[hn['fused']])))
                        counts.update(dict(role_union_negative_records=len(proposed_n), role_union_positive_records=len(proposed_p), extra_negative_records=len(extra_n), extra_positive_records=len(extra_p), extra_negative_fused_hinge_active=len(supported_n), extra_negative_fused_nonpositive_margin=len(wrong_n), anchors_with_extra_hinge_active_negative=int(bool(supported_n)), anchors_with_extra_wrong_order_negative=int(bool(wrong_n)), role_subset_lower_hinge=int(subset < full-1e-12), role_subset_equal_hinge=int(abs(subset-full) <= 1e-12), role_subset_higher_hinge=int(subset > full+1e-12), full_fused_hinge_active=int(full > 0), contains_both_fused_extrema=int(row['contains_fused_negative'] and row['contains_fused_positive'])))
                        for role in ROLES:
                            counts[f'{role}_unique_negative'] += int(unique_role[role])
                    stream.write(json.dumps(row)+'\n')
                condition_rows.append(dict(condition=name,fold=condition['fold'],state=condition['state'],view=condition['view'],protocol=mode,counts=dict(counts)))
            print(json.dumps(dict(completed_conditions=len(condition_rows),condition=name)),flush=True)
    assert len(condition_rows) == 54
    assert sum(x['counts']['all_query_memberships'] for x in condition_rows) == 37152
    result=dict(status='COMPLETE_SOURCE_ROLE_RELATION_COVERAGE',contract_sha256=sha(args.contract),script_sha256=sha(Path(__file__)),model_forwards=0,optimizer_updates=0,heldout_image_reads=0,official_image_reads=0,source_root=str(source),conditions=condition_rows,inputs=files,elapsed_seconds=time.perf_counter()-started,completed_at=datetime.now().astimezone().isoformat(),limitations=['Sealed initial/control/style source models, not history-gradient Q1 endpoints.','Clean gallery and fixed query views, not training-mode queue replay.','Active embedding relations do not establish parameter gradients or heldout benefit.','No subset of the same candidate pool increases its fused hardest hinge; tie gradient conventions are not evaluated.'])
    result['query_rows']=dict(bytes=(args.output/'all_query_relations.jsonl').stat().st_size,sha256=sha(args.output/'all_query_relations.jsonl'))
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
