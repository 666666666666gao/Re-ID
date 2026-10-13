"""Inspect real entry wiring without building models, data or optimizers."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_normalized_role_classification as entry
from trifusion.normalized_role_classification import NormalizedRoleClassification
from trifusion.role_input_detach import DetachedSemanticTriFusion


def location(value):
    return {'module': value.__module__, 'name': value.__qualname__,
            'source': str(Path(inspect.getsourcefile(value)).resolve().relative_to(ROOT))}


def check_bindings(stage):
    inner = entry.inner
    base = entry.previous.previous.base
    bindings = {
        'inner.build_core': (inner.build_core, entry.build_core),
        'foundation.build_core': (inner.foundation.build_core, entry.build_core),
        'inner.AuthorHeadEvidence': (inner.AuthorHeadEvidence, NormalizedRoleClassification),
        'inner.RawFeatureSemanticTriFusion': (inner.RawFeatureSemanticTriFusion, DetachedSemanticTriFusion),
        'clean.control.GlobalTokenTriFusion': (inner.clean.control.GlobalTokenTriFusion, DetachedSemanticTriFusion),
        'inner.loss_values': (inner.loss_values, entry.previous.loss_values),
        'foundation.loss_values': (inner.foundation.loss_values, entry.previous.loss_values),
        'inner.condition': (inner.condition, entry.condition),
        'foundation.condition': (inner.foundation.condition, entry.condition),
        'foundation.train_loader': (inner.foundation.train_loader, inner.train_loader),
        'foundation.optimization': (inner.foundation.optimization, base.entry.optimization),
        'foundation.train': (inner.foundation.train, base.entry.train),
    }
    assert all(actual is expected for actual, expected in bindings.values())
    assert inner.SCHEMA == inner.foundation.SCHEMA == entry.SCHEMA
    return {'stage': stage, 'schema': entry.SCHEMA,
            'bindings': {name: location(actual) for name, (actual, _expected) in bindings.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    base = entry.previous.previous.base
    original_builders = [entry.build_core, entry.original_build_core,
                         entry.previous.original_build_core, entry.previous.previous.original_build_core,
                         base.original_build_core, base.entry.original_build_core]
    assert len(set(original_builders)) == 6
    assert [value.__module__ for value in original_builders] == [
        'tools.run_normalized_role_classification', 'tools.run_global_task_role',
        'tools.run_role_input_detach', 'tools.run_native_research',
        'tools.run_native_partitioned', 'tools.run_independent_native_evidence']
    entry.configure()
    first = check_bindings('after_outer_configure')
    # The actual inner.main invokes this same configure once before dispatch.
    entry.inner.configure()
    second = check_bindings('after_main_inner_configure')
    assert first['bindings'] == second['bindings']
    modules = [inspect.getmodule(value) for value in original_builders]
    paths = {Path(module.__file__).resolve() for module in modules}
    paths.update(Path(inspect.getsourcefile(value)).resolve() for value in (
        NormalizedRoleClassification, DetachedSemanticTriFusion, entry.inner.AuthorHeadEvidence,
        entry.inner.foundation.loss_values, entry.inner.foundation.evaluate))
    paths.add(Path(__file__).resolve())
    result = {'schema': 'normalized-role-entry-source-bindings-v1', 'status': 'PASS_ENTRY_BINDINGS_ONLY',
              'build_delegation_chain': [location(value) for value in original_builders],
              'stages': [first, second],
              'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sorted(paths)},
              'counts': {'outer_configure_calls': 1, 'inner_main_configure_calls': 1,
                         'build_core_calls': 0, 'model_constructions': 0, 'neural_forward_calls': 0,
                         'data_records': 0, 'loss_evaluations': 0, 'autograd_calls': 0,
                         'optimizer_constructions': 0, 'optimizer_updates': 0, 'prepare_runs': 0,
                         'M0_runs': 0, 'training_runs': 0, 'evaluation_runs': 0, 'weights_written': 0},
              'limits': ['Real Python entry import/configuration wiring only; main CLI and build_core not run.',
                         'No production initialization, real M0, data, optimizer coverage or retrieval claim.',
                         'Synthetic CPU head result936 remains separate and is not rerun.']}
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'builders': len(original_builders),
                      'stages': len(result['stages']), 'model_constructions': 0}))


if __name__ == '__main__':
    main()
