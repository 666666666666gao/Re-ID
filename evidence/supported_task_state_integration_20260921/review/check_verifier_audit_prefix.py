"""Bounded stdlib regression of existing audit checks; no tensor/model execution."""
import ast
import copy
import json
import math
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
REPO = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
payload_tree = ast.parse((HERE/'serialization_payload_received.py').read_text(encoding='utf-8'))
modules = next(ast.literal_eval(n.value) for n in payload_tree.body
               if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'modules' for t in n.targets))
old_source = modules['verify_msvr_task_state_records']
new_source = (REPO/'tools/verify_msvr_task_state_records.py').read_text(encoding='utf-8')
(HERE/'verify_msvr_task_state_records_before.py').write_text(old_source, encoding='utf-8')
(HERE/'verify_msvr_task_state_records_after.py').write_text(new_source, encoding='utf-8')
assert "wanted=i<warmup or audit['support']['eligible_anchors']>0" in old_source

# The post-fix verifier reuses this existing stdlib scalar-pair checker.
tools_module = types.ModuleType('tools')
tools_module.__path__ = []
sys.modules['tools'] = tools_module
pair_module = types.ModuleType('tools.verify_msvr_supported_gradient_balance_stats')
sys.modules[pair_module.__name__] = pair_module
exec(compile((REPO/'tools/verify_msvr_supported_gradient_balance_stats.py').read_text(),
             'existing_scalar_pair_checker', 'exec'), pair_module.__dict__)


def audit_prefix(source):
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'verify_states')
    stop = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'proof' for t in n.targets))
    fn.body = fn.body[:stop] + [ast.Return(value=ast.Constant(value=True))]
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = {'math': math}
    exec(compile(module, 'unmodified_audit_prefix', 'exec'), namespace)
    return namespace['verify_states']


ROLES = ('cnn', 'transformer', 'mamba')
CHECKS = ('current_rank', 'historical_rank', 'full_rank', 'auxiliary', 'assembled')
PAIR = dict(first_norm=1.0, second_norm=1.0, difference_norm=0.0, cosine=1.0)
REFERENCE = dict(PAIR, relative_l2_error=0.0, passed=True)


def fixture():
    rows = []
    for i in range(8):
        has_reference = i == 3
        tasks = {role: {key: dict(step=i+1, first_moment_norm=1.0, second_moment_norm=1.0)
                        for key in ('rank', 'auxiliary')} for role in ROLES}
        rows.append(dict(identities=[0, 0, 1, 1], scenes=[0, 1, 0, 1], memory=[],
                         relation_objective={'cross_scene_positive_counts': [1, 1, 1, 1]},
                         support=dict(eligible_anchors=4, eligible_identities=2, identity_directed_scene_relations=4),
                         rank_observed=True, direct_ra_assembly=True,
                         classification_head_gradients_unchanged=True,
                         current_rank_backward_calls=1, current_auxiliary_backward_calls=1,
                         task_states=tasks,
                         direct_single_group_check={'fixture_marker': True} if has_reference else {},
                         direct_component_backward_calls=4 if has_reference else 0,
                         actual_parameter_updates={role: copy.deepcopy(PAIR) for role in ROLES},
                         rank_auxiliary_reference_checks={role: {key: copy.deepcopy(REFERENCE) for key in CHECKS}
                                                          for role in ROLES} if has_reference else {}))
    tr = dict(mode='capacity', current_rank_backward_calls=8, current_auxiliary_backward_calls=8,
              direct_component_backward_calls=4)
    return rows, tr


def run_case(name, mutation):
    rows, tr = fixture()
    mutation(rows, tr)
    result = {'name': name}
    for version, source in [('before', old_source), ('after', new_source)]:
        try:
            audit_prefix(source)(rows, tr, 'split', 2)
            accepted = True
        except AssertionError:
            accepted = False
        result[version+'_audit_prefix_accepted'] = accepted
    return result


def wrong_support(rows, tr):
    rows[3]['support'] = dict(eligible_anchors=0, eligible_identities=0, identity_directed_scene_relations=0)
    rows[3]['rank_observed'] = False
    for row in rows[3:]:
        for tasks in row['task_states'].values():
            tasks['rank']['step'] -= 1


def missing_references(rows, tr):
    rows[3]['rank_auxiliary_reference_checks'] = {}


def missing_updates(rows, tr):
    rows[3]['actual_parameter_updates'] = {}


def nonfinite_update(rows, tr):
    rows[3]['actual_parameter_updates']['cnn']['difference_norm'] = float('nan')


def wrong_backward_total(rows, tr):
    tr['current_rank_backward_calls'] = 0


results = [run_case('valid_prefix', lambda rows, tr: None),
           run_case('wrong_support_consistent_wrong_clock', wrong_support),
           run_case('missing_direct_component_references', missing_references),
           run_case('missing_role_update_witnesses', missing_updates),
           run_case('nonfinite_role_update_witness', nonfinite_update),
           run_case('wrong_total_backward_count', wrong_backward_total)]
assert results[0]['before_audit_prefix_accepted'] and results[0]['after_audit_prefix_accepted']
assert all(row['before_audit_prefix_accepted'] and not row['after_audit_prefix_accepted'] for row in results[1:])
report = dict(status='PASS_NARROW_REGRESSION', results=results, model_forwards=0, optimizer_updates=0,
              scope='Executes the verbatim verifier audit-validation prefix before disk loading; synthetic scalar fixtures. '
                    'Not a full verifier, tensor serialization, real-source M0, or Q1 run.')
(HERE/'verifier_audit_prefix_regression.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
