"""Prepare the bounded native-evidence source option; no runtime execution."""
from pathlib import Path
import ast

repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
draft = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
mapping = {
    'ImageNativeEvidenceReader.py': 'modeling/trifusion/image_native_evidence.py',
    'independent_native_roles.py': 'modeling/trifusion/independent_native_roles.py',
    'evidence_author_heads.py': 'modeling/trifusion/evidence_author_heads.py',
    'run_independent_native_author_option.py': 'tools/run_independent_native_evidence.py',
}
for old, new in mapping.items():
    path = repo / new
    assert not path.exists()
    source = (draft / old).read_text(encoding='utf-8')
    if old == 'run_independent_native_author_option.py':
        source = source.replace("SCHEMA = 'trifusion-independent-native-author-option-v1'", "SCHEMA = 'trifusion-independent-native-evidence-v1'")
        source = source.replace("'architecture': SCHEMA, 'dataset': args.dataset, 'variant': args.variant,", "'architecture': SCHEMA, 'dataset': args.dataset, 'variant': args.variant, 'recipe': args.variant,")
        source = source.replace("original_loss_values(args, output, labels, cameras, loss_fn)", "original_loss_values(argparse.Namespace(**dict(vars(args), recipe='author')), output, labels, cameras, loss_fn)")
        source = source.replace("args.recipe = 'author'", "args.recipe = args.variant")
        source = source.replace("clean.control.GlobalTokenTriFusion = RawFeatureSemanticTriFusion\n    foundation.SCHEMA", "configure()\n    foundation.SCHEMA")
        source = source.replace("    foundation.SCHEMA = SCHEMA\n    foundation.condition = condition\n    foundation.build_core = build_core\n    foundation.train_loader = train_loader\n    foundation.optimization = optimization\n    foundation.loss_values = loss_values\n", "")
        source = source.replace("(foundation.evaluate if args.mode == 'evaluate' else foundation.train)(args, protocol)", "(foundation.evaluate if args.mode == 'evaluate' else train)(args, protocol)")
        source = source.replace("return optimizer, scheduler, loss_fn", "    # Observe the actual unscaled gradients and effective optimizer updates.\n    if args.mode == 'm0':\n        global M0_DIAGNOSTICS\n        M0_DIAGNOSTICS = M0Diagnostics(model, optimizer)\n        optimizer.register_step_post_hook(M0_DIAGNOSTICS.after_step)\n    return optimizer, scheduler, loss_fn")
        source = source.replace("        # Observe", "    # Observe")
        anchor = '\ndef main():\n'
        additions = '''
class M0Diagnostics:
    def __init__(self, model, optimizer):
        self.model = model
        self.detail = {name: p for name, p in model.named_parameters()
                       if '.detail_reader.' in name and p.requires_grad}
        self.initial = {name: p.detach().clone() for name, p in self.detail.items()}
        self.updates = []
        names = {id(p): name for name, p in model.named_parameters()}
        self.groups = [{'names': [names[id(p)] for p in group['params']],
                        'lr': group['lr'], 'weight_decay': group['weight_decay']}
                       for group in optimizer.param_groups]

    def after_step(self, _optimizer, _args, _kwargs):
        assert self.model.signal.training
        assert all(getattr(self.model.signal, neck).training for neck, _ in self.model.head_names)
        self.updates.append({name: {
            'unscaled_gradient_max_abs': float(p.grad.detach().abs().max()) if p.grad is not None else None,
            'parameter_delta_from_initial_max_abs': float((p.detach() - self.initial[name]).abs().max()),
        } for name, p in self.detail.items()})

    def result(self):
        assert len(self.updates) == 8
        if self.detail:
            assert len(self.detail) == 14 and sum(p.numel() for p in self.detail.values()) == 159296
            assert all(any(row[name]['unscaled_gradient_max_abs'] is not None
                           and row[name]['unscaled_gradient_max_abs'] > 0 for row in self.updates)
                       for name in self.detail)
            assert all(self.updates[-1][name]['parameter_delta_from_initial_max_abs'] > 0
                       for name in self.detail)
        tracked = {neck: int(getattr(self.model.signal, neck).num_batches_tracked)
                   for neck, _ in self.model.head_names}
        assert all(value == 8 for value in tracked.values())
        return {'effective_optimizer_updates': 8, 'optimizer_groups_at_construction': self.groups,
                'author_bn_batches_tracked': tracked, 'detail_parameters': list(self.detail),
                'detail_updates': self.updates,
                'boundary': 'Engineering gradient/update support only; no evidence of retrieval benefit.'}


M0_DIAGNOSTICS = None


def train(args, protocol):
    foundation.train(args, protocol)
    if args.mode == 'm0':
        result = M0_DIAGNOSTICS.result()
        path = args.output_dir / 'training.json'
        receipt = json.loads(path.read_text())
        receipt['production_m0_diagnostics'] = result
        path.write_text(json.dumps(receipt, indent=2) + '\\n')


def configure():
    clean.control.GlobalTokenTriFusion = RawFeatureSemanticTriFusion
    foundation.SCHEMA = SCHEMA
    foundation.condition = condition
    foundation.build_core = build_core
    foundation.train_loader = train_loader
    foundation.optimization = optimization
    foundation.loss_values = loss_values

'''
        assert anchor in source
        source = source.replace(anchor, '\n' + additions + anchor)
    ast.parse(source)
    path.write_bytes(source.encode())
print('Four candidate source files prepared; no imports, initializer, M0 or training.')
