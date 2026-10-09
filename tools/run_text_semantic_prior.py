"""Matched pretrained/random frozen text-context packages; unchanged raw tasks."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_global_task_role as previous
from trifusion.global_task_role_heads import GlobalTaskRoleHeads
from trifusion.partitioned_evidence_clip import placement_summary
from trifusion import slot_competition_fp32_roles as fp32_reader
from trifusion.text_semantic_prior import (
    FrozenTextPackage, TextContextConditioner, TextContextTriFusion,
    PERSON_TOKENS, VEHICLE_TOKENS, text_package_state,
)

SCHEMA = 'trifusion-text-semantic-prior-v1'
inner = previous.previous.base.entry.entry
original_build_core = previous.build_core
original_optimization = inner.optimization
original_official_metrics = inner.runner.official_metrics
TEXT_DIAGNOSTICS = None
TEXT_OBSERVATIONS = None
original_allocation = fp32_reader.allocation_weights


class TextContextHeads(GlobalTaskRoleHeads):
    def forward(self, batch, *, return_aux=False):
        if return_aux:
            return super().forward(batch, return_aux=True)
        output = self.evidence_model.forward_features(batch)
        if self.collect_global:
            self.global_rows.append(torch.nn.functional.normalize(
                output['shared_global'].detach().float(), dim=1).cpu())
        return output['fused']


def build_core(args, protocol):
    assert args.variant == 'semantic'
    model, cfg, binding = original_build_core(args, protocol)
    assert isinstance(model.evidence_model, TextContextTriFusion)
    common_sha = inner.runner._module_state_sha256(model)
    rng_before = torch.get_rng_state().clone()
    cuda_before = [value.clone() for value in torch.cuda.get_rng_state_all()]
    from modeling.clip import model as clip_module
    assert Path(clip_module.__file__).resolve() == args.signal_source / 'modeling/clip/model.py'
    public = torch.jit.load(str(args.clip_weight), map_location='cpu').state_dict()
    package = text_package_state(public, args.text_package)
    tokens = PERSON_TOKENS if args.dataset == 'RGBNT201' else VEHICLE_TOKENS
    frozen_text = FrozenTextPackage(clip_module, package, tokens)
    conditioner = TextContextConditioner(frozen_text).to('cuda:0')
    model.evidence_model.text_conditioner = conditioner
    assert torch.equal(rng_before, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all()))
    assert not any(p.requires_grad for p in frozen_text.parameters())
    assert len(frozen_text.state_dict()) == 149
    model.collect_global = args.mode == 'evaluate'
    model.global_rows = []
    binding.update(architecture=SCHEMA, entry_sha256=inner.runner.sha256(Path(__file__)),
        common_initial_model_state_sha256=common_sha,
        initial_model_state_sha256=inner.runner._module_state_sha256(model),
        text_package=args.text_package, added_trainable_parameters=592000,
        added_trainable_tensors=5, frozen_text_state_tensors=149,
        frozen_text_state_elements=38137344,
        text_state_sha256=inner.runner._module_state_sha256(frozen_text),
        conditioner_initial_sha256=inner.runner._module_state_sha256(conditioner),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        trainable_parameter_tensors=sum(p.requires_grad for p in model.parameters()),
        parameter_placement=placement_summary(model),
        scope='Fixed frozen pretrained/random text package changes detached semantic context only; same raw tasks, original visual patch values and1536 inference. No new identity information or originality claim.')
    return model, cfg, binding


def condition(args):
    return {**previous.condition(args), 'text_package': args.text_package,
            'text_template': 'A photo of a X X X X person/vehicle.',
            'pseudo_words': 4, 'text_tokens': 12, 'text_dtype': 'float32',
            'added_trainable_parameters': 592000, 'auxiliary_tasks': 'none'}


class TextM0Diagnostics:
    def __init__(self, model):
        self.module = model.evidence_model.text_conditioner
        self.parameters = {name: p for name, p in self.module.named_parameters() if p.requires_grad}
        assert len(self.parameters) == 5 and sum(p.numel() for p in self.parameters.values()) == 592000
        self.initial = {name: p.detach().clone() for name, p in self.parameters.items()}
        self.context = model.evidence_model.roles.context_queries.weight
        self.context_initial = self.context.detach().clone()
        assert not bool(self.context_initial.count_nonzero())
        self.steps = []

    def after_step(self, _optimizer, _args, _kwargs):
        assert not self.module.text.training
        assert all(p.grad is None for p in self.module.text.parameters())
        self.steps.append(dict(
            context_gradient_max_abs=float(self.context.grad.detach().abs().max()),
            context_delta_max_abs=float((self.context.detach() - self.context_initial).abs().max()),
            parameters={name: dict(
                gradient_max_abs=float(p.grad.detach().abs().max()),
                cumulative_delta_max_abs=float((p.detach() - self.initial[name]).abs().max()))
                for name, p in self.parameters.items()}))

    def result(self):
        assert len(self.steps) == 8
        assert self.steps[0]['context_gradient_max_abs'] > 0 and self.steps[0]['context_delta_max_abs'] > 0
        assert all(value['gradient_max_abs'] == 0 for value in self.steps[0]['parameters'].values())
        assert all(any(row['parameters'][name]['gradient_max_abs'] > 0 for row in self.steps[1:])
                   for name in self.parameters)
        assert all(row['cumulative_delta_max_abs'] > 0 for row in self.steps[-1]['parameters'].values())
        return dict(status='TEXT_INPUT_AUTOGRAD_ACTIVITY_ACCEPTED', updates=8,
            first_step_zero_psi_W_gradient_expected=True, steps=self.steps,
            boundary='Actual task gradients and cumulative updates, not weight-decay-only activity or retrieval benefit.')


class TextTrainingObservations:
    """Observe actual forwards/optimizer steps without changing their outputs."""
    def __init__(self, model, output_dir):
        self.model = model
        self.module = model.evidence_model.text_conditioner
        self.parameters = {name: p for name, p in self.module.named_parameters() if p.requires_grad}
        self.initial = {name: p.detach().clone() for name, p in self.parameters.items()}
        self.frozen_before = inner.runner._module_state_sha256(self.module.text)
        self.updates = 0
        self.metrics, self.projections, self.reads = {}, {}, []
        self.log = (output_dir / 'text_updates.jsonl').open('x')
        self.module.psi.register_forward_hook(self.capture_pseudo_words)
        self.module.text.register_forward_hook(self.capture_text)
        self.module.output.register_forward_hook(self.capture_output)
        self.module.register_forward_hook(self.capture_context)
        roles = model.evidence_model.roles
        for role in range(3):
            roles.query_projections[role].register_forward_hook(
                lambda _m, _a, output, role=role: self.capture_projection(role, 'query', output))
            roles.key_projections[role].register_forward_hook(
                lambda _m, _a, output, role=role: self.capture_projection(role, 'key', output))

    def capture_pseudo_words(self, _module, _args, output):
        if self.model.training:
            self.metrics['pseudo_word_delta_norm_mean'] = float(output.detach().float().reshape(-1, 4, 512).norm(dim=-1).mean())

    def capture_text(self, _module, _args, output):
        if self.model.training:
            value = output.detach().float()
            self.metrics['text_output_norm_mean'] = float(value.norm(dim=-1).mean())
            self.text_features = torch.nn.functional.normalize(value, dim=-1).reshape(-1, 3, 512)

    def capture_output(self, _module, args, output):
        if self.model.training:
            self.context_features = args[0].detach().float()
            self.metrics['text_context_update_norm_mean'] = float(output.detach().float().norm(dim=-1).mean())

    def capture_context(self, _module, args, output):
        if self.model.training:
            value, original = output.detach().float(), args[1].detach().float()
            self.metrics['augmented_context_norm_mean'] = float(value.norm(dim=-1).mean())
            self.metrics['context_change_norm_mean'] = float((value - original).norm(dim=-1).mean())
            self.metrics['context_original_cosine_mean'] = float(torch.nn.functional.cosine_similarity(value, original).mean())

    def capture_projection(self, role, name, output):
        if self.model.training:
            self.projections[role, name] = float(output.detach().float().norm(dim=-1).mean())

    def capture_read(self, scores, weights):
        role = len(self.reads)
        assert role < 3
        self.reads.append(dict(role=role,
            query_norm_mean=self.projections[role, 'query'], key_norm_mean=self.projections[role, 'key'],
            logit_std_mean=float(scores.detach().float().std(dim=-1, unbiased=False).mean()),
            read_entropy_mean=float(torch.special.entr(weights.detach().float()).sum(dim=-1).mean())))

    def components(self, labels):
        assert len(self.reads) == 3
        modal = self.text_features
        self.metrics['text_modal_pair_cosine_means'] = [float((modal[:, a] * modal[:, b]).sum(-1).mean())
                                                       for a, b in ((0, 1), (0, 2), (1, 2))]
        similarities = self.context_features @ self.context_features.T
        same = labels[:, None] == labels[None, :]
        diagonal = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        self.metrics['text_same_identity_cosine_mean'] = float(similarities[same & ~diagonal].mean())
        self.metrics['text_different_identity_cosine_mean'] = float(similarities[~same].mean())
        result = dict(text_context_diagnostics=dict(self.metrics), role_attention_diagnostics=self.reads)
        self.metrics, self.reads = {}, []
        return result

    def after_step(self, _optimizer, _args, _kwargs):
        self.updates += 1
        self.last_parameters = {name: dict(
            unscaled_gradient_max_abs=float(p.grad.detach().abs().max()),
            cumulative_delta_max_abs=float((p.detach() - self.initial[name]).abs().max()))
            for name, p in self.parameters.items()}
        self.log.write(json.dumps(dict(optimizer_update=self.updates, parameters=self.last_parameters)) + '\n')
        self.log.flush()

    def result(self, expected_updates):
        self.log.close()
        frozen_after = inner.runner._module_state_sha256(self.module.text)
        assert self.updates == expected_updates and frozen_after == self.frozen_before
        return dict(status='ACTUAL_TEXT_TRAINING_OBSERVATIONS_COMPLETE', effective_optimizer_updates=self.updates,
            frozen_text_initial_sha256=self.frozen_before, frozen_text_final_sha256=frozen_after,
            full_frozen_text_state_unchanged=True, frozen_state_tensors=149,
            last_parameter_updates=self.last_parameters,
            boundary='Detached statistics of actual training forwards and optimizer updates; full149 text states include both buffers. Observation synchronization is included in measured runtime. Identity cosine diagnostics are training-batch descriptive, not an extra objective.')


def allocation_observed(scores, normalization):
    weights = original_allocation(scores, normalization)
    if TEXT_OBSERVATIONS is not None and TEXT_OBSERVATIONS.model.training:
        TEXT_OBSERVATIONS.capture_read(scores, weights)
    return weights


def optimization(args, model, cfg):
    global TEXT_DIAGNOSTICS, TEXT_OBSERVATIONS
    optimizer, scheduler, loss_fn = original_optimization(args, model, cfg)
    TEXT_OBSERVATIONS = TextTrainingObservations(model, args.output_dir)
    optimizer.register_step_post_hook(TEXT_OBSERVATIONS.after_step)
    if args.mode == 'm0':
        TEXT_DIAGNOSTICS = TextM0Diagnostics(model)
        optimizer.register_step_post_hook(TEXT_DIAGNOSTICS.after_step)
    return optimizer, scheduler, loss_fn


def loss_values(args, output, labels, cameras, loss_fn):
    loss, values = previous.loss_values(args, output, labels, cameras, loss_fn)
    values.update(TEXT_OBSERVATIONS.components(labels))
    if args.mode == 'm0':
        role_loss, _ = previous.previous.loss_values(args, output, labels, cameras, loss_fn)
        parameters = list(TEXT_DIAGNOSTICS.parameters.values())
        # The registered fresh8 M0 uses init_scale256 and growth_interval2000,
        # with no skipped step/scale decrease. Match its actual AMP backward;
        # an unscaled VJP can underflow through the existing AMP path.
        m0_gradient_scale = 256.0
        gradients = torch.autograd.grad(role_loss * m0_gradient_scale, [output['shared_global']] + parameters,
                                        retain_graph=True, allow_unused=True)
        assert gradients[0] is None
        assert all(g is not None and bool(torch.isfinite(g).all()) for g in gradients[1:])
        values['text_role_shared_global_gradient_absent'] = True
        values['text_isolated_vjp_scale'] = m0_gradient_scale
        values['text_isolated_task_gradient_max_abs'] = {
            name: float((g.detach().float() / m0_gradient_scale).abs().max())
            for name, g in zip(TEXT_DIAGNOSTICS.parameters, gradients[1:])}
    return loss, values


def train(args, protocol):
    inner.train(args, protocol)
    path = args.output_dir / 'training.json'
    receipt = json.loads(path.read_text())
    receipt['text_training_diagnostics'] = TEXT_OBSERVATIONS.result(sum(row['steps'] for row in receipt['history']))
    if args.mode == 'm0':
        receipt['text_m0_diagnostics'] = TEXT_DIAGNOSTICS.result()
    path.write_text(json.dumps(receipt, indent=2) + '\n')


def official_metrics(model, protocol, source, *, save_distances=None):
    fused = original_official_metrics(model, protocol, source, save_distances=save_distances)
    if model.collect_global:
        from tools.train_msvr310_signal_oof import scene_scores
        from tools.train_rgbnt100_signal_oof import camera_scores
        from utils import metrics as author_metrics
        assert save_distances is not None and Path(author_metrics.__file__).resolve() == source / 'utils/metrics.py'
        features = torch.cat(model.global_rows)
        model.global_rows = []
        query_count = protocol['counts']['query']
        assert features.shape == (query_count + protocol['counts']['gallery'], 1536)
        distances = inner.runner.distance_matrix(features[:query_count], features[query_count:])
        qrows, grows = protocol['records']['query'], protocol['records']['gallery']
        qids, gids = [np.asarray([row['identity'] for row in rows]) for rows in (qrows, grows)]
        qc, gc = [np.asarray([row['camera'] for row in rows]) for rows in (qrows, grows)]
        qs, gs = [np.asarray([row['scene'] for row in rows]) for rows in (qrows, grows)]
        if protocol['dataset'] == 'MSVR310':
            scores = scene_scores(distances.numpy(), qids, gids, qs, gs)
            cmc, mean_ap = author_metrics.eval_func_msrv(distances.numpy(), qids, gids, qc, gc, qs, gs)
        else:
            scores = camera_scores(distances.numpy(), qids, gids, qc, gc)
            cmc, mean_ap = author_metrics.eval_func(distances.numpy(), qids, gids, qc, gc)
        metrics = {'mAP': float(mean_ap) * 100, **{f'Rank-{rank}': float(cmc[rank - 1]) * 100 for rank in (1, 5, 10)}}
        assert all(abs(metrics[name] - scores['metrics'][name]) < 1e-5 for name in metrics)
        path = save_distances.parent / 'own_global_distances.pt'
        torch.save(dict(fused=distances, query_ids=qids, gallery_ids=gids,
                        query_cameras=qc, gallery_cameras=gc, query_scenes=qs, gallery_scenes=gs), path)
        (path.parent / 'own_global_metrics.json').write_text(json.dumps(dict(
            schema=SCHEMA, status='COMPLETE', dataset=protocol['dataset'], metrics=metrics,
            full_scores=scores, distance_sha256=inner.runner.sha256(path),
            same_original_fused_forward=True, new_global_NN_forwards=0,
            epoch_selection='same_fused_mAP_best', completed_at=datetime.now().astimezone().isoformat()), indent=2) + '\n')
    return fused


def configure():
    previous.configure()
    fp32_reader.allocation_weights = allocation_observed
    inner.RawFeatureSemanticTriFusion = TextContextTriFusion
    inner.clean.control.GlobalTokenTriFusion = TextContextTriFusion
    inner.AuthorHeadEvidence = TextContextHeads
    inner.foundation.SCHEMA = SCHEMA
    inner.foundation.build_core = inner.build_core = build_core
    inner.foundation.condition = inner.condition = condition
    inner.foundation.optimization = optimization
    inner.foundation.loss_values = inner.loss_values = loss_values
    inner.runner.official_metrics = official_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('RGBNT201', 'MSVR310', 'RGBNT100'), required=True)
    parser.add_argument('--text-package', choices=('pretrained', 'random'), required=True)
    parser.add_argument('--mode', choices=('prepare', 'm0', 'train', 'evaluate'), required=True)
    for name in ('protocol', 'signal-source', 'clip-weight', 'initialization', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    for name in ('protocol', 'signal_source', 'clip_weight', 'initialization', 'output_dir'):
        setattr(args, name, getattr(args, name).resolve())
    args.seed, args.epochs, args.variant, args.recipe = 42, 50, 'semantic', 'semantic'
    args.baseline_sha256 = inner.runner.sha256(args.clip_weight)
    configure()
    protocol = inner.runner.read_protocol(args.protocol, args.dataset)
    if args.mode == 'prepare':
        assert not args.initialization.exists()
        _model, _cfg, binding = build_core(args, protocol)
        args.initialization.parent.mkdir(parents=True, exist_ok=True)
        args.initialization.write_text(json.dumps(dict(schema=SCHEMA, status='INITIALIZATION_VERIFIED',
            binding=binding, prepared_at=datetime.now().astimezone().isoformat()), indent=2) + '\n')
    else:
        (inner.foundation.evaluate if args.mode == 'evaluate' else train)(args, protocol)


if __name__ == '__main__':
    main()
