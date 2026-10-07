"""Fixed H2a pilot; unchanged raw author tasks plus same-image dense geometry."""
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_global_task_role as previous
from trifusion.prepool_dense_correspondence import (
    DenseTrainingBatch, PrepoolDenseHeads, attach_visibility,
)

SCHEMA = 'trifusion-prepool-dense-correspondence-v1'
inner = previous.previous.base.entry.entry
original_build_core = previous.build_core
original_condition = previous.condition
original_loss_values = previous.loss_values
original_training_batch = inner.runner._training_batch
original_train_loader = inner.train_loader
original_official_metrics = inner.runner.official_metrics
CURRENT_MODEL = None


def build_core(args, protocol):
    global CURRENT_MODEL
    assert args.variant == 'semantic'
    model, cfg, binding = original_build_core(args, protocol)
    assert cfg.INPUT.PADDING == 10
    assert list(cfg.INPUT.PIXEL_MEAN) == list(cfg.INPUT.PIXEL_STD) == [0.5] * 3
    assert isinstance(model, PrepoolDenseHeads)
    model.verify_teacher_state = args.mode == 'm0'
    model.collect_global = args.mode == 'evaluate'
    CURRENT_MODEL = model
    binding.update(entry_sha256=inner.runner.sha256(Path(__file__)),
        auxiliary_objective='known_same_image_same_modality_geometry_prepool',
        auxiliary_weight=1.0, cosine_logit_scale=128 ** 0.5,
        view_policy='per_modality_batch_shared_flip0.5_then_pad10_minus1_crop_uniform_integer_dy_dx_-10_10',
        main_support_policy='original_erasing_equality_and_conservative10px_border',
        added_model_parameters=0,
        scope='H2a bounded same-modal geometric supervision pilot. Natural existing pre-key gradients; detached CLIP/shared path, original raw author tasks/1536 inference. Not reliable cross-spectral parts or novel formula.')
    return model, cfg, binding


def training_batch(raw):
    # Vehicle evaluation really calls this parser with the original five fields.
    if isinstance(raw, DenseTrainingBatch):
        batch, labels = original_training_batch(raw[:5])
        batch['dense_visibility'] = raw.visible
        return batch, labels
    return original_training_batch(raw)


def train_loader(args, protocol, cfg):
    return attach_visibility(original_train_loader(args, protocol, cfg))


def condition(args):
    return {**original_condition(args), 'auxiliary_tasks': 'prepool_same_modality_dense_geometry',
            'auxiliary_weight': 1.0, 'cosine_logit_scale': 128 ** 0.5,
            'second_view': 'flip0.5_pad10_minus1_crop_uniform_integer_-10_10_per_modality_batch',
            'inference_second_view': False, 'added_model_parameters': 0}


def loss_values(args, output, labels, cameras, loss_fn):
    original, values = original_loss_values(args, output, labels, cameras, loss_fn)
    auxiliary = output['dense_loss']
    values.update(output['dense_stats'], dense_correspondence_loss=float(auxiliary.detach()))
    if args.mode == 'm0':
        parameters = [(name, parameter) for name, parameter in CURRENT_MODEL.named_parameters()
                      if name.startswith('evidence_model.roles.key_projections.')]
        assert len(parameters) == 3
        with torch.autocast('cuda', enabled=False):
            gradients = torch.autograd.grad(auxiliary, [output['shared_global']] +
                [parameter for _, parameter in parameters], retain_graph=True, allow_unused=True)
        assert gradients[0] is None
        assert all(gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients[1:])
        values['dense_isolated_shared_global_gradient_absent'] = True
        values['dense_isolated_key_gradient_norms'] = {
            name: float(gradient.norm()) for (name, _), gradient in zip(parameters, gradients[1:])}
    return original + auxiliary, values


def official_metrics(model, protocol, source, *, save_distances=None):
    fused = original_official_metrics(model, protocol, source, save_distances=save_distances)
    if model.collect_global:
        from tools.train_msvr310_signal_oof import scene_scores
        from tools.train_rgbnt100_signal_oof import camera_scores
        from utils import metrics as author_metrics

        assert save_distances is not None and Path(author_metrics.__file__).resolve() == source / 'utils/metrics.py'
        global_features = torch.cat(model.global_rows)
        model.global_rows = []
        query_count = protocol['counts']['query']
        assert global_features.shape == (query_count + protocol['counts']['gallery'], 1536)
        distances = inner.runner.distance_matrix(global_features[:query_count], global_features[query_count:])
        qrows, grows = protocol['records']['query'], protocol['records']['gallery']
        qids, gids = [np.asarray([row['identity'] for row in rows]) for rows in (qrows, grows)]
        qcameras, gcameras = [np.asarray([row['camera'] for row in rows]) for rows in (qrows, grows)]
        qscenes, gscenes = [np.asarray([row['scene'] for row in rows]) for rows in (qrows, grows)]
        if protocol['dataset'] == 'MSVR310':
            scores = scene_scores(distances.numpy(), qids, gids, qscenes, gscenes)
            cmc, mean_ap = author_metrics.eval_func_msrv(distances.numpy(), qids, gids, qcameras, gcameras, qscenes, gscenes)
        else:
            scores = camera_scores(distances.numpy(), qids, gids, qcameras, gcameras)
            cmc, mean_ap = author_metrics.eval_func(distances.numpy(), qids, gids, qcameras, gcameras)
        metrics = {'mAP': float(mean_ap) * 100, **{f'Rank-{rank}': float(cmc[rank - 1]) * 100 for rank in (1, 5, 10)}}
        assert all(abs(metrics[name] - scores['metrics'][name]) < 1e-5 for name in metrics)
        path = save_distances.parent / 'own_global_distances.pt'
        torch.save({'fused': distances, 'query_ids': qids, 'gallery_ids': gids,
                    'query_cameras': qcameras, 'gallery_cameras': gcameras,
                    'query_scenes': qscenes, 'gallery_scenes': gscenes}, path)
        (path.parent / 'own_global_metrics.json').write_text(json.dumps({
            'schema': SCHEMA, 'status': 'COMPLETE', 'dataset': protocol['dataset'],
            'metrics': metrics, 'full_scores': scores, 'distance_sha256': inner.runner.sha256(path),
            'same_original_fused_forward': True, 'new_global_NN_forwards': 0,
            'epoch_selection': 'same_fused_mAP_best', 'completed_at': datetime.now().astimezone().isoformat(),
        }, indent=2) + '\n')
    return fused


def configure():
    previous.configure()
    inner.AuthorHeadEvidence = PrepoolDenseHeads
    inner.runner._training_batch = training_batch
    inner.loss_values = loss_values
    inner.train_loader = train_loader
    inner.runner.official_metrics = official_metrics
    base = previous.previous.base
    base.SCHEMA, base.build_core = SCHEMA, build_core
    base.configure()
    inner.condition = inner.foundation.condition = condition


if __name__ == '__main__':
    configure()
    inner.main()
