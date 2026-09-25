"""Fixed 20-epoch R2 or V27 role training on one complete official train split."""

import itertools
import json
from pathlib import Path
import time

from tools.train_msvr310_trifusion_oof import OUTPUT_WIDTHS

PLAIN_WIDTHS = {"baseline_only": 1536, "fused": 6144,
                "cnn": 3072, "transformer": 3072, "mamba": 3072}


def _setup(model, config):
    import torch
    from trifusion.signal_preserving_v8 import ExpertFormationV8Criterion

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config["OPTIMIZATION"]["NEW_MODULE_LR"],
                                  weight_decay=config["OPTIMIZATION"]["WEIGHT_DECAY"])
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
    criterion = ExpertFormationV8Criterion(triplet_margin=.3, label_smoothing=.1).cuda()
    return optimizer, scaler, criterion


def _v27_loss(parts, config):
    weights = config["LOSS"]
    common = weights["ID_FUSED"] * parts["id_fused"] + weights["TRIPLET_FUSED"] * parts["triplet_fused"]
    for expert in ("cnn", "transformer", "mamba"):
        common = common + weights["ID_BRANCH"] * parts[f"id_{expert}"]
        common = common + weights["TRIPLET_BRANCH"] * parts[f"triplet_{expert}"]
        common = common + weights["ID_RESIDUAL"] * parts[f"id_residual_{expert}"]
    residual = weights["TRIPLET_RESIDUAL"] * sum(
        parts[f"triplet_residual_{expert}"] for expert in ("cnn", "transformer", "mamba"))
    return common.float() + residual.float()


def train_v27(model, protocol, records, config, *, m0, directory, seed=42, style=True,
              plain_baseline=False):
    import torch
    import numpy as np
    from tools.official_three_dataset_data import loader_for
    from tools.run_signal_preserving_v5 import (
        _module_state_sha256, _set_seed, _training_batch, learning_rate_multiplier,
    )
    from tools.train_msvr310_trifusion_oof import frozen_state_sha, output_mapping
    from trifusion.source_style_v27 import make_style_plan

    _set_seed(seed)
    model.train()
    initial, frozen = _module_state_sha256(model), frozen_state_sha(model)
    optimizer, scaler, criterion = _setup(model, config)
    method = "V27" if style else "PLAIN_V8" if plain_baseline else "SIGNAL_V8"
    loader = loader_for(protocol, records, training=True, method=method, seed=seed)
    epochs = 1 if m0 else 20
    history, live, steps, overflow = [], set(), 0, 0
    with (directory / "training_steps.jsonl").open("x", encoding="utf-8") as log:
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            lr = config["OPTIMIZATION"]["NEW_MODULE_LR"] * (
                1 if m0 else learning_rate_multiplier(epoch, max_epochs=20, warmup_epochs=5))
            for group in optimizer.param_groups:
                group["lr"] = lr
            losses = []
            batches = itertools.islice(loader, 8) if m0 else loader
            for raw in batches:
                assert sorted(torch.unique(raw[1], return_counts=True)[1].tolist()) == [8] * 8
                batch, labels = _training_batch(raw)
                if style:
                    model.baseline.style_enabled = True
                    model.baseline.style_plan = make_style_plan(raw[2].numpy(), fold=0, step=steps)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    output = model(batch, return_aux=True)
                    output_mapping(output, widths=PLAIN_WIDTHS if plain_baseline else OUTPUT_WIDTHS)
                    components = criterion(output, labels)
                    loss = _v27_loss(components, config)
                scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad and parameter.grad is not None:
                        assert torch.isfinite(parameter.grad).all(), name
                        if bool(parameter.grad.abs().sum() > 0):
                            live.add(name)
                scaler.step(optimizer)
                scaler.update()
                overflow += int(scaler.get_scale() < scale)
                steps += 1
                losses.append(float(loss.detach()))
                log.write(json.dumps(dict(step=steps, epoch=epoch, loss=losses[-1],
                                          style_active=model.baseline.last_style_stats["style_active"] if style else False,
                                          style_plan=model.baseline.style_plan if style else None,
                                          amp_scale_after=scaler.get_scale())) + "\n")
            log.flush()
            row = dict(epoch=epoch, steps=len(losses), mean_loss=float(np.mean(losses)),
                       seconds=time.perf_counter() - started)
            history.append(row)
            print(json.dumps(dict(event=f"official_{method.lower()}_epoch",
                                  dataset=protocol["dataset"], **row)), flush=True)
    if style:
        model.baseline.style_plan = None
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert live == trainable and overflow == 0 and frozen == frozen_state_sha(model)
    return dict(method=method, epochs=epochs, optimizer_steps=steps, history=history,
                initial_state_sha256=initial, final_state_sha256=_module_state_sha256(model),
                frozen_state_unchanged=True, missing_nonzero_gradients=[], overflow_events=0)


def train_r2(model, protocol, records, config, *, m0, directory, seed=42, top1=False,
             balanced=True):
    import torch
    import torch.nn.functional as F
    import numpy as np
    from tools.official_three_dataset_data import loader_for
    from tools.msvr_cross_scene_smooth_ap import objectives, cross_environment_top1_from_distances
    from tools.msvr_freshness_probe import ViewFields
    from tools.msvr_instance_memory import InstanceMemory, expanded_triplet
    from tools.msvr_role_set_relations import fused_distances
    from tools.msvr_supported_gradient_balance import (
        SupportedBalance, role_indices, scalar_gradients, combine, support_counts,
    )
    from tools.probe_msvr_role_set_gradients import refresh_all
    from tools.probe_msvr_history_candidate_gradients import encode_graph
    from tools.run_signal_preserving_v5 import (
        _module_state_sha256, _set_seed, _training_batch, learning_rate_multiplier,
        weighted_training_loss,
    )
    from tools.train_msvr310_trifusion_oof import frozen_state_sha, output_mapping

    _set_seed(seed)
    model.train()
    initial, frozen = _module_state_sha256(model), frozen_state_sha(model)
    optimizer, scaler, criterion = _setup(model, config)
    selected = [(name, parameter) for name, parameter in model.named_parameters()
                if parameter.requires_grad and name.startswith("encoder.")]
    parameters = [parameter for _, parameter in selected]
    groups = role_indices([name for name, _ in selected])
    heads = [parameter for name, parameter in model.named_parameters()
             if parameter.requires_grad and not name.startswith("encoder.")]
    assert len(parameters) == 189 and len(heads) == 14
    controller = SupportedBalance()
    source = protocol["records"]["train"]
    memory = InstanceMemory(source, capacity=512, maximum_age=8)
    fields = ViewFields()
    lookup = {Path(row[0] if isinstance(row[0], str) else row[0][0]).name: index
              for index, row in enumerate(records)}
    assert len(lookup) == len(records)
    loader = loader_for(protocol, records, training=True, method="R2", seed=seed)
    epochs, warmup = (1, 2) if m0 else (20, 65)
    history, live, steps, overflow, supported_steps, historical_vjp_groups = [], set(), 0, 0, 0, 0
    with (directory / "training_steps.jsonl").open("x", encoding="utf-8") as log:
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            lr = config["OPTIMIZATION"]["NEW_MODULE_LR"] * (
                1 if m0 else learning_rate_multiplier(epoch, max_epochs=20, warmup_epochs=5))
            for group in optimizer.param_groups:
                group["lr"] = lr
            losses = []
            batches = itertools.islice(loader, 8) if m0 else loader
            for raw in batches:
                assert sorted(torch.unique(raw[1], return_counts=True)[1].tolist()) == [8] * 8
                indices = [lookup[name] for name in raw[-1]]
                identities = [source[index]["identity"] for index in indices]
                environments = [source[index]["scene"] for index in indices]
                batch, labels = _training_batch(raw)
                optimizer.zero_grad(set_to_none=True)
                _, metadata = memory.read(steps, indices, torch.empty((0, 7680), device="cuda"))
                refreshed, _ = refresh_all(model, fields, metadata, steps)
                fresh = refreshed["fused"]
                with torch.autocast("cuda", dtype=torch.float16):
                    output, item = fields.capture(model, batch, indices)
                    output_mapping(output)
                    components = criterion(output, labels)
                _, basic, unit, current_distances, history_distances, _ = expanded_triplet(
                    output.fused_embedding, identities, environments, fresh, metadata)
                assert torch.equal(basic, components["triplet_fused"])
                history_ids = [row["identity"] for row in metadata]
                history_environments = [row["scene"] for row in metadata]
                _, _, cross_ap, _, _, positive_counts = objectives(
                    current_distances, history_distances, identities, history_ids,
                    environments, history_environments)
                if top1:
                    top1_loss, top1_count = cross_environment_top1_from_distances(
                        current_distances, history_distances, identities, history_ids,
                        environments, history_environments)
                    assert int(top1_count) == int((positive_counts > 0).sum())
                    rank_objective = cross_ap + top1_loss
                else:
                    rank_objective = cross_ap
                active = steps >= warmup
                if active:
                    components["triplet_fused"] = rank_objective
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = weighted_training_loss(components, config)
                scale = scaler.get_scale()
                rank = scalar_gradients(components["triplet_fused"], parameters, scale)
                auxiliary_components = dict(components)
                auxiliary_components["triplet_fused"] = components["triplet_fused"] * 0
                with torch.autocast("cuda", dtype=torch.float16):
                    auxiliary_loss = weighted_training_loss(auxiliary_components, config)
                auxiliary = scalar_gradients(auxiliary_loss, parameters, scale)
                upstream = None
                if metadata:
                    leaf = fresh.detach().requires_grad_(True)
                    partial_current, partial_history = fused_distances(output.fused_embedding.detach(), leaf)
                    partial = objectives(partial_current, partial_history, identities, history_ids,
                                         environments, history_environments)[2]
                    if top1:
                        partial_top1, _ = cross_environment_top1_from_distances(
                            partial_current, partial_history, identities, history_ids,
                            environments, history_environments)
                        partial = partial + partial_top1
                    assert torch.equal(partial.detach(), rank_objective.detach())
                    upstream = torch.autograd.grad(partial, leaf)[0].detach()
                scaler.scale(loss).backward()
                current = [parameter.grad.detach().float().clone() for parameter in parameters]
                head_gradients = [parameter.grad.detach().clone() for parameter in heads]
                historical = [torch.zeros_like(parameter, dtype=torch.float32) for parameter in parameters]
                for key in sorted({row["stored_step"] for row in metadata}):
                    positions = [i for i, row in enumerate(metadata) if row["stored_step"] == key]
                    if not bool(upstream[positions].abs().sum() > 0):
                        continue
                    previous = fields.fields[key]
                    encoded = encode_graph(model, previous)
                    offsets = [previous["positions"][metadata[i]["record_index"]] for i in positions]
                    assert torch.equal(encoded[offsets].detach(), fresh[positions])
                    coefficients = torch.zeros_like(encoded)
                    coefficients[offsets] = upstream[positions]
                    values = torch.autograd.grad((encoded * coefficients).sum() * scale,
                                                 parameters, allow_unused=True)
                    for target, value in zip(historical, values, strict=True):
                        if value is not None:
                            target.add_(value.float())
                    historical_vjp_groups += 1
                support = support_counts(identities, environments, metadata, positive_counts.tolist())
                supported = active and support["eligible_anchors"] > 0
                if supported:
                    supported_steps += 1
                balance, _, _ = combine(parameters, current, rank, auxiliary, historical,
                                        scale, groups, controller, supported, balanced)
                assert all(torch.equal(parameter.grad, gradient)
                           for parameter, gradient in zip(heads, head_gradients, strict=True))
                scaler.unscale_(optimizer)
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad and parameter.grad is not None:
                        assert torch.isfinite(parameter.grad).all(), name
                        if bool(parameter.grad.abs().sum() > 0):
                            live.add(name)
                scaler.step(optimizer)
                scaler.update()
                overflow += int(scaler.get_scale() < scale)
                if active:
                    memory.update(steps, indices, unit.detach())
                    fields.fields[steps] = item
                steps += 1
                losses.append(float(loss.detach()))
                log.write(json.dumps(dict(step=steps, epoch=epoch, loss=losses[-1],
                                          active_fused_metric=("cross_environment_smooth_ap_plus_top1" if top1
                                                               else "cross_environment_smooth_ap") if active else "hard_triplet",
                                          top1_loss=float(top1_loss.detach()) if top1 else None,
                                          eligible_anchors=support["eligible_anchors"],
                                          memory_records=len(metadata),
                                          historical_vjp_groups=historical_vjp_groups,
                                          role_weights={role: [row["applied_rank_weight"], row["applied_auxiliary_weight"]]
                                                        for role, row in balance.items()},
                                          amp_scale_after=scaler.get_scale())) + "\n")
            log.flush()
            row = dict(epoch=epoch, steps=len(losses), mean_loss=float(np.mean(losses)),
                       seconds=time.perf_counter() - started)
            history.append(row)
            print(json.dumps(dict(event="official_r2_epoch", dataset=protocol["dataset"], **row)), flush=True)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert live == trainable and overflow == 0 and frozen == frozen_state_sha(model)
    assert supported_steps > 0 and historical_vjp_groups > 0
    return dict(method="R2_TOP1" if top1 else "R2" if balanced else "R2_UNIFORM",
                epochs=epochs, optimizer_steps=steps, history=history,
                initial_state_sha256=initial, final_state_sha256=_module_state_sha256(model),
                frozen_state_unchanged=True, missing_nonzero_gradients=[], overflow_events=0,
                supported_steps=supported_steps, historical_vjp_groups=historical_vjp_groups,
                gradient_balance_applied=balanced,
                gradient_balance_state=controller.states)
