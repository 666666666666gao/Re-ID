"""Source-only V8 parity, eight-loss derivative, and checkpoint checks."""

import argparse
from argparse import Namespace
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from tools.queue_official_extra_seed import WEIGHTS
    from tools.run_official_three_dataset_roles import (
        initialize, read_protocol, save_role_checkpoint, checkpoint_names,
    )
    from tools.official_three_dataset_data import records_for, loader_for
    from tools.official_three_dataset_model import sha256
    from tools.run_signal_preserving_v5 import _training_batch, _module_state_sha256
    from tools.train_msvr310_trifusion_oof import output_mapping
    from tools.train_official_three_dataset_roles import _v27_loss

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(WEIGHTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    assert not cli.output.exists()
    cli.output.mkdir(parents=True)
    weight, digest = WEIGHTS[cli.dataset]
    args = Namespace(dataset=cli.dataset, method="SIGNAL_V8", seed=42,
                     signal_source=ROOT / "comparators/Signal-cd1b0a6",
                     clip_weight=ROOT / "pertrained-model/ViT-B-16.pt",
                     signal_checkpoint=ROOT / "pertrained-model" / weight,
                     signal_sha256=digest)
    protocol_path = ROOT / "logs/official_three_dataset_protocols_20260923" / f"{cli.dataset}.json"
    protocol = read_protocol(protocol_path, cli.dataset)
    model, _, original_config, _ = initialize(args, protocol)
    original_sha = _module_state_sha256(model)
    original_trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    loader = loader_for(protocol, records_for(protocol, "train"), training=True,
                        method=args.method, seed=42)
    raw = next(iter(loader))
    batch, labels = _training_batch(raw)
    del loader, raw

    def features(current):
        from tools.msvr310_exact_signal_inference import exact_signal_forward
        current.eval()
        context = torch.inference_mode() if cli.dataset == "RGBNT201" else torch.no_grad()
        with context:
            output = (current(batch, return_aux=True) if cli.dataset == "RGBNT201"
                      else exact_signal_forward(current, batch))
            return {k: v.float().cpu() for k, v in output_mapping(output).items()}

    original_features = features(model)
    del model
    args.method = "SIGNAL_V8_BRANCH_ONLY"
    model, _, config, binding = initialize(args, protocol)
    assert _module_state_sha256(model) == original_sha
    assert all(torch.equal(original_features[k], v) for k, v in features(model).items())
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert original_trainable - trainable == set(binding["disabled_residual_aux_parameters"])
    assert not trainable - original_trainable
    expected_config = copy.deepcopy(original_config)
    expected_config["LOSS"]["ID_RESIDUAL"] = 0.0
    expected_config["LOSS"]["TRIPLET_RESIDUAL"] = 0.0
    assert config == expected_config

    checkpoint = cli.output / "initial_branch_only.pth"
    save_role_checkpoint(checkpoint, model, args, {"protocol_sha256": sha256(protocol_path)}, original_sha)
    del model
    model, _, _, reload_binding = initialize(args, protocol)
    assert reload_binding == binding
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["role_state_dict"]
    assert set(state) == checkpoint_names(model, args.method)
    complete = model.state_dict()
    complete.update(state)
    model.load_state_dict(complete, strict=True)
    assert _module_state_sha256(model) == original_sha
    assert all(torch.equal(original_features[k], v) for k, v in features(model).items())

    # Evaluation mode isolates the algebra from BN buffer updates; M0 checks train mode.
    from trifusion.signal_preserving_v8 import ExpertFormationV8Criterion
    model.eval()
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(batch, return_aux=True)
        parts = ExpertFormationV8Criterion(triplet_margin=.3, label_smoothing=.1).cuda()(output, labels)
        loss = _v27_loss(parts, config)
        original_loss = _v27_loss(parts, original_config)
    assert len(parts) == 14 and torch.isfinite(loss)
    coefficients = {"id_fused": .25, "triplet_fused": 1.0}
    for role in ("cnn", "transformer", "mamba"):
        coefficients.update({f"id_{role}": 1 / 12, f"triplet_{role}": .25,
                             f"id_residual_{role}": 0.0, f"triplet_residual_{role}": 0.0})
    derivatives = torch.autograd.grad(loss, tuple(parts.values()), retain_graph=True)
    for (name, value), derivative in zip(parts.items(), derivatives):
        torch.testing.assert_close(derivative, torch.full_like(value, coefficients[name]), rtol=0, atol=0)
    removed = sum(original_config["LOSS"]["ID_RESIDUAL"] * parts[f"id_residual_{r}"]
                  + original_config["LOSS"]["TRIPLET_RESIDUAL"] * parts[f"triplet_residual_{r}"]
                  for r in ("cnn", "transformer", "mamba"))
    torch.testing.assert_close(original_loss.float() - loss.float(), removed.float(), rtol=1e-5, atol=1e-5)
    loss.backward()
    live = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            if torch.count_nonzero(parameter.grad):
                live.append(name)
        else:
            assert parameter.grad is None, name
    assert all(any(role in name for name in live) for role in ("cnn", "transformer", "mamba"))
    result = dict(status="PASS", dataset=cli.dataset, source_only=True, optimizer_updates=0,
                  source_batch_size=len(labels), initial_features_exact=True, strict_reload_exact=True,
                  loss_coefficients=coefficients, loss=float(loss.detach()),
                  original_loss=float(original_loss.detach()), removed_loss=float(removed.detach()),
                  disabled_parameters=binding["disabled_residual_aux_parameters"],
                  trainable_count=len(trainable), finite_nonzero_gradient_count=len(live),
                  model_state_sha256=original_sha, checkpoint_sha256=sha256(checkpoint),
                  scope="Initial source batch; no optimization; train-mode coverage deferred to M0")
    (cli.output / "witness.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
