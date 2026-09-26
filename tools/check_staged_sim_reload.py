"""Source-only check of frozen-phase selection flags and strict joint reload."""

import argparse
from argparse import Namespace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from tools.queue_official_extra_seed import WEIGHTS
    from tools.run_official_three_dataset_roles import (
        initialize, read_protocol, save_role_checkpoint, checkpoint_names, distance_matrix,
    )
    from tools.official_three_dataset_data import records_for, loader_for
    from tools.run_signal_preserving_v5 import _training_batch, _module_state_sha256
    from tools.train_msvr310_trifusion_oof import output_mapping
    from tools.train_signal_preserving_v18 import image_batch

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(WEIGHTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    assert not cli.output.exists()
    cli.output.mkdir(parents=True)
    weight, digest = WEIGHTS[cli.dataset]
    args = Namespace(dataset=cli.dataset, method="SIGNAL_SIM_JOINT_STAGED", seed=42,
                     signal_source=ROOT / "comparators/Signal-cd1b0a6",
                     clip_weight=ROOT / "pertrained-model/ViT-B-16.pt",
                     signal_checkpoint=ROOT / "pertrained-model" / weight,
                     signal_sha256=digest)
    protocol_path = ROOT / "logs/official_three_dataset_protocols_20260923" / f"{cli.dataset}.json"
    protocol = read_protocol(protocol_path, cli.dataset)
    model, _, _, binding = initialize(args, protocol)
    raw = next(iter(loader_for(protocol, records_for(protocol, "train"),
                               training=False, method=args.method)))
    if cli.dataset == "RGBNT201":
        images, _, _, cameras, _, _ = raw
        batch = image_batch(images, cameras)
    else:
        batch, _ = _training_batch(raw)
    sim = list(model.baseline.signal.SIM.modal_interactive.parameters())
    assert len(sim) == 12
    for p in sim:
        p.requires_grad_(False)
    # The training callback restores these flags before any best-epoch evaluation.
    for p in sim:
        p.requires_grad_(True)

    def features(current):
        current.eval()
        context = torch.inference_mode() if cli.dataset == "RGBNT201" else torch.no_grad()
        with context:
            return {k: v.float().cpu() for k, v in output_mapping(current(batch, return_aux=True)).items()}

    before = features(model)
    state_sha = _module_state_sha256(model)
    checkpoint = cli.output / "initial_joint.pth"
    from tools.official_three_dataset_model import sha256
    save_role_checkpoint(checkpoint, model, args, {"protocol_sha256": sha256(protocol_path)}, state_sha)
    del model, sim
    reloaded, _, _, reload_binding = initialize(args, protocol)
    assert binding == reload_binding
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["role_state_dict"]
    assert set(state) == checkpoint_names(reloaded, args.method)
    assert sum(k.startswith("baseline.signal.SIM.modal_interactive.") for k in state) == 12
    complete = reloaded.state_dict()
    complete.update(state)
    reloaded.load_state_dict(complete, strict=True)
    assert _module_state_sha256(reloaded) == state_sha
    after = features(reloaded)
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert all(torch.equal(distance_matrix(before[k], before[k]),
                           distance_matrix(after[k], after[k])) for k in before)
    result = dict(status="PASS", dataset=cli.dataset, source_only=True, optimizer_updates=0,
                  scope="Fresh initialization frozen-phase evaluation flags and strict reload; not learned SIM validation",
                  source_batch_size=len(before["fused"]), all_output_features_exact=True,
                  all_output_distances_exact=True, sim_tensors=12, model_state_sha256=state_sha,
                  checkpoint_sha256=sha256(checkpoint))
    (cli.output / "witness.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
