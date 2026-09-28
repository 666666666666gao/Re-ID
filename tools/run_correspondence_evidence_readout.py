#!/usr/bin/env python3
"""Use the unchanged 50-epoch runner for the M2 selection/readout comparison."""

import argparse
from functools import partial
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "modeling"))

from tools import run_correspondence_roles as runner
from tools.official_three_dataset_model import sha256
from trifusion.correspondence_evidence_readout import EvidenceReadoutTriFusion


BASE_BUILD = runner.build
REFINEMENT = {}


def build(args, protocol):
    assert args.m1 and args.m2 and args.m3 and args.width == 128
    model, cfg, config, binding = BASE_BUILD(args, protocol)
    binding.update(
        architecture="correspondence_evidence_readout_v1",
        refinement=REFINEMENT.copy(),
        entry_sha256=sha256(Path(__file__)),
        reused_entry_sha256=sha256(ROOT / "tools/run_correspondence_roles.py"),
        evidence_source_sha256=sha256(ROOT / "modeling/trifusion/correspondence_evidence_readout.py"),
    )
    return model, cfg, config, binding


def save_checkpoint(path, model, args, protocol, epoch, metrics):
    torch.save({"schema": "trifusion-correspondence-evidence-readout-v1",
                "dataset": args.dataset, "seed": args.seed, "epoch": epoch,
                "protocol_sha256": sha256(args.protocol),
                "baseline_sha256": args.baseline_sha256,
                "variants": {"m1": args.m1, "m2": args.m2, "m3": args.m3},
                "refinement": REFINEMENT.copy(), "metrics": metrics,
                "state": runner.checkpoint_state(model)}, path)


def load_checkpoint(path, model, args, protocol):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["schema"] == "trifusion-correspondence-evidence-readout-v1"
    assert payload["dataset"] == args.dataset and payload["seed"] == args.seed
    assert payload["protocol_sha256"] == sha256(args.protocol)
    assert payload["baseline_sha256"] == args.baseline_sha256
    assert payload["variants"] == {"m1": args.m1, "m2": args.m2, "m3": args.m3}
    assert payload["refinement"] == REFINEMENT
    assert set(payload["state"]) == set(runner.checkpoint_state(model))
    state = model.state_dict()
    state.update(payload["state"])
    model.load_state_dict(state, strict=True)
    return payload


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--selection", choices=("single", "uniform", "query"), required=True)
    parser.add_argument("--structured-readout", action=argparse.BooleanOptionalAction, default=False)
    options, remaining = parser.parse_known_args()
    REFINEMENT.update(selection=options.selection, structured_readout=options.structured_readout)
    runner.CorrespondenceTriFusion = partial(EvidenceReadoutTriFusion, **REFINEMENT)
    runner.build = build
    runner.save_checkpoint = save_checkpoint
    runner.load_checkpoint = load_checkpoint
    sys.argv = [sys.argv[0], *remaining]
    runner.main()


if __name__ == "__main__":
    main()
