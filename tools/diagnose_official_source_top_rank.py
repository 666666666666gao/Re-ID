#!/usr/bin/env python3
"""Measure top-rank relation support on official training identities only."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from tools.official_three_dataset_model import sha256
from tools.run_official_three_dataset_roles import extract, initialize, read_protocol
from tools.run_signal_preserving_v5 import _module_state_sha256
from tools.train_msvr310_signal_oof import scene_scores


def summarize(features, rows, environment_key):
    signal = torch.nn.functional.normalize(features["baseline_only"].float(), dim=1)
    fused = torch.nn.functional.normalize(features["fused"].float(), dim=1)
    identities = torch.tensor([row["identity"] for row in rows])
    environments = torch.tensor([row[environment_key] for row in rows])
    same_identity = identities[:, None] == identities[None, :]
    positive = same_identity & (environments[:, None] != environments[None, :])
    negative = ~same_identity
    eligible = positive.any(dim=1)
    assert negative.any(dim=1).all()
    signal_scores = signal @ signal.T
    fused_scores = fused @ fused.T
    margins = {}
    for name, scores in (("signal", signal_scores), ("fused", fused_scores)):
        best_positive = scores.masked_fill(~positive, -torch.inf).max(dim=1).values
        best_negative = scores.masked_fill(~negative, -torch.inf).max(dim=1).values
        margins[name] = (best_positive - best_negative)[eligible]
        distance = (2 - 2 * scores)[eligible].numpy()
        checked = scene_scores(distance, identities[eligible].numpy(), identities.numpy(),
                               environments[eligible].numpy(), environments.numpy())
        assert abs(checked["metrics"]["Rank-1"]
                   - 100 * int((margins[name] > 0).sum()) / int(eligible.sum())) < 1e-10
    base, candidate = margins["signal"], margins["fused"]
    positive_index = signal_scores.masked_fill(~positive, -torch.inf).argmax(dim=1)
    negative_index = signal_scores.masked_fill(~negative, -torch.inf).argmax(dim=1)
    position = torch.arange(len(rows))
    same_pair_candidate = (fused_scores[position, positive_index]
                           - fused_scores[position, negative_index])[eligible]
    correct = base > 0
    candidate_correct = candidate > 0
    sigmoid = torch.sigmoid(-candidate / 0.01)
    return {
        "source_records": len(rows),
        "source_identities": len(set(identities.tolist())),
        "eligible_queries": int(eligible.sum()),
        "eligible_identities": len(set(identities[eligible].tolist())),
        "signal_top1_correct": int(correct.sum()),
        "fused_top1_correct": int(candidate_correct.sum()),
        "signal_correct_fused_wrong": int((correct & ~candidate_correct).sum()),
        "signal_wrong_fused_correct": int((~correct & candidate_correct).sum()),
        "signal_correct_same_pair_fused_wrong": int((correct & (same_pair_candidate <= 0)).sum()),
        "signal_correct_same_pair_drop_gt_0_02": int((correct & (same_pair_candidate < base - 0.02)).sum()),
        "signal_correct_near_0_02_same_pair_drop_gt_0_02": int((correct & (base <= 0.02)
                                                                & (same_pair_candidate < base - 0.02)).sum()),
        "signal_correct_margin_le_0_01": int((correct & (base <= 0.01)).sum()),
        "signal_correct_margin_le_0_02": int((correct & (base <= 0.02)).sum()),
        "signal_correct_margin_le_0_05": int((correct & (base <= 0.05)).sum()),
        "fused_margin_nonpositive": int((candidate <= 0).sum()),
        "top1_margin_sensitivity_gt_0_01": int((sigmoid > 0.01).sum()),
        "top1_margin_sensitivity_gt_0_1": int((sigmoid > 0.1).sum()),
        "signal_margin_quantiles": torch.quantile(base, torch.tensor([0., .1, .5, .9, 1.])).tolist(),
        "fused_margin_quantiles": torch.quantile(candidate, torch.tensor([0., .1, .5, .9, 1.])).tolist(),
        "boundary": "Training identities only; author Signal has seen these identities. No official query/gallery scores or gradients are used.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--signal-source", type=Path, required=True)
    parser.add_argument("--clip-weight", type=Path, required=True)
    parser.add_argument("--role-state", choices=("initial", "final"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.training_receipt.read_text(encoding="utf-8"))
    assert receipt["status"] == "FIXED_EPOCH20_TRAINING_COMPLETE"
    assert receipt["dataset"] == "MSVR310" and receipt["method"] == "R2"
    protocol_path = Path(receipt["protocol"])
    protocol = read_protocol(protocol_path, receipt["dataset"])
    assert sha256(protocol_path) == receipt["protocol_sha256"]
    assert sha256(receipt["checkpoint"]) == receipt["checkpoint_sha256"]
    model_args = SimpleNamespace(
        dataset=receipt["dataset"], method=receipt["method"],
        signal_source=args.signal_source, clip_weight=args.clip_weight,
        signal_checkpoint=Path(receipt["initializer"]["author_checkpoint"]),
        signal_sha256=receipt["initializer"]["author_checkpoint_sha256"],
        seed=receipt["seed"],
    )
    model, _, _, binding = initialize(model_args, protocol)
    assert binding == receipt["initializer"]
    if args.role_state == "final":
        payload = torch.load(receipt["checkpoint"], map_location="cpu", weights_only=True)
        assert payload["dataset"] == receipt["dataset"]
        assert payload["method"] == receipt["method"]
        state = model.state_dict()
        assert set(payload["role_state_dict"]) == {name for name in state if not name.startswith("baseline.")}
        state.update(payload["role_state_dict"])
        model.load_state_dict(state, strict=True)
        assert _module_state_sha256(model) == receipt["training"]["final_state_sha256"]
    else:
        assert _module_state_sha256(model) == receipt["training"]["initial_state_sha256"]
    features = extract(model, protocol, "train", receipt["method"])
    result = {
        "schema": "trifusion-msvr310-source-top-rank-probe-v3",
        "completed_at": datetime.now().astimezone().isoformat(),
        "role_state": args.role_state,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "training_receipt_sha256": sha256(args.training_receipt),
        "protocol_sha256": sha256(protocol_path),
        "loaded_role_checkpoint_sha256": sha256(receipt["checkpoint"]) if args.role_state == "final" else None,
        "result": summarize(features, protocol["records"]["train"], protocol["environment_key"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert not args.output.exists()
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
