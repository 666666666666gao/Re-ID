#!/usr/bin/env python3
"""Compare two completed official receipts without selecting a checkpoint."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare(control_path, candidate_path):
    control, candidate = read(control_path), read(candidate_path)
    for row in (control, candidate):
        assert row["status"] == "COMPLETE" and row["fixed_epoch"] == 20
        assert row["independent_upstream_metrics_equal"]
    assert control["dataset"] == candidate["dataset"]
    assert control["seed"] == candidate["seed"]
    assert control["protocol_sha256"] == candidate["protocol_sha256"]
    assert control["author_checkpoint_sha256"] == candidate["author_checkpoint_sha256"]
    left_train = read(control_path.parent / "training.json")
    right_train = read(candidate_path.parent / "training.json")
    assert left_train["training"]["initial_state_sha256"] == right_train["training"]["initial_state_sha256"]
    assert left_train["initializer"] == right_train["initializer"]
    protocol = read(Path(left_train["protocol"]))
    ids = np.asarray([row["identity"] for row in protocol["records"]["query"]])

    left_signal = control["outputs"]["baseline_only"]
    right_signal = candidate["outputs"]["baseline_only"]
    assert np.array_equal(left_signal["first_match_rank"], right_signal["first_match_rank"])
    assert np.allclose(left_signal["average_precision"], right_signal["average_precision"], atol=1e-6)
    left, right = control["outputs"]["fused"], candidate["outputs"]["fused"]
    ap_left = np.asarray(left["average_precision"], dtype=np.float64)
    ap_right = np.asarray(right["average_precision"], dtype=np.float64)
    rank_left = np.asarray(left["first_match_rank"])
    rank_right = np.asarray(right["first_match_rank"])
    signal_rank = np.asarray(left_signal["first_match_rank"])
    assert len(ids) == len(ap_left) == len(ap_right) == len(rank_left) == len(rank_right)
    delta = ap_right - ap_left
    identity_changes = defaultdict(list)
    for identity, value in zip(ids, delta, strict=True):
        identity_changes[str(identity)].append(float(value))
    identity_mean = {name: float(np.mean(values) * 100) for name, values in identity_changes.items()}
    metric_changes = {name: right["metrics"][name] - left["metrics"][name]
                      for name in ("mAP", "Rank-1", "Rank-5", "Rank-10")}
    return dict(dataset=control["dataset"], seed=control["seed"],
                control_method=control["method"], candidate_method=candidate["method"],
                control_metrics=left["metrics"], candidate_metrics=right["metrics"],
                delta_percentage_points=metric_changes,
                query_ap_improved=int((delta > 1e-12).sum()),
                query_ap_declined=int((delta < -1e-12).sum()),
                query_ap_equal=int((abs(delta) <= 1e-12).sum()),
                candidate_repaired_control_rank1=int(((rank_left > 1) & (rank_right == 1)).sum()),
                candidate_broke_control_rank1=int(((rank_left == 1) & (rank_right > 1)).sum()),
                signal_correct_control_wrong=int(((signal_rank == 1) & (rank_left > 1)).sum()),
                signal_correct_candidate_wrong=int(((signal_rank == 1) & (rank_right > 1)).sum()),
                identities_improved=sum(value > 1e-12 for value in identity_mean.values()),
                identities_declined=sum(value < -1e-12 for value in identity_mean.values()),
                identities_equal=sum(abs(value) <= 1e-12 for value in identity_mean.values()),
                worst_identity_delta_pp=sorted(identity_mean.items(), key=lambda item: item[1])[:3],
                best_identity_delta_pp=sorted(identity_mean.items(), key=lambda item: item[1], reverse=True)[:3],
                worst_query_ap_delta_pp=[dict(index=int(i), delta_pp=float(delta[i] * 100),
                                              control_rank1=bool(rank_left[i] == 1),
                                              candidate_rank1=bool(rank_right[i] == 1))
                                         for i in np.argsort(delta)[:3]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    report = compare(args.control, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
