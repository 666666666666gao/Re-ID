#!/usr/bin/env python3
"""Read-only terminal files and saved retrieval-array verification on the GPU host."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    import numpy as np
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    run, project = args.run_root.resolve(), args.project_root.resolve()
    output = run / "baseline_terminal_file_verification.json"
    ranks_path = run / "baseline_saved_distance_rankings.json"
    assert not output.exists() and not ranks_path.exists()
    assert int((run / "baseline_exit.txt").read_text()) == 0
    started = time.perf_counter()
    summary_path = run / "baseline/summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "COMPLETE_BASELINE_NOT_METHOD_QUALIFICATION"
    assert summary["mode"] == "train" and len(summary["folds"]) == 3
    config_path = project / "configs/MSVR310/Signal-source-oof-v1.json"
    config = json.loads(config_path.read_text())
    assert sha256(config_path) == summary["config_sha256"]
    assert sha256(project / config["protocol"]) == summary["protocol_sha256"]
    assert sha256(run / "m0/summary.json") == summary["preflight_receipt_sha256"]
    for name, expected in config["project_source_file_sha256"].items():
        assert sha256(project / name) == expected, name
    signal = Path(config["signal_source"])
    for name, expected in config["signal_source_file_sha256"].items():
        assert sha256(signal / name) == expected, name
    assert subprocess.check_output(["git", "-C", str(signal), "rev-parse", "HEAD"], text=True).strip() == config["signal_commit"]
    assert hashlib.sha256(subprocess.check_output(["git", "-C", str(signal), "diff", "--binary"])).hexdigest() == config["signal_diff_sha256"]
    assert sha256(config["clip_weight"]) == config["clip_weight_sha256"]
    protocol = json.loads((project / config["protocol"]).read_text())
    files = {}
    selected = sorted((run / "baseline").rglob("*"))
    selected += [run / name for name in ("baseline.log", "baseline_exit.txt", "baseline_launch.json",
                 "baseline_wrapper.log", "baseline_wrapper.py", "original_baseline_launch.json")]
    for path in selected:
        if path.is_file():
            files[path.relative_to(run).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    rankings, comparisons = [], []
    for actual, fold in zip(summary["folds"], protocol["folds"], strict=True):
        directory = run / "baseline" / f"fold_{fold['fold']}"
        assert actual == json.loads((directory / "receipt.json").read_text())
        assert actual["training"] == json.loads((directory / "training.json").read_text())
        checkpoint = directory / "signal_epoch50.pth"
        assert str(checkpoint) == actual["checkpoint"]
        assert files[checkpoint.relative_to(run).as_posix()]["sha256"] == actual["checkpoint_sha256"]
        array_path = directory / "retrieval_arrays.pt"
        array_sha = files[array_path.relative_to(run).as_posix()]["sha256"]
        assert array_sha == actual["retrieval"]["retrieval_arrays_sha256"]
        arrays = torch.load(array_path, map_location="cpu", weights_only=True)
        features, distances = arrays["features"], arrays["distances"]
        gallery_indices = fold["gallery_record_indices"]
        positions = [row["gallery_position"] for row in fold["query_rows"]]
        assert arrays["gallery_record_indices"] == gallery_indices
        assert arrays["query_gallery_positions"] == positions
        assert tuple(features.shape) == (len(gallery_indices), 3072)
        assert tuple(distances.shape) == (len(positions), len(gallery_indices))
        assert torch.isfinite(features).all() and torch.isfinite(distances).all()
        normalized = torch.nn.functional.normalize(features.float(), dim=1)
        qf = normalized[positions]
        recalculated = qf.square().sum(1, keepdim=True) + normalized.square().sum(1)[None]
        recalculated.addmm_(qf, normalized.T, beta=1, alpha=-2)
        discrepancy = float((recalculated - distances).abs().max())
        order = np.argsort(distances.numpy(), axis=1)
        rankings.append({"fold": fold["fold"], "gallery_record_indices": gallery_indices,
                         "query_gallery_positions": positions, "sorted_gallery_positions": order.tolist(),
                         "source_array_sha256": array_sha})
        comparisons.append({"fold": fold["fold"], "saved_feature_shape": list(features.shape),
                            "saved_distance_shape": list(distances.shape),
                            "recomputed_distance_max_absolute_difference": discrepancy,
                            "recomputed_distance_bitwise_equal": bool(torch.equal(recalculated, distances)),
                            "rankings_from_saved_distances": True})
        assert sha256(array_path) == array_sha
    ranks_path.write_text(json.dumps({"scope": "Complete categorical rankings from original saved distances; no model/image forwards",
                                      "summary_sha256": sha256(summary_path), "folds": rankings}, indent=2) + "\n")
    result = {"verified_at": datetime.now(timezone.utc).isoformat(),
              "scope": "Whole files plus saved feature/distance arithmetic; no new model, image, optimizer or backward execution",
              "run_root": str(run), "execution_commit": summary["project_commit"],
              "verification_project_commit": subprocess.check_output(["git", "-C", str(project), "rev-parse", "HEAD"], text=True).strip(),
              "verifier_sha256": sha256(__file__), "summary_sha256": sha256(summary_path),
              "files": files, "checkpoint_files_verified": 3, "retrieval_array_files_verified": 3,
              "fold_receipts_equal_summary": True, "training_files_equal_receipts": True,
              "project_source_bindings_verified": len(config["project_source_file_sha256"]),
              "signal_source_bindings_verified": len(config["signal_source_file_sha256"]),
              "saved_array_comparisons": comparisons, "rankings_sha256": sha256(ranks_path),
              "elapsed_seconds": time.perf_counter() - started}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "files"}))


if __name__ == "__main__":
    main()
