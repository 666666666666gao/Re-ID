from pathlib import Path
import argparse,hashlib,json,math
import numpy as np

NAMES = ("baseline_only", "fused", "cnn", "transformer", "mamba")
ENDS = ("control", "joint_tokens")


def main(directory):
    source = directory / "run_summary.json"
    summary = json.loads(source.read_bytes())
    audit = json.loads((directory / "complete_terminal_verification.json").read_bytes())
    assert summary["status"] in ("Q1_PASS", "Q1_FAIL")
    assert summary["repository_commit"] == "bf8de956e685311dd70631395009a2c06a2c8591"
    assert audit["run_summary_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert audit["status"] == "PASS_COMPLETE_V28_FP32_R2_FILES_TRAINING_ARRAYS_RANKINGS_AND_SCORES"
    assert [x["fold"] for x in summary["folds"]] == [0, 1, 2]
    aps = {e: {n: [] for n in NAMES} for e in ENDS}
    ranks = {e: {n: [] for n in NAMES} for e in ENDS}
    identities, fold_gains = [], []
    step_count = 0
    max_loss_error = 0.
    for fold in summary["folds"]:
        gallery = fold["gallery_manifest"]
        cameras_by_id = {}
        for row in gallery:
            cameras_by_id.setdefault(row["identity"], set()).add(row["camera"])
        eligible = [i for i, row in enumerate(gallery) if len(cameras_by_id[row["identity"]]) > 1]
        assert len(eligible) == (190, 179, 202)[fold["fold"]]
        identities.extend(gallery[i]["identity"] for i in eligible)
        for endpoint in ENDS:
            saved = fold["endpoints"][endpoint]
            training = saved["training"]
            assert training["epochs"] == 20 and len(training["history"]) == 20
            raw = (directory / Path(training["all_training_steps_path"]).name).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == training["all_training_steps_sha256"]
            rows = [json.loads(x) for x in raw.splitlines()]
            assert len(rows) == training["optimizer_steps"] == (580, 560, 540)[fold["fold"]]
            step_count += len(rows)
            for row in rows:
                losses = row["losses"]
                value = .25 * losses["id_fused"] + losses["triplet_fused"]
                for role in NAMES[2:]:
                    value += (losses["id_" + role] + losses["id_residual_" + role]) / 12
                    value += .25 * (losses["triplet_" + role] + losses["triplet_residual_" + role])
                assert math.isfinite(value) and not row["overflow"]
                max_loss_error = max(max_loss_error, abs(value - losses["total"]))
            for name in NAMES:
                output = saved["outputs"][name]
                assert output["query_indices"] == eligible
                aps[endpoint][name].extend(output["average_precision"])
                ranks[endpoint][name].extend(output["first_match_rank"])
        fold_gains.append(
            float(np.mean(fold["endpoints"]["joint_tokens"]["outputs"]["fused"]["average_precision"]) * 100)
            - float(np.mean(fold["endpoints"]["control"]["outputs"]["fused"]["average_precision"]) * 100))
    assert step_count == 3360 and len(identities) == 571 and len(set(identities)) == 21
    assert max_loss_error < 2e-6
    aggregate, metric_error = {}, 0.
    for endpoint in ENDS:
        aggregate[endpoint] = {}
        for name in NAMES:
            ap, rank = np.asarray(aps[endpoint][name]), np.asarray(ranks[endpoint][name])
            assert len(ap) == len(rank) == 571
            result = {"mAP": float(ap.mean() * 100),
                      **{f"Rank-{k}": float(np.mean(rank <= k) * 100) for k in (1, 5, 10)}}
            aggregate[endpoint][name] = result
            metric_error = max(metric_error, *(abs(v-summary["aggregate"][endpoint][name][k]) for k,v in result.items()))
    assert metric_error < 1e-10
    gains = {n: aggregate["joint_tokens"][n]["mAP"] - aggregate["control"][n]["mAP"] for n in NAMES}
    labels = np.asarray(identities)
    delta = np.asarray(aps["joint_tokens"]["fused"]) - np.asarray(aps["control"]["fused"])
    ids = np.unique(labels)
    sums = np.asarray([delta[labels == identity].sum() for identity in ids])
    counts = np.asarray([(labels == identity).sum() for identity in ids])
    draws = np.random.default_rng(42).choice(np.arange(21), size=(10000,21), replace=True)
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lower = float(np.quantile(means, .025) * 100)
    assert abs(lower-summary["bootstrap"]["lower_bound_95_mAP"]) < 1e-10
    checks = {
        "aggregate_fused_gain_at_least_1pp": gains["fused"] >= 1.,
        "all_fold_fused_nonnegative": all(x >= 0 for x in fold_gains),
        "all_expert_aggregate_nonnegative": all(gains[n] >= 0 for n in NAMES[2:]),
        "fused_bootstrap_lower_positive": lower > 0,
        "fused_beats_baseline_and_experts": all(aggregate["joint_tokens"]["fused"]["mAP"] > aggregate["joint_tokens"][n]["mAP"] for n in ("baseline_only", *NAMES[2:])),
    }
    assert checks == summary["scientific_checks"] == audit["scientific_checks"]
    assert summary["status"] == ("Q1_PASS" if all(checks.values()) else "Q1_FAIL")
    proof = {"status": "PASS_LOCAL_FULL_TERMINAL_SCALAR_RECOMPUTATION",
             "scientific_status": summary["status"], "scientific_checks": checks,
             "aggregate": aggregate, "matched_gains_mAP": gains, "fold_fused_gains_mAP": fold_gains,
             "bootstrap_lower_95_mAP": lower, "checked_updates": step_count,
             "all_five_outputs_queries": 571, "identity_clusters": 21,
             "maximum_loss_recompute_error": max_loss_error, "maximum_metric_error_pp": metric_error,
             "source_summary_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "numpy_version": np.__version__, "local_torch_or_model_image_calls": 0,
             "new_training_or_retrieval": 0, "independent_external_review": False}
    output = directory / "local_terminal_scalar_recomputation.json"
    assert not output.exists()
    output.write_bytes((json.dumps(proof, indent=2) + chr(10)).encode())
    print(json.dumps(proof))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    main(parser.parse_args().directory)
