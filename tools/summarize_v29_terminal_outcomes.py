from pathlib import Path
import argparse, hashlib, json
from datetime import datetime

NAMES = ("baseline_only", "fused", "cnn", "transformer", "mamba")

def main(directory):
    report_path = directory / "complete_comparison.json"
    report = json.loads(report_path.read_bytes())
    local = json.loads((directory / "local_terminal_scalar_recomputation.json").read_bytes())
    assert local["status"] == "PASS_LOCAL_COMPLETE_V29_SCALARS_AND_GEOMETRY_RECEIPTS"
    assert local["scientific_status"] == report["scientific_status"]
    assert len(report["all_identity_rows"]) == 21 and len(report["all_query_rows"]) == 571
    changes = {}
    for name in NAMES:
        rows = report["all_query_rows"]
        repaired = [r for r in rows if r["outputs"][name]["control_first_match_rank"] > 1
                    and r["outputs"][name]["candidate_first_match_rank"] == 1]
        broken = [r for r in rows if r["outputs"][name]["control_first_match_rank"] == 1
                  and r["outputs"][name]["candidate_first_match_rank"] > 1]
        same_camera_broken = [r for r in broken if r["camera"] == r["outputs"][name]["candidate_first_gallery"]["camera"]]
        identity_rows = []
        for row in report["all_identity_rows"]:
            gain = row["gains_percentage_points"][name]["mAP"]
            identity_rows.append({"fold": row["fold"], "identity": row["original_identity"],
                                  "queries": row["queries"], "gain_mAP_pp": gain,
                                  "weighted_net_contribution_pp": gain * row["queries"] / 571})
        net = sum(r["weighted_net_contribution_pp"] for r in identity_rows)
        assert abs(net - report["matched_gains_mAP"][name]) < 1e-10
        positive = sorted([r for r in identity_rows if r["gain_mAP_pp"] > 1e-10],
                          key=lambda r: r["weighted_net_contribution_pp"], reverse=True)
        negative = [r for r in identity_rows if r["gain_mAP_pp"] < -1e-10]
        changes[name] = {
            "identity_improved": len(positive), "identity_declined": len(negative),
            "identity_equal": 21-len(positive)-len(negative),
            "all_identity_contributions": identity_rows, "positive_identity_contributions_descending": positive,
            "positive_contribution_pp": sum(r["weighted_net_contribution_pp"] for r in positive),
            "negative_contribution_pp": sum(r["weighted_net_contribution_pp"] for r in negative),
            "net_contribution_pp": net,
            "top_two_positive_contribution_pp": sum(r["weighted_net_contribution_pp"] for r in positive[:2]),
            "rank1_repaired": len(repaired), "rank1_broken": len(broken),
            "same_camera_among_broken": len(same_camera_broken),
            "all_broken_queries": [{"fold": r["fold"], "file": r["file"], "camera": r["camera"],
                                   "candidate_first_gallery": r["outputs"][name]["candidate_first_gallery"]}
                                  for r in broken]}
        expected = report["all_query_paired_changes"][name]
        assert len(repaired) == expected["rank1_repaired"] and len(broken) == expected["rank1_broken"]
    costs = {}
    for endpoint in ("control", "bounded_joint"):
        rows = [r for r in report["all_checkpoint_training_bindings"] if r["endpoint"] == endpoint]
        assert len(rows) == 3 and sum(r["training"]["optimizer_steps"] for r in rows) == 1680
        seconds = sum(r["training"]["elapsed_seconds"] for r in rows)
        costs[endpoint] = {"training_seconds": seconds, "seconds_per_update": seconds / 1680,
                           "maximum_peak_reserved_mib": max(r["training"]["peak_reserved_mib"] for r in rows)}
    costs["candidate_training_time_increase_percent"] = (
        costs["bounded_joint"]["training_seconds"] / costs["control"]["training_seconds"] - 1) * 100
    training_terms = {}
    loss_names = ("id_fused", "triplet_fused",
                  *[n for role in NAMES[2:] for n in
                    ("id_" + role, "triplet_" + role, "id_residual_" + role, "triplet_residual_" + role)])
    for endpoint in ("control", "bounded_joint"):
        records = []
        for fold in range(3):
            p = directory / f"fold_{fold}_{endpoint}_all_training_steps.jsonl"
            records.extend(json.loads(line) for line in p.read_bytes().splitlines())
        assert len(records) == 1680
        training_terms[endpoint] = {}
        for period in ("all", "epoch20"):
            selected = records if period == "all" else [r for r in records if r["epoch"] == 20]
            assert len(selected) == (1680 if period == "all" else 84)
            means = {n: sum(r["losses"][n] for r in selected) / len(selected) for n in loss_names}
            weighted_id = .25 * means["id_fused"] + sum(
                (means["id_" + role] + means["id_residual_" + role]) / 12 for role in NAMES[2:])
            weighted_triplet = means["triplet_fused"] + sum(
                .25 * (means["triplet_" + role] + means["triplet_residual_" + role]) for role in NAMES[2:])
            saved_mean = sum(r["losses"]["total"] for r in selected) / len(selected)
            assert abs(weighted_id + weighted_triplet - saved_mean) < 2e-6
            training_terms[endpoint][period] = {
                "steps": len(selected), "all14_loss_means": means,
                "weighted_id_total": weighted_id, "weighted_triplet_total": weighted_triplet,
                "total_mean": saved_mean,
                "fused_triplet_positive_batches": sum(r["losses"]["triplet_fused"] > 0 for r in selected),
                "joint_correction_abs_mean": sum(r["losses"]["joint_correction_abs_mean"] for r in selected) / len(selected)}
    result = {"generated_at": datetime.now().astimezone().isoformat(),
              "status": "COMPLETE_TERMINAL_DESCRIPTIVE_DIGEST",
              "scientific_status": report["scientific_status"], "changes": changes, "costs": costs, "training_terms": training_terms,
              "source_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "boundaries": ["All 571 queries and all 21 identities included.",
                             "Identity sign counts use 1e-10 pp tolerance only; science gates unchanged.",
                             "Camera association is descriptive and does not establish visual cause.",
                             "Training time is observed cost, not inference latency or FLOPs.",
                             "Training losses describe full trajectories and epoch20; they do not prove gradient conflict.",
                             "No new training, retrieval, model, image access, or independent external review."]}
    output = directory / "terminal_outcome_digest.json"
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2) + chr(10), encoding="utf-8")
    print(json.dumps({"scientific_status": result["scientific_status"], "costs": costs,
                      "changes": {n: {k: v for k,v in row.items() if not isinstance(v,list)} for n,row in changes.items()}}))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    main(parser.parse_args().directory)
