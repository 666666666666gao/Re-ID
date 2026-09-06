#!/usr/bin/env python3
"""Report every fixed source-style condition after complete array verification."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

STATES = ("initial", "control_final", "style_final")
VIEWS = ("original", "registered_style")
PROTOCOLS = ("identity", "cross_camera")
STRATA = ("active", "inactive")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(data)
    return digest.hexdigest()


def group_key(row):
    return (row["fold"], row["state"], row["view"], row["protocol"], row["plan_stratum"])


def descriptor(key):
    return dict(zip(("fold", "state", "view", "protocol", "plan_stratum"), key, strict=True))


def new_group():
    return {
        "cases": 0, "triplets": 0,
        "nonpositive": [0] * 18, "hinge_positive": [0] * 18,
        "margin_sum": [0.] * 18, "hinge_sum": [0.] * 18,
        "weight_sum": [0.] * 9, "joint_nonpositive": [0] * 9,
        "support": [[0] * 16 for _ in range(9)],
        "auxiliary_sum": 0., "derivative_ratio_sum": 0.,
    }


def append_case(group, value):
    count = value["triplets"]
    group["cases"] += 1
    group["triplets"] += count
    for i, row in enumerate(value["margin_values"]):
        group["nonpositive"][i] += int(row[3])
        group["hinge_positive"][i] += int(row[5])
        group["margin_sum"][i] += row[0] * count
        group["hinge_sum"][i] += row[4] * count
    for i, row in enumerate(value["weight_values"]):
        group["weight_sum"][i] += row[0] * count
        group["joint_nonpositive"][i] += int(row[7])
        for j, number in enumerate(value["slot_nonpositive_support_patterns"][i]):
            group["support"][i][j] += number
    group["auxiliary_sum"] += value["v26_auxiliary_loss"] * count
    group["derivative_ratio_sum"] += value["v26_to_uniform_margin_derivative_norm_ratio"]


def combine(groups):
    out = new_group()
    for value in groups:
        for key in ("cases", "triplets", "auxiliary_sum", "derivative_ratio_sum"):
            out[key] += value[key]
        for key in ("nonpositive", "hinge_positive", "margin_sum", "hinge_sum", "weight_sum", "joint_nonpositive"):
            out[key] = [a + b for a, b in zip(out[key], value[key], strict=True)]
        for i in range(9):
            out["support"][i] = [a + b for a, b in zip(out["support"][i], value["support"][i], strict=True)]
    return out


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def report(args):
    run = args.run_dir
    assert not args.output_dir.exists()
    assert int(Path(str(run) + ".exit").read_text()) == 0
    assert int((run / "verification.exit").read_text()) == 0
    terminal = json.loads(Path(str(run) + ".terminal.json").read_bytes())
    pipeline = json.loads((run / "pipeline.json").read_bytes())
    assert terminal["exit_code"] == pipeline["verification_exit_code"] == 0
    assert pipeline["stage"] == "PIPELINE_COMPLETE"
    assert not Path("/proc", str(terminal["original_pid"])).exists()
    assert not Path("/proc", str(pipeline["verification_pid"])).exists()

    summary_path = run / "style_relation_support.json"
    proof_path = run / "verification.json"
    summary, proof = json.loads(summary_path.read_bytes()), json.loads(proof_path.read_bytes())
    assert summary["status"] == "COMPLETE_FIXED_SOURCE_STYLE_RELATION_DIAGNOSTIC_PENDING_CPU_VERIFICATION"
    assert proof["status"] == "PASS_ALL_FIXED_SOURCE_STYLE_RELATIONS_AND_AGGREGATES"
    assert proof["source_summary_sha256"] == sha(summary_path)
    assert summary["completed_batches"] == 1680 and summary["model_forwards"] == 10080
    assert proof["checked_model_forward_cases"] == 10080
    assert proof["checked_protocol_triplets"] == 273297024
    assert proof["checked_similarity_values"] == 743178240
    assert proof["all_source_exposures_match"] and proof["all_reported_count_fields_exact"]
    assert summary["new_optimizer_updates"] == summary["dev_image_reads"] == summary["official_image_reads"] == summary["heldout_image_reads"] == 0
    assert not summary["encoder_gradients_computed"]
    assert (tuple(summary["states"]), tuple(summary["views"]), tuple(summary["protocols"])) == (STATES, VIEWS, PROTOCOLS)
    outputs = summary["outputs"]
    assert len(outputs) == 18

    original_ids, exposure, plans = {}, defaultdict(Counter), {}
    for fold in summary["folds"]:
        number = fold["fold"]
        manifest = fold["source_manifest"]
        original_ids[number] = {row["identity"]: row["file"].split("_", 1)[0] for row in manifest}
        assert len(original_ids[number]) == 94
        receipt_path = Path(fold["batch_receipts"]["path"])
        assert sha(receipt_path) == fold["batch_receipts"]["sha256"]
        receipts = [json.loads(x) for x in receipt_path.read_text().splitlines()]
        assert len(receipts) == (580, 560, 540)[number]
        for index, row in enumerate(receipts):
            assert row["case_index"] == index
            identities = [manifest[i]["identity"] for i in row["sampler_indices"]]
            cameras = [manifest[i]["camera"] for i in row["sampler_indices"]]
            counts = Counter(identities)
            assert sorted(counts.values()) == [8] * 8
            stratum = "active" if row["style_plan"]["active"] else "inactive"
            plans[number, index] = stratum
            for protocol in PROTOCOLS:
                target = exposure[number, protocol, stratum]
                for i, identity in enumerate(identities):
                    positives = sum(j != i and other == identity and
                                    (protocol == "identity" or cameras[j] != cameras[i])
                                    for j, other in enumerate(identities))
                    target[identity] += positives * 56
    assert Counter(plans.values()) == {"active": 819, "inactive": 861}
    assert len(plans) == 1680

    case_path = Path(summary["cases"]["path"])
    assert sha(case_path) == summary["cases"]["sha256"]
    groups = defaultdict(new_group)
    seen = set()
    with gzip.open(case_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            identifier = (row["fold"], row["case_index"], row["state"], row["view"])
            assert identifier not in seen
            seen.add(identifier)
            stratum = plans[row["fold"], row["case_index"]]
            assert row["plan_active"] == (stratum == "active")
            assert row["actual_style_active"] == (stratum == "active" and row["view"] == "registered_style")
            assert row["state"] in STATES and row["view"] in VIEWS
            for protocol in PROTOCOLS:
                append_case(groups[row["fold"], row["state"], row["view"], protocol, stratum], row["protocols"][protocol])
    assert len(seen) == 10080
    assert set(groups) == set(itertools.product(range(3), STATES, VIEWS, PROTOCOLS, STRATA))
    assert sum(value["triplets"] for value in groups.values()) == 273297024
    verified = {group_key(row): row for row in proof["aggregates"]}
    assert set(verified) == set(groups) and len(groups) == 72
    identity_rows = []
    for key, value in groups.items():
        audit = verified[key]
        assert (value["cases"], value["triplets"]) == (audit["cases"], audit["triplets"])
        assert value["nonpositive"] == audit["nonpositive_per_output"]
        assert value["hinge_positive"] == audit["hinge_positive_per_output"]
        for raw_key, mean_key in (("margin_sum", "mean_margin_per_output"), ("hinge_sum", "mean_euclidean_hinge_per_output"), ("weight_sum", "mean_slot_weight")):
            for observed, expected in zip(value[raw_key], audit[mean_key], strict=True):
                assert math.isclose(observed / value["triplets"], expected, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(value["auxiliary_sum"] / value["triplets"], audit["mean_auxiliary_loss"], rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(value["derivative_ratio_sum"] / value["cases"], audit["mean_margin_derivative_ratio_per_batch"], rel_tol=1e-9, abs_tol=1e-9)
        for i, cells in enumerate(value["support"]):
            assert sum(cells) == value["nonpositive"][5 + i]
            assert sum(c for bits, c in enumerate(cells) if bits & 1) == value["joint_nonpositive"][i]
        fold, state, view, protocol, stratum = key
        denominators = exposure[fold, protocol, stratum]
        assert sum(denominators.values()) == value["triplets"]
        assert sum(audit["fused_nonpositive_by_source_identity"].values()) == value["nonpositive"][1]
        assert set(audit["fused_nonpositive_by_source_identity"]) <= set(original_ids[fold].values())
        for identity, original in sorted(original_ids[fold].items()):
            count = audit["fused_nonpositive_by_source_identity"].get(original, 0)
            denominator = denominators[identity]
            assert 0 <= count <= denominator
            identity_rows.append({**descriptor(key), "encoded_identity": identity, "original_identity": original,
                                  "query_relation_exposures": denominator, "fused_nonpositive": count,
                                  "fused_nonpositive_percent": count / denominator * 100 if denominator else None})
    assert len(identity_rows) == 72 * 94

    for state, view, protocol, stratum in itertools.product(STATES, VIEWS, PROTOCOLS, STRATA):
        groups["pooled", state, view, protocol, stratum] = combine(groups[f, state, view, protocol, stratum] for f in range(3))
    for fold, state, view, protocol in itertools.product((0, 1, 2, "pooled"), STATES, VIEWS, PROTOCOLS):
        groups[fold, state, view, protocol, "all"] = combine(groups[fold, state, view, protocol, stratum] for stratum in STRATA)
    assert len(groups) == 144

    metrics, weights, signs, conditions = [], [], [], []
    for key, value in groups.items():
        common = descriptor(key)
        n, batches = value["triplets"], value["cases"]
        conditions.append({**common, "cases": batches, "triplets": n,
                           "auxiliary_loss": value["auxiliary_sum"] / n,
                           "mean_margin_derivative_ratio_per_batch": value["derivative_ratio_sum"] / batches})
        for i, name in enumerate(outputs):
            metrics.append({**common, "output": name, "cases": batches, "triplets": n,
                            "nonpositive": value["nonpositive"][i], "nonpositive_percent": value["nonpositive"][i] / n * 100,
                            "hinge_positive": value["hinge_positive"][i], "hinge_positive_percent": value["hinge_positive"][i] / n * 100,
                            "mean_cosine_margin": value["margin_sum"][i] / n, "mean_euclidean_hinge": value["hinge_sum"][i] / n})
        for i, name in enumerate(outputs[5:14]):
            cells = value["support"][i]
            weights.append({**common, "slot": name, "triplets": n, "mean_weight": value["weight_sum"][i] / n,
                            "joint_slot_and_fused_nonpositive": value["joint_nonpositive"][i]})
            signs.append({**common, "slot": name, "slot_nonpositive": sum(cells),
                          "same_role_positive_and_fused_positive": sum(c for bits, c in enumerate(cells) if not bits & 3),
                          "bank_positive_and_fused_positive": sum(c for bits, c in enumerate(cells) if not bits & 5),
                          "role_nonpositive_bank_positive_fused_positive": sum(c for bits, c in enumerate(cells) if bits & 2 and not bits & 5),
                          "bank_nonpositive_fused_positive": sum(c for bits, c in enumerate(cells) if bits & 4 and not bits & 1),
                          "fused_nonpositive": sum(c for bits, c in enumerate(cells) if bits & 1),
                          **{f"sign_{bits:04b}": c for bits, c in enumerate(cells)}})
    assert len(metrics) == 2592 and len(weights) == len(signs) == 1296
    paired = []
    for fold, state, protocol, stratum in itertools.product((0, 1, 2, "pooled"), STATES, PROTOCOLS, (*STRATA, "all")):
        a, b = (groups[fold, state, view, protocol, stratum] for view in VIEWS)
        assert (a["triplets"], a["cases"]) == (b["triplets"], b["cases"])
        for i, name in enumerate(outputs):
            if stratum == "inactive":
                assert all(a[k][i] == b[k][i] for k in ("nonpositive", "hinge_positive", "margin_sum", "hinge_sum"))
            paired.append({"fold": fold, "state": state, "protocol": protocol, "plan_stratum": stratum, "output": name,
                           "triplets_per_view": a["triplets"], "original_nonpositive": a["nonpositive"][i],
                           "style_nonpositive": b["nonpositive"][i], "nonpositive_delta": b["nonpositive"][i] - a["nonpositive"][i],
                           "original_hinge_positive": a["hinge_positive"][i], "style_hinge_positive": b["hinge_positive"][i],
                           "hinge_positive_delta": b["hinge_positive"][i] - a["hinge_positive"][i],
                           "mean_hinge_delta": (b["hinge_sum"][i] - a["hinge_sum"][i]) / a["triplets"]})
    assert len(paired) == 1296
    rows_by_name = {"all_output_metrics.csv": metrics, "all_slot_weights.csv": weights, "all_slot_support_patterns.csv": signs,
                    "all_conditions.csv": conditions, "all_source_identities.csv": identity_rows, "all_paired_input_changes.csv": paired}

    args.output_dir.mkdir()
    for name, rows in rows_by_name.items():
        write_csv(args.output_dir / name, rows)
    result = {"status": "COMPLETE_ALL_SOURCE_STYLE_RELATION_REPORTS_AFTER_ARRAY_VERIFICATION",
              "created_at": datetime.now().astimezone().isoformat(), "reporter_sha256": sha(__file__),
              "execution_commit": summary["execution_commit"], "contract_sha256": summary["contract_sha256"],
              "source_summary_sha256": sha(summary_path), "array_verification_sha256": sha(proof_path),
              "all_batch_cases_sha256": sha(case_path), "original_cells": 72, "cells_including_pooled_and_all_strata": 144,
              "checked_cases": len(seen), "checked_protocol_triplet_exposures": 273297024,
              "new_model_forwards": 0, "new_optimizer_updates": 0, "image_reads": 0, "torch_imports": 0,
              "independent_audit": False, "prior_q1_status": "Q1_FAIL",
              "files": {name: {"rows": len(rows), "bytes": (args.output_dir / name).stat().st_size,
                               "sha256": sha(args.output_dir / name)} for name, rows in rows_by_name.items()},
              "pooled_conditions": [row for row in conditions if row["fold"] == "pooled"],
              "pooled_metrics": [row for row in metrics if row["fold"] == "pooled"],
              "pooled_slot_weights": [row for row in weights if row["fold"] == "pooled"],
              "pooled_slot_support": [row for row in signs if row["fold"] == "pooled"],
              "pooled_paired_changes": [row for row in paired if row["fold"] == "pooled"]}
    lines = ["# V27 来源统计扰动：完整关系支持诊断", "",
             "全部三折、固定初始化/对照终点/统计扰动终点、两种输入和两种关系协议已完成完整数组复算。",
             "这不是 Q1 检索或新训练结果；V27 原 Q1_FAIL（4/5）保持。下表分母是重复来源关系曝光，不能当作独立图片或身份。",
             "零余弦间隔非正与欧氏0.3间隔未满足分别报告；后者不是原batch-hard Triplet的同一统计。", ""]
    for protocol, stratum in itertools.product(PROTOCOLS, ("all", *STRATA)):
        lines += [f"## {protocol} / 计划{stratum}：全部模型和输入", "",
                  "| 模型状态 | 输入 | 表示 | 关系曝光 | 非正关系 | 非正% | 0.3未满足% | 平均hinge |",
                  "|---|---|---|---:|---:|---:|---:|---:|"]
        for row in result["pooled_metrics"]:
            if (row["protocol"], row["plan_stratum"]) == (protocol, stratum):
                lines.append(f"| {row['state']} | {row['view']} | {row['output']} | {row['triplets']} | {row['nonpositive']} | {row['nonpositive_percent']:.8f} | {row['hinge_positive_percent']:.8f} | {row['mean_euclidean_hinge']:.10g} |")
        lines += ["", "| 模型状态 | 输入 | 关系曝光 | V26形式辅助损失 | 同形目标间隔导数范数比（批均值） |",
                  "|---|---|---:|---:|---:|"]
        for row in result["pooled_conditions"]:
            if (row["protocol"], row["plan_stratum"]) == (protocol, stratum):
                lines.append(f"| {row['state']} | {row['view']} | {row['triplets']} | {row['auxiliary_loss']:.10g} | {row['mean_margin_derivative_ratio_per_batch']:.10g} |")
        lines.append("")
    lines += ["## 完整输出和解释边界", "",
              "全部72个原始条件、各折及三折汇总、active/inactive及其完整合计均保存，CSV包含零计数；没有筛选有利batch或身份。",
              "all_source_identities.csv保留72×94=6768行，包括没有该协议正例的来源身份；其关系分母0，百分比为空而不是伪造0%性能。",
              "槽位支持表保留全部16格：bit8/4/2/1依次表示Signal、纯银行、同角色纯残差、fused间隔非正。计数单位为槽位-关系曝光，跨槽位可能重复。",
              "间隔导数比只比较责任加权与未加权同形目标，未经过encoder Jacobian；不能称为模型参数梯度、总损失梯度或已证明的梯度冲突。",
              "固定eval状态与原训练逐步变化的参数/BN状态不同。原V27只保存前8批像素摘要，不能声称逐像素复现全部优化轨迹。",
              "本诊断只有批内候选，不能证明完整来源图库的实例难例覆盖、缓存慢漂移、未知身份泛化或官方测试提升。",
              "报告只整理预先固定的完整统计，不增加责任loss、记忆库、参数更新或调参晋级。", "",
              f"源摘要SHA256：{result['source_summary_sha256']}；完整数组核验SHA256：{result['array_verification_sha256']}。", ""]
    (args.output_dir / "complete_comparison.md").write_bytes("\n".join(lines).encode())
    result["files"]["complete_comparison.md"] = {"bytes": (args.output_dir / "complete_comparison.md").stat().st_size,
                                                "sha256": sha(args.output_dir / "complete_comparison.md")}
    (args.output_dir / "report.json").write_bytes((json.dumps(result, indent=2) + "\n").encode())
    print(json.dumps({k: result[k] for k in ("status", "checked_cases", "checked_protocol_triplet_exposures", "original_cells", "new_model_forwards")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    report(parser.parse_args())
