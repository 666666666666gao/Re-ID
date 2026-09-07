#!/usr/bin/env python3
"""Export all verified V29 source drift scalars; no model or retrieval evaluation."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.v29_source_drift_math import (
    CHANGE_FIELDS, COMPARISONS, OUTPUTS, PROTOCOLS, STATES, VIEWS,
)

STABLE_FIELDS = (
    "reliable_original_and_style_count", "stable_lost_correct_count",
    "stable_hinge_violated_count",
)
ARRAY_SHAPES = {
    "changes": (3, 2, 18, 12), "stable": (3, 2, 18, 3),
    "joint_changes": (2, 12), "joint_stable": (2, 3),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_bytes())


def ratio(value, denominator):
    # Zero eligible/reliable relations are part of this registered source protocol.
    return float(value / denominator) if denominator else None


def measurements(changes, stable, triplets):
    result = {}
    for name, value in zip(CHANGE_FIELDS, changes, strict=True):
        if name.endswith("_sum"):
            result[name.removesuffix("_sum") + "_mean"] = ratio(value, triplets)
        else:
            assert value == int(value) and 0 <= value <= triplets
            result[name] = int(value)
    for name, value in zip(STABLE_FIELDS, stable, strict=True):
        assert value == int(value) and 0 <= value <= triplets
        result[name] = int(value)
    reliable = result[STABLE_FIELDS[0]]
    assert result[STABLE_FIELDS[1]] <= reliable and result[STABLE_FIELDS[2]] <= reliable
    result["stable_lost_correct_rate"] = ratio(stable[1], reliable)
    result["stable_hinge_violated_rate"] = ratio(stable[2], reliable)
    return result


def expand_relations(groups, identity=False):
    for group in groups:
        keys = ("fold", "protocol", "identity", "encoded_identity") if identity else (
            "fold", "protocol", "stratum")
        base = {key: group[key] for key in keys}
        base.update(cases=group["cases"], triplets=group["triplets"])
        for ci, comparison in enumerate(COMPARISONS):
            for vi, view in enumerate(VIEWS):
                for oi, output in enumerate(OUTPUTS):
                    yield {**base, "comparison": comparison, "view": view, "output": output,
                           **measurements(group["changes"][ci, vi, oi],
                                          group["stable"][ci, vi, oi], group["triplets"])}


def expand_joint(groups, identity=False):
    for group in groups:
        keys = ("fold", "protocol", "identity", "encoded_identity") if identity else (
            "fold", "protocol", "stratum")
        base = {key: group[key] for key in keys}
        base.update(cases=group["cases"], triplets=group["triplets"])
        for vi, view in enumerate(VIEWS):
            yield {**base, "comparison": "same_forward_candidate_joint_contribution", "view": view,
                   **measurements(group["joint_changes"][vi], group["joint_stable"][vi],
                                  group["triplets"])}


def flatten_vectors(rows):
    for row in rows:
        result = {key: value for key, value in row.items() if key not in ("cosine", "chord")}
        assert row["cosine"]["count"] == row["chord"]["count"]
        for metric in ("cosine", "chord"):
            for statistic, value in row[metric].items():
                result[metric + "_" + statistic] = value
        yield result


def write_csv(path, rows, expected_rows):
    rows = iter(rows)
    first = next(rows)
    count = 1
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        for row in rows:
            writer.writerow(row)
            count += 1
    assert count == expected_rows, (path.name, count, expected_rows)
    return {"rows": count, "bytes": path.stat().st_size, "sha256": sha(path)}


def aggregate(groups):
    result = []
    for protocol in PROTOCOLS:
        selected = [g for g in groups if g["protocol"] == protocol]
        result.append({"fold": "all_separate_model_coordinates", "protocol": protocol,
                       "stratum": "all", "cases": sum(g["cases"] for g in selected),
                       "triplets": sum(g["triplets"] for g in selected),
                       **{name: sum(g[name] for g in selected) for name in ARRAY_SHAPES}})
    assert [row["triplets"] for row in result] == [42147840, 3401664]
    assert all(row["cases"] == 1680 for row in result)
    return result


def report(args):
    run = args.run_dir
    pipeline = load_json(str(run) + "_pipeline_exit.json")
    assert pipeline["stage"] == "COMPLETE_VERIFIED_SOURCE_ROLE_DRIFT" and pipeline["exit_code"] == 0
    for stage in ("math", "diagnostic", "verification"):
        assert load_json(str(run) + "_" + stage + "_exit.json")["exit_code"] == 0
    proof_path = run / "complete_source_drift_verification.json"
    proof = load_json(proof_path)
    assert proof["status"] == "PASS_COMPLETE_V29_SOURCE_ROLE_DRIFT_AND_RELATIONS"
    assert proof["checked_batches"] == 1680 and proof["checked_original_model_forwards"] == 10080
    assert proof["checked_similarity_values"] == 743178240
    assert proof["checked_vector_observations"] == 7741440
    assert proof["checked_protocol_triplets_per_state_view"] == 45549504
    assert proof["source_summary_sha256"] == sha(run / "source_role_drift.json")
    assert proof["comparisons"] == list(COMPARISONS) and proof["views"] == list(VIEWS)
    assert proof["states"] == list(STATES) and proof["outputs"] == list(OUTPUTS)
    assert proof["relation_change_fields"] == list(CHANGE_FIELDS)
    assert proof["stable_fields"] == ["reliable_original_and_style_count", "lost_correct_count", "hinge_violated_count"]
    summary = load_json(run / "source_role_drift.json")
    assert summary["status"] == "COMPLETE_FIXED_SOURCE_ROLE_DRIFT_PENDING_CPU_VERIFICATION"
    assert summary["new_optimizer_updates"] == 0
    files = (
        ("all_source_relation_groups.json", 12), ("all_source_identity_relations.json", 564),
        ("all_source_vector_groups.json", 648), ("all_source_identity_vectors.json", 20304),
    )
    data = {}
    for filename, expected_count in files:
        path = run / filename
        assert path.stat().st_size == proof["files"][filename]["bytes"]
        assert sha(path) == proof["files"][filename]["sha256"]
        data[filename] = load_json(path)
        assert len(data[filename]) == expected_count
    groups, identities = (data[name] for name, _ in files[:2])
    for group in [*groups, *identities]:
        for name, shape in ARRAY_SHAPES.items():
            group[name] = np.asarray(group[name], dtype=np.float64)
            assert group[name].shape == shape and np.isfinite(group[name]).all()
    assert len({(g["fold"], g["protocol"], g["stratum"]) for g in groups}) == 12
    assert len({(g["fold"], g["protocol"], g["encoded_identity"]) for g in identities}) == 564
    for fold in range(3):
        for protocol in PROTOCOLS:
            source = [g for g in identities if (g["fold"], g["protocol"]) == (fold, protocol)]
            target = [g for g in groups if (g["fold"], g["protocol"]) == (fold, protocol)]
            assert len(source) == 94 and len(target) == 2
            assert sum(g["triplets"] for g in source) == sum(g["triplets"] for g in target)
            for name in ARRAY_SHAPES:
                assert np.allclose(sum(g[name] for g in source), sum(g[name] for g in target),
                                   rtol=1e-10, atol=1e-8)
    totals = aggregate(groups)
    total_rows = list(expand_relations(totals))
    joint_rows = list(expand_joint(totals))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    exports = {}
    for filename, rows, count in (
        ("relation_groups.csv", expand_relations(groups), 1296),
        ("identity_relations.csv", expand_relations(identities, True), 60912),
        ("joint_groups.csv", expand_joint(groups), 24),
        ("joint_identity_relations.csv", expand_joint(identities, True), 1128),
        ("vector_groups.csv", flatten_vectors(data[files[2][0]]), 648),
        ("identity_vectors.csv", flatten_vectors(data[files[3][0]]), 20304),
        ("relation_totals.csv", total_rows, 216),
        ("joint_totals.csv", joint_rows, 4),
    ):
        exports[filename] = write_csv(args.output_dir / filename, rows, count)
    lines = [
        "# V29 完整来源角色漂移与关系变化", "",
        "这是完整固定来源诊断的标量汇总，没有新增训练或检索成绩。V29 Q1_FAIL0/5保持。", "",
        f"实际诊断提交：{summary['execution_commit']}。原始前向10080次、1680个批次，全部CPU核验已完成。",
        "identity关系42147840次，cross_camera关系3401664次；它们包含重复曝光，跨模型和输入条件复用，不是独立试验数量。",
        "原输入包含注册训练增强；registered_style额外施加既有V27计划，其中未激活批次两输入完全一致。",
        "每个模型在自身坐标系内计算相似度，只跨fold累加标量。来源身份在不同fold有重叠，不计算独立样本置信区间。", "",
        "## 候选相对匹配控制的全部18输出", "",
        "以下合计active与inactive。间隔为正负余弦差；hinge是单位向量欧氏Triplet margin0.3。",
        "可靠集合由比较起点在两输入条件都满足margin>0且hinge=0定义。单纯间隔下降不等于排序错误。", "",
        "|协议|输入|输出|起点非正|终点非正|破坏正确|修复|可靠关系|可靠正确被破坏|可靠间隔违规|平均间隔变化|",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in total_rows:
        if row["comparison"] != "control_to_bounded":
            continue
        delta = row["after_margin_mean"] - row["before_margin_mean"]
        fields = [row[k] for k in ("protocol", "view", "output", "before_nonpositive_count",
                  "after_nonpositive_count", "lost_correct_count", "repaired_count", *STABLE_FIELDS)]
        lines.append("|" + "|".join(map(str, fields)) + f"|{delta:.9g}|")
    lines += ["", "## 同一次候选前向中的联合贡献", "",
              "比较.5Signal+.5原角色h银行与实际fused；这是保存矩阵的标量分解，没有新检索头或反事实mAP。", "",
              "|协议|输入|原h分量非正|实际fused非正|破坏正确|修复|可靠关系|可靠正确被破坏|可靠间隔违规|",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in joint_rows:
        fields = [row[k] for k in ("protocol", "view", "before_nonpositive_count",
                  "after_nonpositive_count", "lost_correct_count", "repaired_count", *STABLE_FIELDS)]
        lines.append("|" + "|".join(map(str, fields)) + "|")
    lines += ["", "## 完整明细与解释边界", "",
              "所有比较和来源身份见下列CSV；空比率表示分母为0，不表示0错误率。无合法跨camera正例身份保留。",
              "向量分布保留每fold/输入/角色/模态/比较/计划的精确分位数；未平均分位数冒充总体分位数。",
              "向量旋转本身不是身份证据丢失；应结合模型内部的真实关系变化。当前h到y有界不限制初始化到最终h的漂移。",
              "本报告从已核验JSON导出，不重新计算模型特征；执行器核验不是外部独立审计。", "",
              "|文件|完整行数|字节|", "|---|---:|---:|"]
    lines.extend(f"|[{name}]({name})|{item['rows']}|{item['bytes']}|" for name, item in exports.items())
    report_path = args.output_dir / "README.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    exports[report_path.name] = {"bytes": report_path.stat().st_size, "sha256": sha(report_path)}
    receipt = {
        "status": "PASS_COMPLETE_SOURCE_DRIFT_SCALAR_EXPORT", "created_at": datetime.now().astimezone().isoformat(),
        "execution_commit": summary["execution_commit"], "source_summary_sha256": proof["source_summary_sha256"],
        "source_verification_sha256": sha(proof_path), "reporter_sha256": sha(__file__),
        "math_module_sha256": sha(Path(__file__).with_name("v29_source_drift_math.py")),
        "source_files": proof["files"], "exports": exports,
        "all_identity_relation_sums_rechecked": True,
        "zero_eligible_identity_rows": sum(g["triplets"] == 0 for g in identities),
        "new_model_forwards": 0, "optimizer_updates": 0, "image_reads": 0,
        "torch_imports": 0, "retrieval_evaluations": 0, "independent_external_review": False,
    }
    (args.output_dir / "export_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    report(parser.parse_args())
