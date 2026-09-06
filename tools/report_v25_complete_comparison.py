#!/usr/bin/env python3
"""Render all V25 folds, identities and queries after complete terminal verification."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

ENDPOINTS = ("control", "two_cross_camera")
OUTPUTS = ("baseline_only", "fused", "cnn", "transformer", "mamba")
METRICS = ("mAP", "Rank-1", "Rank-5", "Rank-10")


def build_comparison(summary, audited):
    assert summary["status"] in ("Q1_PASS", "Q1_FAIL")
    assert len(summary["folds"]) == 3 and audited["status"] == "PASS_COMPLETE_V25_FILES_TRAINING_ARRAYS_RANKINGS_AND_SCORES"
    assert summary["scientific_checks"] == audited["scientific_checks"]
    fold_rows, identity_rows, checkpoints = [], [], []
    for fold in summary["folds"]:
        gallery = fold["gallery_manifest"]
        ends = fold["endpoints"]
        query = ends[ENDPOINTS[0]]["outputs"]["fused"]["query_indices"]
        ids = np.array([gallery[index]["identity"] for index in query])
        for name in OUTPUTS:
            for endpoint in ENDPOINTS:
                result = ends[endpoint]["outputs"][name]
                assert result["query_indices"] == query
                fold_rows.append({"fold": fold["fold"], "gallery": len(gallery),
                                  "queries": len(query), "output": name, "endpoint": endpoint,
                                  **result["metrics_percent"]})
        for identity in np.unique(ids):
            positions = np.flatnonzero(ids == identity)
            original_ids = {gallery[query[i]]["file"].split("_", 1)[0] for i in positions}
            assert len(original_ids) == 1
            metrics, gains = {}, {}
            for name in OUTPUTS:
                metrics[name] = {}
                for endpoint in ENDPOINTS:
                    result = ends[endpoint]["outputs"][name]
                    ap = np.asarray(result["average_precision"], dtype=np.float64)[positions]
                    rank = np.asarray(result["first_match_rank"])[positions]
                    metrics[name][endpoint] = {"mAP": float(ap.mean() * 100),
                                              **{f"Rank-{k}": float(np.mean(rank <= k) * 100)
                                                 for k in (1, 5, 10)}}
                gains[name] = {metric: metrics[name][ENDPOINTS[1]][metric]
                              - metrics[name][ENDPOINTS[0]][metric] for metric in METRICS}
            identity_rows.append({"fold": fold["fold"], "encoded_identity": int(identity),
                                  "original_identity": original_ids.pop(), "queries": len(positions),
                                  "metrics_percent": metrics, "gains_percentage_points": gains})
        for endpoint in ENDPOINTS:
            receipt = ends[endpoint]
            checkpoints.append({"fold": fold["fold"], "endpoint": endpoint,
                                **{key: receipt[key] for key in
                                   ("checkpoint", "checkpoint_sha256", "binding", "strict_reload",
                                    "read_only_evaluation")},
                                "training": receipt["training"]})
    assert len(fold_rows) == 30 and len(identity_rows) == 21 and len(checkpoints) == 6
    assert sum(row["queries"] for row in identity_rows) == 571
    changes = {}
    assert len(audited["all_queries"]) == 571
    for name in OUTPUTS:
        rows = [row["outputs"][name] for row in audited["all_queries"]]
        differences = [row["candidate_ap"] - row["control_ap"] for row in rows]
        changes[name] = {
            "ap_improved": sum(value > 1e-12 for value in differences),
            "ap_declined": sum(value < -1e-12 for value in differences),
            "ap_equal": sum(abs(value) <= 1e-12 for value in differences),
            "rank1_repaired": sum(row["control_first_match_rank"] > 1 and row["candidate_first_match_rank"] == 1 for row in rows),
            "rank1_broken": sum(row["control_first_match_rank"] == 1 and row["candidate_first_match_rank"] > 1 for row in rows),
        }
        assert sum(changes[name][key] for key in ("ap_improved", "ap_declined", "ap_equal")) == 571
    return {"evaluation_type": summary["evaluation_type"], "scientific_status": summary["status"],
            "execution_source_commit": summary["repository_commit"],
            "elapsed_seconds": summary["elapsed_seconds"], "aggregate": summary["aggregate"],
            "matched_gains_mAP": summary["matched_gains_mAP"],
            "fold_fused_gains_mAP": summary["fold_fused_gains_mAP"],
            "scientific_checks": summary["scientific_checks"], "bootstrap": summary["bootstrap"],
            "fold_metric_rows": fold_rows, "all_identity_rows": identity_rows,
            "all_query_paired_changes": changes,
            "all_query_rows": audited["all_queries"],
            "all_checkpoint_training_bindings": checkpoints,
            "scope": {"gallery_records": 3126, "eligible_queries": 571,
                      "query_identities": 21, "excluded_only_from_query": 2555,
                      "optimizer_steps": audited["checked_training_steps"],
                      "dev_access_count": summary["dev_access_count"],
                      "official_test_access_count": summary["official_test_access_count"],
                      "d1_executed": summary["d1_executed"]}}


def write_csv(path, rows):
    assert rows
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(report):
    lines = [
        "# V25 真实跨摄像头采样：完整三折六端比较", "",
        f"生成时间：{report['generated_at']}。科学状态 **{report['scientific_status']}**。",
        "六端固定20epoch、3360优化步完整完成，全部原始数组和排序经CPU全量复算。",
        f"fused mAP相对本轮control增益{report['matched_gains_mAP']['fused']:+.6f}pp；"
        + "三fold增益" + "/".join(f"{v:+.6f}" for v in report["fold_fused_gains_mAP"]) + "pp。",
        f"全21身份、10000次bootstrap的95%下界{report['bootstrap']['lower_bound_95_mAP']:+.6f}pp。", "",
        "本次为重复使用的训练内部身份隔离Q1开发资格，不是固定30-dev或官方测试。",
        "完整141个heldout身份的3126条图库记录保留；571条合法query来自21个跨摄像头身份。",
        "其余2555条只从query分母排除，仍留作图库干扰。仅评分固定epoch20，未选择中间终点。", "",
        "## 全部五路指标", "",
        "| 输出 | control mAP | candidate mAP | mAP增益 | control R1 | candidate R1 | control R5 | candidate R5 | control R10 | candidate R10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in OUTPUTS:
        a, b = [report["aggregate"][endpoint][name] for endpoint in ENDPOINTS]
        values = [a["mAP"], b["mAP"], b["mAP"] - a["mAP"],
                  a["Rank-1"], b["Rank-1"], a["Rank-5"], b["Rank-5"], a["Rank-10"], b["Rank-10"]]
        lines.append("| " + name + " | " + " | ".join(f"{v:.6f}" for v in values) + " |")
    lines += ["", "## 三折两端五路完整结果", "",
              "| fold | gallery | query | 输出 | 端点 | mAP | R1 | R5 | R10 |",
              "|---|---:|---:|---|---|---:|---:|---:|---:|"]
    for row in report["fold_metric_rows"]:
        values = " | ".join(f"{row[key]:.6f}" for key in METRICS)
        lines.append(f"| {row['fold']} | {row['gallery']} | {row['queries']} | {row['output']} | {row['endpoint']} | {values} |")
    lines += ["", "## 全部21个身份", "",
              "身份原名取自原gallery文件名，并保留loader编码。配套CSV包含每身份每输出的全部mAP和Rank-1/5/10。",
              "以下列出全部身份的mAP增益，没有省略下降身份。", "",
              "| fold | 原始身份 | query | Signal增益 | fused增益 | CNN增益 | Transformer增益 | Mamba增益 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["all_identity_rows"]:
        values = " | ".join(f"{row['gains_percentage_points'][name]['mAP']:+.6f}" for name in OUTPUTS)
        lines.append(f"| {row['fold']} | {row['original_identity']} | {row['queries']} | {values} |")
    lines += ["", "## 全部query的变化", "",
              "AP改善/下降/相等计数采用AP单位1e-12容差，仅用于计数；指标和科学门保持原全精度。",
              "逐query CSV包含全部571×5=2855行及候选首位图库记录，JSON也保留全部原始比较。", "",
              "| 输出 | AP改善 | AP下降 | AP相等 | R1修复 | R1新增错误 |",
              "|---|---:|---:|---:|---:|---:|"]
    for name in OUTPUTS:
        row = report["all_query_paired_changes"][name]
        values = " | ".join(str(row[key]) for key in (
            "ap_improved", "ap_declined", "ap_equal", "rank1_repaired", "rank1_broken"
        ))
        lines.append(f"| {name} | {values} |")
    lines += ["", "## 原科学条件", "", "| 条件 | 判定 |", "|---|---|"]
    for name, passed in report["scientific_checks"].items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines += ["", "## 实际训练与成本", "",
              "两端均为原Signal加V8三角色，参数98,800,141/可训练7,841,292/203张量；新增推理参数0。",
              "相同V12合法起点、原七组ID/Triplet、单视图共享几何、B64/K8、seed42、固定20epoch。",
              "唯一干预是物理采样：candidate每批两组跨摄像头身份和六组单摄像头身份，并对跨相机身份建立两份原分组。",
              "两端各自所有实际batch与预先冻结采样记录逐项一致，全部source图像曝光次数也相符。",
              "候选改变了跨摄像头正关系和身份曝光分配，同时增加重复采样；不能将效果单独归因于相机风格去除。",
              "未加入V23模态MLP、V24原型/双视图、MCNL、Router或新融合；既有稠密三角色推理保持。",
              "每端从合法原始模型重新构造；M0权重没有用于Q1；最终checkpoint严格重载后才检索。", "",
              "| fold | endpoint | 更新 | 样本曝光 | 跨相机正对% | reserved MiB | 训练秒数 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in report["all_checkpoint_training_bindings"]:
        t = row["training"]
        lines.append(f"| {row['fold']} | {row['endpoint']} | {t['optimizer_steps']} | {t['sample_exposures']} | "
                     f"{t['cross_camera_positive_pair_fraction']*100:.6f} | {t['peak_reserved_mib']:.3f} | {t['elapsed_seconds']:.3f} |")
    lines += ["", f"原训练过程总耗时{report['elapsed_seconds']:.3f}秒；包括M0和终点评价。",
              "没有跨fold特征距离、dev/official访问、reranking、测试时训练或预成功消融。", "",
              "## 完整证据与边界", "",
              f"- 训练执行commit：{report['execution_source_commit']}。",
              f"- 原终态摘要SHA256：{report['source_summary_sha256']}。",
              f"- 全量CPU核验SHA256：{report['array_audit_sha256']}。",
              f"- 报告生成器SHA256：{report['generator_sha256']}。",
              "- 6个完整终点权重及全部特征/距离/排名数组留在服务器；本地仅处理JSON、CSV与文本。",
              "- 工程核验PASS不代表科学Q1_PASS。外部独立审计额度不可用，执行器核验不得标为独立审计。",
              "- 本次21个身份已反复用于开发，bootstrap不能消除选择偏差。",
              "- 保留原五项科学门及任何负结果；不得按本次结果扫描采样配比、seed、epoch或新loss。", ""]
    return "\n".join(lines)


def main(args):
    for path in (args.output_json, args.output_md, args.query_csv, args.identity_csv):
        assert not path.exists(), path
    raw, audit_raw = args.summary.read_bytes(), args.verification.read_bytes()
    summary, audited = json.loads(raw), json.loads(audit_raw)
    source_sha = hashlib.sha256(raw).hexdigest()
    assert audited["run_summary_sha256"] == source_sha
    assert audited["verifier_sha256"] == "cc1cf0c1db52dc3b9bf770cd9cdc427d9142f751b62afda78bb5a73fa4e5a277"
    assert summary["repository_commit"] == "97468dd01bcef55d90455b35b1373aa03a2e335f"
    report = build_comparison(summary, audited)
    report.update({
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_summary_sha256": source_sha,
        "array_audit_sha256": hashlib.sha256(audit_raw).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    identities = []
    for row in report["all_identity_rows"]:
        for name in OUTPUTS:
            entry = {"fold": row["fold"], "original_identity": row["original_identity"],
                     "encoded_identity": row["encoded_identity"], "queries": row["queries"], "output": name}
            for metric in METRICS:
                entry["control_" + metric] = row["metrics_percent"][name]["control"][metric]
                entry["candidate_" + metric] = row["metrics_percent"][name]["two_cross_camera"][metric]
                entry["gain_" + metric + "_pp"] = row["gains_percentage_points"][name][metric]
            identities.append(entry)
    queries = []
    for row in report["all_query_rows"]:
        for name in OUTPUTS:
            item = row["outputs"][name]
            top = item["candidate_first_gallery"]
            queries.append({
                "fold": row["fold"], "query_index": row["query_index"],
                "original_identity": row["file"].split("_", 1)[0],
                "encoded_identity": row["identity"], "file": row["file"], "camera": row["camera"],
                "output": name, "control_ap_percent": item["control_ap"] * 100,
                "candidate_ap_percent": item["candidate_ap"] * 100, "ap_gain_pp": item["ap_gain_pp"],
                "control_first_match_rank": item["control_first_match_rank"],
                "candidate_first_match_rank": item["candidate_first_match_rank"],
                "rank1_repaired": item["control_first_match_rank"] > 1 and item["candidate_first_match_rank"] == 1,
                "rank1_new_error": item["control_first_match_rank"] == 1 and item["candidate_first_match_rank"] > 1,
                "candidate_first_gallery_index": item["candidate_first_gallery_index"],
                "candidate_first_gallery_file": top["file"],
                "candidate_first_gallery_identity": top["identity"],
                "candidate_first_gallery_camera": top["camera"],
            })
    assert len(identities) == 105 and len(queries) == 2855
    write_csv(args.identity_csv, identities)
    write_csv(args.query_csv, queries)
    report["identity_csv"] = {"path": args.identity_csv.as_posix(), "rows": 105,
                            "sha256": hashlib.sha256(args.identity_csv.read_bytes()).hexdigest()}
    report["query_csv"] = {"path": args.query_csv.as_posix(), "rows": 2855,
                         "sha256": hashlib.sha256(args.query_csv.read_bytes()).hexdigest()}
    args.output_json.write_bytes((json.dumps(report, indent=2) + "\n").encode())
    args.output_md.write_bytes(render(report).encode())
    print(json.dumps({"scientific_status": report["scientific_status"],
                      "fold_metric_rows": len(report["fold_metric_rows"]), "identity_csv_rows": 105,
                      "query_csv_rows": 2855, "endpoints": len(report["all_checkpoint_training_bindings"]),
                      "summary_sha256": source_sha}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--query-csv", type=Path, required=True)
    parser.add_argument("--identity-csv", type=Path, required=True)
    main(parser.parse_args())
