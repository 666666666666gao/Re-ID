#!/usr/bin/env python3
"""Verify all decomposition text and report every representation and sign cell."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    for path in (args.output_json, args.output_md, args.metrics_csv, args.sign_csv):
        assert not path.exists(), path
    raw = args.summary.read_bytes()
    summary = json.loads(raw)
    assert summary["status"] == "COMPLETE_ALL_SOURCE_SUPPORT_DECOMPOSITION"
    assert len(summary["folds"]) == 3 and summary["all_query_rows"] == 25008
    count, relations = 0, 0
    metrics, signs = [], []
    for fold in summary["folds"]:
        assert len(fold["comparisons"]) == 4
        for comparison in fold["comparisons"]:
            descriptor = comparison["all_query_rows"]
            row_path = args.row_dir / Path(descriptor["path"]).name
            assert sha(row_path) == descriptor["sha256"]
            aggregates = {name: {"ap_sum": 0., "rank1_hits": 0, "nonpositive_pairs": 0, "queries_ap_below_one": 0} for name in comparison["outputs"]}
            counts = {name: [0] * 16 for name in comparison["slot_error_joint_sign_counts"]}
            eligible = query_count = pairs = 0
            with gzip.open(row_path, "rt", encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    row = json.loads(line)
                    assert (row["fold"], row["view"], row["protocol"], row["query_index"]) == (fold["fold"], comparison["view"], comparison["protocol"], index)
                    query_count += 1
                    if not row["eligible"]:
                        continue
                    eligible += 1
                    pairs += row["positive_records"] * row["negative_records"]
                    assert set(row["outputs"]) == set(aggregates)
                    assert set(row["slot_error_joint_sign_counts"]) == set(counts)
                    for name, value in row["outputs"].items():
                        aggregates[name]["ap_sum"] += value["average_precision"]
                        aggregates[name]["rank1_hits"] += value["first_match_rank"] == 1
                        aggregates[name]["nonpositive_pairs"] += value["nonpositive_pairs"]
                        aggregates[name]["queries_ap_below_one"] += value["average_precision"] < 1 - 1e-12
                    for name, cells in row["slot_error_joint_sign_counts"].items():
                        assert len(cells) == 16 and sum(cells) == row["outputs"][name]["nonpositive_pairs"]
                        counts[name] = [a + b for a, b in zip(counts[name], cells)]
            assert query_count == descriptor["rows"] == comparison["all_queries"]
            assert eligible == comparison["eligible_queries"] and pairs == comparison["all_positive_negative_relations"]
            count += query_count
            relations += pairs
            common = {"fold": fold["fold"], "view": comparison["view"], "protocol": comparison["protocol"], "eligible_queries": eligible}
            for name, values in aggregates.items():
                for key, value in values.items():
                    assert abs(value - comparison["outputs"][name][key]) < 1e-9
                values.update(source_mAP_percent=values["ap_sum"] / eligible * 100,
                              source_Rank1_percent=values["rank1_hits"] / eligible * 100)
                for key in ("source_mAP_percent", "source_Rank1_percent"):
                    assert abs(values[key] - comparison["outputs"][name][key]) < 1e-9
                metrics.append({**common, "output": name, **values})
            assert counts == comparison["slot_error_joint_sign_counts"]
            for name, cells in counts.items():
                signs.append({**common, "slot": name, **{f"sign_{i:04b}": value for i, value in enumerate(cells)}})
    assert count == 25008 and relations == 634354228 and len(metrics) == 216 and len(signs) == 108
    for path, rows in ((args.metrics_csv, metrics), (args.sign_csv, signs)):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    combined, combined_signs = [], []
    for view in ("clean", "augmented"):
        for protocol in ("identity_exclude_record", "cross_camera"):
            names = list(summary["folds"][0]["comparisons"][0]["outputs"])
            for name in names:
                rows = [r for r in metrics if (r["view"], r["protocol"], r["output"]) == (view, protocol, name)]
                assert len(rows) == 3
                n = sum(r["eligible_queries"] for r in rows)
                combined.append({"view": view, "protocol": protocol, "output": name, "eligible_queries": n,
                                 "source_mAP_percent": sum(r["ap_sum"] for r in rows) / n * 100,
                                 "source_Rank1_percent": sum(r["rank1_hits"] for r in rows) / n * 100,
                                 "nonpositive_pairs": sum(r["nonpositive_pairs"] for r in rows),
                                 "queries_ap_below_one": sum(r["queries_ap_below_one"] for r in rows)})
            for name in summary["folds"][0]["comparisons"][0]["slot_error_joint_sign_counts"]:
                rows = [r for r in signs if (r["view"], r["protocol"], r["slot"]) == (view, protocol, name)]
                cells = [sum(r[f"sign_{i:04b}"] for r in rows) for i in range(16)]
                combined_signs.append({"view": view, "protocol": protocol, "slot": name, "all_16_sign_counts": cells,
                                       "slot_nonpositive": sum(cells),
                                       "same_role_positive_and_fused_positive": sum(v for i, v in enumerate(cells) if not (i & 3)),
                                       "bank_positive_and_fused_positive": sum(v for i, v in enumerate(cells) if not (i & 5)),
                                       "bank_nonpositive_but_fused_positive": sum(v for i, v in enumerate(cells) if i & 4 and not i & 1),
                                       "role_nonpositive_but_bank_and_fused_positive": sum(v for i, v in enumerate(cells) if i & 2 and not i & 5),
                                       "fused_nonpositive": sum(v for i, v in enumerate(cells) if i & 1)})
    result = {"status": "PASS_ALL25008_QUERY_ROWS_216_METRICS_108_SIGN_TABLES", "summary_sha256": sha(args.summary),
              "generator_sha256": sha(__file__), "execution_commit": summary["execution_commit"],
              "all_query_rows_checked": count, "all_positive_negative_relations": relations,
              "combined_outputs": combined, "combined_slot_signs": combined_signs,
              "all_fold_output_metrics": metrics, "all_fold_slot_signs": signs,
              "numerical_checks": [r for fold in summary["folds"] for r in fold["numerical_checks"]],
              "model_forwards": 0, "image_reads": 0, "optimizer_updates": 0, "independent_audit": False}
    lines = ["# 固定来源特征：完整检索支持分解（2026-09-07）", "",
             "三折全部来源特征、干净/增强query、两合法协议均完成。没有图像读取、模型干预或新训练。",
             "原14表示的每query AP/R1/非正关系与此前全量核验结果一致，新增4种表示完整报告。",
             "25008个完整query行、216个表示统计、108组16格符号表全部从压缩文本重汇总通过。",
             "纯银行由原fused后4608D归一化得到；纯角色由完整角色向量后1536D归一化得到。",
             "所有统计仅描述已见来源身份和固定特征，不能当未知身份效果、训练消融或唯一因果解释。", ""]
    for view in ("clean", "augmented"):
        for protocol in ("identity_exclude_record", "cross_camera"):
            lines += [f"## {view} / {protocol}：完整三折汇总", "",
                      "| 表示 | 合法query | source mAP | source R1 | AP不满query | 非正关系 |",
                      "|---|---:|---:|---:|---:|---:|"]
            for row in combined:
                if (row["view"], row["protocol"]) == (view, protocol):
                    lines.append(f"| {row['output']} | {row['eligible_queries']} | {row['source_mAP_percent']:.6f} | {row['source_Rank1_percent']:.6f} | {row['queries_ap_below_one']} | {row['nonpositive_pairs']} |")
            lines += ["", "下列计数有重叠；完整16格无遗漏组合见CSV及JSON。单位为槽位—正负关系曝光，不是独立图片。", "",
                      "| 槽位 | 槽位非正 | 同角色正且融合正 | 银行正且融合正 | 角色非正而银行/融合正 | 银行非正但融合正 | 融合仍非正 |",
                      "|---|---:|---:|---:|---:|---:|---:|"]
            for row in combined_signs:
                if (row["view"], row["protocol"]) == (view, protocol):
                    values = " | ".join(str(row[k]) for k in ("slot_nonpositive", "same_role_positive_and_fused_positive", "bank_positive_and_fused_positive", "role_nonpositive_but_bank_and_fused_positive", "bank_nonpositive_but_fused_positive", "fused_nonpositive"))
                    lines.append(f"| {row['slot']} | {values} |")
            lines.append("")
    lines += ["## 完整记录与边界", "",
              "每个符号格bit8/4/2/1分别表示Signal/纯银行/同角色纯残差/fused非正。",
              "例如角色正、银行非正、融合正的组合也完整保留，不能被分层归因删掉。",
              "所有相似度取实际保存特征；近似等能量/槽位平均误差另有完整数值记录。",
              "固定表示项移除的分数比较不等于重新训练后的模型性能，不确定唯一的学习原因。",
              "没有新模型权重或特征文件；没有更新任何失败版本的损失或门槛。",
              f"执行commit {summary['execution_commit']}，摘要SHA {result['summary_sha256']}。",
              "全部三折216行metrics和108组sign表已另存CSV，所有query保存为12份gzip JSONL。", ""]
    args.output_json.write_bytes((json.dumps(result, indent=2) + "\n").encode())
    args.output_md.write_bytes("\n".join(lines).encode())
    print(json.dumps({"status": result["status"], "rows": count, "relations": relations}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--row-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--sign-csv", type=Path, required=True)
    main(parser.parse_args())
