#!/usr/bin/env python3
"""Report all source-census outputs after every saved row has been verified."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

OUTPUTS = ("baseline_only", "fused", "cnn", "transformer", "mamba",
           *(f"{e}_{m}_residual" for e in ("cnn", "transformer", "mamba")
             for m in ("RGB", "NI", "TI")))
COUNTS = ("all_query_records", "eligible_queries", "excluded_query_records_kept_in_gallery",
          "prototype_correct", "prototype_correct_but_instance_rank1_wrong",
          "prototype_correct_but_instance_ap_below_one", "negative_prototype_misses_rank1_competitor",
          "nonpositive_ordered_positive_negative_pairs", "all_ordered_positive_negative_pairs")


def main(args):
    for path in (args.output_json, args.output_md, args.output_csv):
        assert not path.exists(), path
    raw = args.summary.read_bytes()
    summary = json.loads(raw)
    proof = json.loads(args.verification.read_bytes())
    assert summary["status"] == "COMPLETE_SOURCE_ONLY_INSTANCE_PROTOTYPE_CENSUS"
    assert proof["status"] == "PASS_ALL_SOURCE_INSTANCE_CENSUS_ROWS_AND_AGGREGATES"
    assert proof["summary_sha256"] == hashlib.sha256(raw).hexdigest()
    assert proof["rows_checked"] == 350112 and len(proof["all_aggregates"]) == 168
    rows = proof["all_aggregates"]
    combined = []
    for view in ("clean", "augmented"):
        for protocol in ("identity_exclude_record", "cross_camera"):
            for name in OUTPUTS:
                selected = [r for r in rows if (r["view"], r["protocol"], r["output"]) == (view, protocol, name)]
                assert len(selected) == 3 and {r["fold"] for r in selected} == {0, 1, 2}
                counts = {key: sum(r[key] for r in selected) for key in COUNTS}
                entry = {"view": view, "protocol": protocol, "output": name, **counts,
                         **{key: sum(r[key] * r["eligible_queries"] for r in selected) / counts["eligible_queries"]
                            for key in ("source_probe_mAP_percent", "source_probe_Rank1_percent")}}
                combined.append(entry)
    assert len(combined) == 56
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report = {"status": "COMPLETE_ALL_SOURCE_CENSUS_REPORT", "execution_commit": summary["execution_commit"],
              "source_summary_sha256": hashlib.sha256(raw).hexdigest(),
              "source_verification_sha256": hashlib.sha256(args.verification.read_bytes()).hexdigest(),
              "contract_sha256": summary["contract_sha256"],
              "combined_view_protocol_outputs": combined, "all_fold_view_protocol_outputs": rows,
              "model_forward_batches": summary["total_model_forward_batches"],
              "source_record_model_pairs": summary["total_source_record_model_pairs"],
              "all_query_output_protocol_rows": proof["rows_checked"],
              "optimizer_updates": 0, "limitations": summary["limitations"],
              "feature_files": [fold["features"] for fold in summary["folds"]],
              "census_elapsed_seconds": summary["elapsed_seconds"],
              "verification_elapsed_seconds": proof["elapsed_seconds"],
              "maximum_verification_numeric_error": proof["maximum_numeric_error"],
              "all_aggregate_csv_sha256": hashlib.sha256(args.output_csv.read_bytes()).hexdigest()}
    lines = ["# 完整来源实例—原型普查（2026-09-07）", "",
             "这是来源模型在已见训练身份上的只读机制诊断。没有新训练或未知身份检索资格结果。",
             "每折使用原V12合法初始化，覆盖全部2126/2075/2051记录；共6252记录—模型配对。",
             "干净/固定增强两视图各完整一次，全部14输出、两关系协议、200前向batch。",
             "全部350112输出行和168聚合项已经CPU独立实现复算；执行器核验不是外部独立审计。", "",
             "identity_exclude_record排除本记录正例；cross_camera只接受同身份异相机正例。",
             "所有来源记录始终留在干净图库。跨相机无正例的query不计AP，但完整输出且仍能作为其他身份的负例。",
             "真身份原型仅由query的合法干净正例构造，负身份原型由各自全部干净实例构造。",
             "单实例为保存的FP32归一化特征，转float64求点积；prototype求和后归一化。", ""]
    for view in ("clean", "augmented"):
        for protocol in ("identity_exclude_record", "cross_camera"):
            lines += [f"## {view} / {protocol}：全部三折按合法query数加权", "",
                      "以下数量包含不同fold坐标内的重复来源身份与图片，不能作为独立身份样本量。", "",
                      "| 输出 | 全部query | 合法query | source mAP | source R1 | 原型正确 | 原型正确但R1错 | 原型正确但AP<1 | 负原型隐藏首位竞争 | 非正关系/全部关系 |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for row in combined:
                if (row["view"], row["protocol"]) != (view, protocol):
                    continue
                lines.append(f"| {row['output']} | {row['all_query_records']} | {row['eligible_queries']} | "
                             f"{row['source_probe_mAP_percent']:.6f} | {row['source_probe_Rank1_percent']:.6f} | "
                             f"{row['prototype_correct']} | {row['prototype_correct_but_instance_rank1_wrong']} | "
                             f"{row['prototype_correct_but_instance_ap_below_one']} | {row['negative_prototype_misses_rank1_competitor']} | "
                             f"{row['nonpositive_ordered_positive_negative_pairs']}/{row['all_ordered_positive_negative_pairs']} |")
            lines.append("")
    lines += ["## 全部三折分项", "", "| fold | view | protocol | output | 合法query | mAP | R1 | 原型正确但R1错 | 原型正确但AP<1 |",
              "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['fold']} | {row['view']} | {row['protocol']} | {row['output']} | {row['eligible_queries']} | "
                     f"{row['source_probe_mAP_percent']:.6f} | {row['source_probe_Rank1_percent']:.6f} | "
                     f"{row['prototype_correct_but_instance_rank1_wrong']} | {row['prototype_correct_but_instance_ap_below_one']} |")
    lines += ["", "## 工件与限制", "",
              f"执行commit：{summary['execution_commit']}。",
              f"完整来源摘要SHA：{report['source_summary_sha256']}。",
              f"全量核验SHA：{report['source_verification_sha256']}。",
              f"数值核验最大误差：{proof['maximum_numeric_error']:.12g}。",
              f"提取与统计{summary['elapsed_seconds']:.3f}秒，CPU核验{proof['elapsed_seconds']:.3f}秒。",
              "保存三份完整特征与12份全行记录，未保存模型权重或大型相似度矩阵。",
              "一次固定增强普查不能代表所有增强或训练模式分布，也没有测缓存年龄与特征漂移。",
              "非正关系按negative>=positive计；stable排名的同分次序按原记录索引保留。",
              "原型正确但实例误排的数量只说明存在性和范围，不证明增加记忆、XBM或排序loss能改善新身份。",
              "不得将这里的source mAP与OOF、30-dev、官方测试或论文数值直接相减。", ""]
    args.output_json.write_bytes((json.dumps(report, indent=2) + "\n").encode())
    args.output_md.write_bytes("\n".join(lines).encode())
    print(json.dumps({"status": report["status"], "complete_rows": proof["rows_checked"], "aggregate_rows": 168}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    main(parser.parse_args())
