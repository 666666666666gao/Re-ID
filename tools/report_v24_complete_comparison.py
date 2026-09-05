#!/usr/bin/env python3
"""Render the complete V24 comparison from terminal, numerically audited arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ENDPOINTS = ("ordinary_two_view", "environment_identity_prototype")
OUTPUTS = ("baseline_only", "fused", "cnn", "transformer", "mamba")
METRICS = ("mAP", "Rank-1", "Rank-5", "Rank-10")


def build_comparison(summary, audited):
    assert summary["status"] in ("Q1_PASS", "Q1_FAIL")
    assert len(summary["folds"]) == 3 and audited["verification_passed"]
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
    return {"evaluation_type": summary["evaluation_type"], "scientific_status": summary["status"],
            "execution_source_commit": summary["repository_commit"],
            "elapsed_seconds": summary["elapsed_seconds"], "aggregate": summary["aggregate"],
            "matched_gains_mAP": summary["matched_gains_mAP"],
            "fold_fused_gains_mAP": summary["fold_fused_gains_mAP"],
            "scientific_checks": summary["scientific_checks"], "bootstrap": summary["bootstrap"],
            "fold_metric_rows": fold_rows, "all_identity_rows": identity_rows,
            "all_query_paired_changes": audited["all_query_paired_changes"],
            "all_checkpoint_training_bindings": checkpoints,
            "scope": {"gallery_records": 3126, "eligible_queries": 571,
                      "query_identities": 21, "excluded_only_from_query": 2555,
                      "optimizer_steps": audited["optimizer_steps_recounted"],
                      "dev_access_count": summary["dev_access_count"],
                      "official_test_access_count": summary["official_test_access_count"],
                      "d1_executed": summary["d1_executed"]}}


def render_markdown(report, summary_path, audit_path, output_json):
    lines = [
        "# V24 双视图下 source 身份/相机原型监督完整三折终态（2026-09-06）", "",
        f"三折两端各20epoch、3360优化步全部完成，seed42，状态 **{report['scientific_status']}**。",
        f"相对本次实际匹配对照，fused mAP增益为{report['matched_gains_mAP']['fused']:+.6f}个百分点；",
        "三折fused增益为" + "/".join(f"{value:+.6f}" for value in report["fold_fused_gains_mAP"]) + "。",
        f"21身份聚类bootstrap10000次的95%下界为{report['bootstrap']['lower_bound_95_mAP']:+.6f} mAP。", "",
        "这是复用的训练内complete-path身份OOF开发资格，非独立dev或官方结果。",
        "全部141 heldout身份、3126 gallery记录保留；571合法query来自21跨camera身份。",
        "2555条仅排除于query分母，仍留在gallery。只评价每端最终20epoch checkpoint。",
        "所有表格包含完整范围；mAP与Rank-k为百分数，增益为百分点。", "",
        "## 完整五路汇总", "",
        "| 输出 | 对照mAP | 候选mAP | 增益 | 对照R1 | 候选R1 | 对照R5 | 候选R5 | 对照R10 | 候选R10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in OUTPUTS:
        a, b = [report["aggregate"][endpoint][name] for endpoint in ENDPOINTS]
        values = [a["mAP"], b["mAP"], b["mAP"] - a["mAP"],
                  a["Rank-1"], b["Rank-1"], a["Rank-5"], b["Rank-5"],
                  a["Rank-10"], b["Rank-10"]]
        lines.append("| " + name + " | " + " | ".join(f"{v:.6f}" for v in values) + " |")
    lines += ["", "## 全三折、两端、五路", "",
              "| 折 | gallery | query | 输出 | 端点 | mAP | R1 | R5 | R10 |",
              "|---|---:|---:|---|---|---:|---:|---:|---:|"]
    for row in report["fold_metric_rows"]:
        values = " | ".join(f"{row[key]:.6f}" for key in METRICS)
        lines.append(f"| {row['fold']} | {row['gallery']} | {row['queries']} | {row['output']} | {row['endpoint']} | {values} |")
    lines += ["", "## 全部21身份和全部查询变化", "",
              "身份原名从原gallery文件名取得，编码身份亦保存在配套JSON；保留全部负收益。",
              "JSON还包含每身份两端五路的mAP与Rank1/5/10，以下列出全部五路mAP差。", "",
              "| 折 | 原始身份 | query | baseline增益 | fused增益 | CNN增益 | Transformer增益 | Mamba增益 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["all_identity_rows"]:
        values = " | ".join(f"{row['gains_percentage_points'][name]['mAP']:+.6f}" for name in OUTPUTS)
        lines.append(f"| {row['fold']} | {row['original_identity']} | {row['queries']} | {values} |")
    lines += ["", "| 输出 | AP改善 | AP下降 | AP相等 | R1修复 | R1新增错误 |",
              "|---|---:|---:|---:|---:|---:|"]
    for name in OUTPUTS:
        row = report["all_query_paired_changes"][name]
        values = " | ".join(str(row[key]) for key in
                            ("ap_improved", "ap_declined", "ap_equal", "rank1_repaired", "rank1_broken"))
        lines.append(f"| {name} | {values} |")
    lines += ["", "## 固定科学门与训练合同", "",
              "| 科学条件 | 判定 |", "|---|---|"]
    for name, passed in report["scientific_checks"].items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines += ["", "两端总参数98,800,141；可训练参数均7,841,292、203张量；新增推理参数0。",
              "两端都使用相同弱/强双视图、相同原采样器、原七组ID/Triplet监督，并计算及更新相同结构的source原型。",
              "ordinary_two_view的原型损失系数为0；environment_identity_prototype为1；其余注册合同相同。",
              "全局身份原型按真实相机均衡；强视图接受全局与同相机身份CE，只有弱视图按身份/相机分组更新EMA。",
              "原型每折/端/阶段重置，只使用该折全部94个source身份的108个真实身份/相机对，维度7680。",
              "原型在模型之外，仅用于训练；保存模型后新建模型strict reload，最终检索不读取原型文件。",
              "完整Signal及共享CLIP尾部冻结，CNN/T/M角色模块和全部原分类头继续训练，原五路等能量拼接保留。",
              "配套核验覆盖所有120个epoch的两视图14项损失均值、原型项及加权总项；Q1未保存每一次更新的完整损失行。",
              "全部六端的整段采样顺序SHA、20个epoch的108原型年龄和更新计数，与纯标签采样重放逐项绑定。",
              "每端原始分量和完整原型年龄/计数保存在配套JSON；不能把陈旧度计数当作特征慢漂移的证明。",
              "物理batch采样器没有改变，新增的是batch外原型竞争和真实跨相机正关系，不声称batch正对比例提高。",
              "这是相同双视图条件下新增原型监督的配对效应，不自动等同于相对旧单视图训练的总收益。",
              "每端从对应V12 source初始化，固定20epoch终态；无跨fold特征坐标比较，无dev或官方训练特征。",
              "本报告不单凭训练损失或工程门判定检索有效，不把训练内资格与固定dev/官方成绩混用。",
              f"运行耗时{report['elapsed_seconds']:.6f}秒；本次D1/dev/official访问均为0。", "",
              "## 证据与核验范围", "",
              f"- 原始完整汇总：{summary_path.as_posix()}；SHA {report['source_summary_sha256']}。",
              f"- 全数组算术核验：{audit_path.as_posix()}；SHA {report['array_audit_sha256']}。",
              f"- 完整比较和训练绑定：{output_json.as_posix()}。",
              f"- 实际执行源码：{report['execution_source_commit']}。",
              "- 本生成器仅处理已完成且经数值核验的JSON；不加载模型、权重张量或图像，不生成检索距离。",
              "- 远端文件SHA核验和独立终态审计须以各自完成的原始报告为准。",
              "- M0独立审计不覆盖完整Q1。此处生成数值表，不替代随后独立终态审计。", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--array-audit", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    raw, audit_raw = args.summary.read_bytes(), args.array_audit.read_bytes()
    summary, audited = json.loads(raw), json.loads(audit_raw)
    source_sha = hashlib.sha256(raw).hexdigest()
    assert audited["input_summary_sha256"] == source_sha
    report = build_comparison(summary, audited)
    report.update({"source_summary_sha256": source_sha,
                   "array_audit_sha256": hashlib.sha256(audit_raw).hexdigest(),
                   "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    args.output_json.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    args.output_md.write_bytes(render_markdown(report, args.summary, args.array_audit, args.output_json).encode("utf-8"))
    print(json.dumps({"scientific_status": report["scientific_status"],
                      "fold_metric_rows": len(report["fold_metric_rows"]),
                      "identities": len(report["all_identity_rows"]),
                      "endpoints": len(report["all_checkpoint_training_bindings"])}))
