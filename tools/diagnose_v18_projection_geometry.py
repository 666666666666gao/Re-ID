#!/usr/bin/env python3
"""Replay all six V18 final heads on bound full-gallery frozen feature caches."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
import torch.nn.functional as F
from tools.audit_v17_full_gallery import full_gallery_scores
from tools.diagnose_v17_failure_geometry import match_rows
from tools.train_signal_preserving_v17 import _sha256, _tensor_mapping_sha256
from trifusion.signal_preserving_v17 import TriadicCorrectionV17
from trifusion.signal_preserving_v18 import PairedViewProjectionV18

EXPERTS = ("cnn", "transformer", "mamba")
OUTPUTS = ("baseline_only", "fused", *EXPERTS)

def run(args):
    started = time.time()
    torch.set_num_threads(4)
    assert _sha256(args.q1_summary) == args.q1_sha256
    q1 = json.loads(args.q1_summary.read_text())
    prior = json.loads(args.v17_geometry.read_text())
    assert q1["status"] in ("Q1_PASS", "Q1_FAIL")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    folds = []
    changes = {name: [] for name in OUTPUTS}
    for fold, old in zip(q1["folds"], prior["folds"], strict=True):
        index = fold["fold"]
        assert index == old["fold"]
        manifest = fold["gallery_manifest"]
        assert manifest == [{"file": Path(r["files"][0]).name, "identity": r["identity"], "camera": r["camera"]} for r in old["gallery_manifest"]]
        cached = old["endpoints"]["weight0"]
        cache_path = Path(cached["cache"])
        assert _sha256(cache_path) == cached["cache_sha256"]
        tensors = torch.load(cache_path, map_location="cpu", weights_only=True)
        baseline = tensors["baseline"]
        teacher = {e: tensors["teacher_" + e] for e in EXPERTS}
        assert len(baseline) == len(manifest)
        directions = torch.tensor(q1["calibrations"][index]["directions"])
        assert _tensor_mapping_sha256({"directions": directions}) == q1["calibrations"][index]["directions_sha256"]
        identities = np.array([r["identity"] for r in manifest])
        cameras = np.array([r["camera"] for r in manifest])
        energy = {e: ((teacher[e] * directions[i]).sum(1).square() / teacher[e].square().sum(1)).tolist() for i,e in enumerate(EXPERTS)}
        result = {"fold": index, "gallery_manifest": manifest, "cache": str(cache_path),
                  "cache_sha256": cached["cache_sha256"], "teacher_projection_energy_fraction": energy,
                  "endpoints": {}}
        for endpoint, saved in fold["endpoints"].items():
            path = Path(saved["checkpoint"])
            assert _sha256(path) == saved["checkpoint_sha256"]
            payload = torch.load(path, map_location="cpu", weights_only=True)
            assert payload["projection_enabled"] == (endpoint == "projected")
            assert payload["plan_sha256"] == q1["plan_sha256"]
            core = PairedViewProjectionV18(TriadicCorrectionV17(residual_width=1536, adapter_width=256),
                                          directions, enabled=payload["projection_enabled"]).cuda().eval()
            state = {k.removeprefix("correction."): v for k,v in payload["v18_state_dict"].items() if k.startswith("correction.")}
            core.load_state_dict(state, strict=True)
            before = _tensor_mapping_sha256(core.state_dict())
            features = {name: [] for name in OUTPUTS}
            corrected = {e: [] for e in EXPERTS}
            delta = {e: [] for e in EXPERTS}
            def hook_for(expert):
                def capture(_module, _inputs, value):
                    delta[expert].append(value.float().cpu())
                return capture
            handles = [core.core.receiver_projections[e].register_forward_hook(hook_for(e)) for e in EXPERTS]
            with torch.inference_mode():
                for start in range(0, len(manifest), 128):
                    batch = {e: teacher[e][start:start+128].cuda() for e in EXPERTS}
                    base = baseline[start:start+128].cuda()
                    output = core(batch)
                    norm = base.norm(dim=1, keepdim=True)
                    fused = torch.cat((base, output.fused_residual * norm), dim=1)
                    assert torch.equal(fused[:, :3072], base)
                    features["baseline_only"].append(base.cpu())
                    features["fused"].append(fused.cpu())
                    for expert in EXPERTS:
                        value = output.corrected_residuals[expert]
                        features[expert].append(torch.cat((base, value * norm), dim=1).cpu())
                        corrected[expert].append(value.cpu())
            for handle in handles:
                handle.remove()
            outputs = {}
            for name in OUTPUTS:
                value = F.normalize(torch.cat(features[name]), dim=1)
                distances = torch.cdist(value, value).numpy()
                scores = full_gallery_scores(distances, identities, cameras)
                expected = saved["outputs"][name]
                assert scores == expected, (index, endpoint, name)
                outputs[name] = {"metrics_percent": scores["metrics_percent"],
                                 "queries": match_rows(distances, identities, cameras, scores)}
            geometry = {}
            for i, expert in enumerate(EXPERTS):
                value = torch.cat(corrected[expert])
                coefficient = (value * directions[i]).sum(1)
                geometry[expert] = {"correction_norm": torch.cat(delta[expert]).norm(dim=1).tolist(),
                                    "teacher_corrected_cosine": F.cosine_similarity(teacher[expert], value).tolist(),
                                    "post_correction_direction_coefficient": coefficient.tolist()}
                if endpoint == "projected":
                    assert coefficient.abs().max().item() < 1e-6
            assert _tensor_mapping_sha256(core.state_dict()) == before
            assert _sha256(path) == saved["checkpoint_sha256"]
            result["endpoints"][endpoint] = {"checkpoint_sha256": saved["checkpoint_sha256"],
                                             "head_state_sha256": before, "state_unchanged": True,
                                             "all_five_metric_arrays_exact": True, "outputs": outputs, "geometry": geometry}
            print(json.dumps({"fold": index, "endpoint": endpoint, "all_five_metric_arrays_exact": True}), flush=True)
            del core, payload, features, corrected, delta
            torch.cuda.empty_cache()
        for name in OUTPUTS:
            a,b=(result["endpoints"][e]["outputs"][name]["queries"] for e in ("uncentered","projected"))
            for left,right in zip(a,b,strict=True):
                qi = left["query_index"]
                assert qi == right["query_index"]
                row = {"fold":index, "file":manifest[qi]["file"], "identity":manifest[qi]["identity"],
                       "camera":manifest[qi]["camera"], "ap_delta_pp":100*(right["average_precision"]-left["average_precision"]),
                       "rank1_repaired":left["first_match_rank"]>1 and right["first_match_rank"]==1,
                       "rank1_broken":left["first_match_rank"]==1 and right["first_match_rank"]>1,
                       **{key+"_delta":right[key]-left[key] for key in ("positive_distance","negative_distance","nearest_margin")},
                       "uncentered_negative_same_camera":manifest[left["nearest_negative"]]["camera"]==manifest[qi]["camera"],
                       "projected_negative_same_camera":manifest[right["nearest_negative"]]["camera"]==manifest[qi]["camera"]}
                changes[name].append(row)
        folds.append(result)
    aggregate = {}
    for name,rows in changes.items():
        aggregate[name] = {key: float(np.mean([r[key] for r in rows])) for key in
                           ("positive_distance_delta","negative_distance_delta","nearest_margin_delta",
                            "uncentered_negative_same_camera","projected_negative_same_camera")}
        aggregate[name].update({"rank1_repaired":sum(r["rank1_repaired"] for r in rows),
                                "rank1_broken":sum(r["rank1_broken"] for r in rows)})
    result = {"status":"ALL_SIX_FINAL_HEAD_REPLAYS_MATCH", "scope":"read_only_train_only_postmortem_not_new_validation",
              "script_sha256":_sha256(Path(__file__)), "q1_summary_sha256":args.q1_sha256,
              "v17_geometry_sha256":_sha256(args.v17_geometry),
              "repository_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
              "optimizer_steps":0, "checkpoint_writes":0, "dev_access_count":0, "official_test_access_count":0,
              "folds":folds, "all_query_changes":changes, "aggregate":aggregate, "elapsed_seconds":time.time()-started}
    (args.output_dir/"diagnostic.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("status","aggregate","elapsed_seconds")}),flush=True)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q1-summary",type=Path,required=True)
    parser.add_argument("--q1-sha256",required=True)
    parser.add_argument("--v17-geometry",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    run(parser.parse_args())
