#!/usr/bin/env python3
"""Replay all saved V29 source matrices, vector scalars, exposures, and relations."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import time
import numpy as np
from tools.verify_v27_source_style_relations import sha
from tools.v29_source_drift_math import (
    STATES, VIEWS, OUTPUTS, EXPERTS, MODALITIES, COMPARISONS, VECTOR_COMPARISONS,
    VECTOR_FIELDS, PROTOCOLS, CHANGE_FIELDS, case_relations,
    validate_matrices, validate_vector_statistics,
)


def new_group():
    return {"cases":0, "triplets":0, "changes":np.zeros((3,2,18,12)),
            "stable":np.zeros((3,2,18,3),dtype=np.int64),
            "joint_changes":np.zeros((2,12)), "joint_stable":np.zeros((2,3),dtype=np.int64)}


def encoded(group):
    return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in group.items()}


def quantiles(values):
    x = np.asarray(values,dtype=np.float64)
    assert x.size > 0 and np.isfinite(x).all()
    return {"count":int(x.size),"mean":float(x.mean()),"min":float(x.min()),
            "p05":float(np.quantile(x,.05)),"median":float(np.median(x)),
            "p95":float(np.quantile(x,.95)),"max":float(x.max())}


def verify(args):
    started = time.time()
    run = args.run_dir
    exit_receipt = json.loads(Path(str(run)+"_diagnostic_exit.json").read_bytes())
    assert exit_receipt["exit_code"] == 0
    assert not Path("/proc",str(exit_receipt["original_pid"])).exists()
    raw = (run/"source_role_drift.json").read_bytes()
    summary = json.loads(raw)
    assert summary["status"] == "COMPLETE_FIXED_SOURCE_ROLE_DRIFT_PENDING_CPU_VERIFICATION"
    assert summary["completed_batches"] == 1680 and summary["model_forwards"] == 10080
    contract = json.loads(args.contract.read_bytes())
    assert sha(args.contract) == summary["contract_sha256"]
    assert contract["verifier_sha256"] == sha(__file__)
    assert summary["runner_sha256"] == contract["runner_sha256"]
    for path, expected in contract["source_file_sha256"].items():
        assert sha(path) == expected, path
    assert summary["new_optimizer_updates"] == 0 and not summary["encoder_gradients_computed"]
    assert summary["dev_image_reads"] == summary["official_image_reads"] == summary["heldout_image_reads"] == 0
    assert not summary["retrieval_evaluation"] and not summary["model_selection"]
    assert summary["states"] == list(STATES) and summary["outputs"] == list(OUTPUTS)
    assert summary["vector_fields"] == list(VECTOR_FIELDS)
    assert summary["vector_comparisons"] == list(VECTOR_COMPARISONS)
    prior_dir = Path(contract["v29_run"])
    assert sha(prior_dir/"run_summary.json") == contract["v29_summary_sha256"]
    assert sha(prior_dir/"complete_terminal_verification.json") == contract["v29_verification_sha256"]
    prior = json.loads((prior_dir/"run_summary.json").read_bytes())
    config = json.loads(Path(contract["v29_config"]).read_bytes())
    replay = json.loads(Path(config["SUPERVISION_METADATA"]["PATH"]).read_bytes())
    groups, identities = {}, {}
    vector_groups, vector_identities = [], []
    checked_batches, checked_triplets, vector_error = 0, 0, 0.
    for fold in summary["folds"]:
        number = fold["fold"]
        assert fold["all_model_states_unchanged"] and fold["all_gradients_absent"]
        for name in ("similarities","vectors","receipts"):
            item = fold[name]
            assert Path(item["path"]).stat().st_size == item["bytes"]
            assert sha(item["path"]) == item["sha256"]
        for i,binding in enumerate(fold["bindings"]):
            assert binding["state"] == STATES[i]
            assert binding["initial_state_sha256"] == prior["preflight"][number]["endpoints"][int(i==2)]["initial_state_sha256"]
            if i:
                endpoint = "control" if i==1 else "bounded_joint"
                expected = prior["folds"][number]["endpoints"][endpoint]
                assert binding["fixed_state_sha256"] == expected["training"]["final_state_sha256"]
                assert binding["checkpoint_sha256"] == sha(binding["checkpoint"]) == expected["checkpoint_sha256"]
        count = (580,560,540)[number]
        matrices = np.load(fold["similarities"]["path"],mmap_mode="r")
        vectors = np.load(fold["vectors"]["path"],mmap_mode="r")
        assert matrices.shape == (count,3,2,18,64,64) and matrices.dtype == np.float32
        assert vectors.shape == (count,2,4,64,3,3,6) and vectors.dtype == np.float64
        receipts = [json.loads(line) for line in Path(fold["receipts"]["path"]).read_text().splitlines()]
        assert len(receipts) == count
        metadata = replay["folds"][number]
        manifest = metadata["source_manifest"]
        assert fold["source_manifest"] == manifest
        original_ids = {r["identity"]:r["file"].split("_",1)[0] for r in manifest}
        assert len(original_ids) == 94
        for protocol in PROTOCOLS:
            for identity in original_ids:
                identities[(number,protocol,identity)] = new_group()
        exposure = Counter()
        batch_labels = []
        active_flags = []
        for case,(receipt,wanted) in enumerate(zip(receipts,metadata["arms"]["control"]["batches"],strict=True)):
            assert receipt["case_index"] == case
            for name in ("epoch","step","sampler_indices","sample_order_sha256","directed_cross_camera_positive_pairs"):
                assert receipt[name] == wanted[name]
            indices = receipt["sampler_indices"]
            exposure.update(indices)
            labels = np.array([manifest[i]["identity"] for i in indices])
            cameras = np.array([manifest[i]["camera"] for i in indices])
            assert sorted(Counter(labels).values()) == [8]*8
            batch_labels.append(labels)
            rng = np.random.default_rng(np.random.SeedSequence([42,number,case]))
            active = bool(rng.uniform()<.5)
            scores = rng.uniform(size=(64,64))
            donors = [int(np.flatnonzero(cameras!=camera)[np.argmin(scores[i,cameras!=camera])])
                      for i,camera in enumerate(cameras)]
            plan = {"fold":number,"step":case,"active":active,"forced_active":False,
                    "donors":donors,"coefficients":rng.beta(.1,.1,size=64).tolist(),
                    "all_donors_cross_camera":True}
            assert receipt["style_plan"] == plan
            active_flags.append(active)
            if case < 8:
                for endpoint in ("control","bounded_joint"):
                    assert receipt["raw_receipt"] == prior["folds"][number]["endpoints"][endpoint]["training"]["first_eight_batch_receipts"][case]
            matrix = matrices[case]
            validate_matrices(matrix,active)
            values = vectors[case]
            vector_error = max(vector_error,validate_vector_statistics(values))
            assert values[:,3,...,4].min() >= 1/np.sqrt(1.25)-2e-6
            if not active:
                assert np.array_equal(values[0],values[1])
            assert len(receipt["candidate_geometry_per_view"]) == 2
            for stats in receipt["candidate_geometry_per_view"]:
                assert stats["geometry_bound"] == .5 and stats["geometry_slot_observations"] == 576
                assert stats["geometry_update_ratio_max"] <= .5+2e-6
                assert stats["geometry_cosine_min"] >= 1/np.sqrt(1.25)-2e-6
            for protocol in PROTOCOLS:
                result = case_relations(matrix,labels,cameras,protocol)
                expected_count = 25088 if protocol == "identity" else wanted["directed_cross_camera_positive_pairs"]*56
                assert result["triplets"] == expected_count
                checked_triplets += expected_count
                key = (number,protocol,"active" if active else "inactive")
                if key not in groups: groups[key] = new_group()
                group = groups[key]
                group["cases"] += 1; group["triplets"] += result["triplets"]
                for name in ("changes","stable","joint_changes","joint_stable"):
                    group[name] += result[name]
                for row in result["relation_by_identity"]:
                    target = identities[(number,protocol,row["identity"])]
                    target["cases"] += 1
                    target["triplets"] += row["triplets"]
                    for name in ("changes","joint_changes","joint_stable"):
                        target[name] += np.asarray(row[name])
                for row in result["stable_by_identity"]:
                    target = identities[(number,protocol,row["identity"])]
                    index = row["comparison"]
                    target["stable"][index,:, :,0] += np.asarray(row["reliable_count_per_output"])[None]
                    target["stable"][index,:, :,1] += np.asarray(row["lost_per_view_output"])
                    target["stable"][index,:, :,2] += np.asarray(row["hinge_violated_per_view_output"])
            checked_batches += 1
        assert len(exposure) == len(manifest) and min(exposure.values()) > 0
        for identity in metadata["arms"]["control"]["identities"]:
            assert [exposure[i] for i in identity["record_indices"]] == identity["record_exposures"]
        labels = np.stack(batch_labels)
        active = np.asarray(active_flags)
        for view in range(2):
            for comparison in range(4):
                for role in range(3):
                    for modality in range(3):
                        data = vectors[:,view,comparison,:,role,modality]
                        for stratum,mask in (("all",np.ones(count,dtype=bool)),("active",active),("inactive",~active)):
                            selected = data[mask]
                            vector_groups.append({"fold":number,"view":VIEWS[view],"comparison":VECTOR_COMPARISONS[comparison],
                                "role":EXPERTS[role],"modality":MODALITIES[modality],"stratum":stratum,
                                "cosine":quantiles(selected[...,4]),"chord":quantiles(selected[...,5])})
                        for identity,original in sorted(original_ids.items()):
                            selected = data[labels==identity]
                            vector_identities.append({"fold":number,"identity":original,"encoded_identity":identity,
                                "view":VIEWS[view],"comparison":VECTOR_COMPARISONS[comparison],
                                "role":EXPERTS[role],"modality":MODALITIES[modality],
                                "cosine":quantiles(selected[:,4]),"chord":quantiles(selected[:,5])})
        # Check identity sums against the full active+inactive relations.
        for protocol in PROTOCOLS:
            identity_rows = [v for (f,p,_),v in identities.items() if f==number and p==protocol]
            global_rows = [v for (f,p,_),v in groups.items() if f==number and p==protocol]
            assert sum(v["triplets"] for v in identity_rows) == sum(v["triplets"] for v in global_rows)
            for name in ("changes","stable","joint_changes","joint_stable"):
                assert np.allclose(sum(v[name] for v in identity_rows),sum(v[name] for v in global_rows),rtol=1e-10,atol=1e-8)
        del matrices,vectors
        print(json.dumps({"verified_fold":number,"checked_batches":checked_batches,
                          "checked_protocol_triplets":checked_triplets,"elapsed_seconds":time.time()-started}),flush=True)
    assert checked_batches == 1680 and checked_triplets == 45549504
    assert len(groups) == 12 and len(identities) == 564
    assert len(vector_groups) == 648 and len(vector_identities) == 20304
    relation_groups = [{"fold":f,"protocol":p,"stratum":s,**encoded(g)} for (f,p,s),g in sorted(groups.items())]
    relation_identities = []
    for (f,p,i),g in sorted(identities.items()):
        original = next(r["file"].split("_",1)[0] for r in replay["folds"][f]["source_manifest"] if r["identity"]==i)
        relation_identities.append({"fold":f,"protocol":p,"encoded_identity":i,"identity":original,**encoded(g)})
    for filename,data in (("all_source_relation_groups.json",relation_groups),
                          ("all_source_identity_relations.json",relation_identities),
                          ("all_source_vector_groups.json",vector_groups),
                          ("all_source_identity_vectors.json",vector_identities)):
        path = run/filename
        assert not path.exists()
        path.write_text(json.dumps(data,separators=(",",":"))+"\n",encoding="utf-8")
    proof = {
        "status":"PASS_COMPLETE_V29_SOURCE_ROLE_DRIFT_AND_RELATIONS",
        "verified_at":datetime.now().astimezone().isoformat(),"source_summary_sha256":sha(run/"source_role_drift.json"),
        "verifier_sha256":sha(__file__),"contract_sha256":sha(args.contract),
        "checked_batches":checked_batches,"checked_original_model_forwards":10080,
        "checked_similarity_values":743178240,"checked_vector_observations":7741440,
        "checked_protocol_triplets_per_state_view":checked_triplets,
        "maximum_vector_gpu_numpy_algebra_error":vector_error,
        "relation_group_count":12,"relation_identity_count":564,
        "vector_group_count":648,"vector_identity_count":20304,
        "relation_change_fields":list(CHANGE_FIELDS),
        "stable_fields":["reliable_original_and_style_count","lost_correct_count","hinge_violated_count"],
        "comparisons":list(COMPARISONS),"states":list(STATES),"views":list(VIEWS),"outputs":list(OUTPUTS),
        "all_source_exposures_match":True,"all_model_states_unchanged_receipted":True,
        "zero_eligible_cross_camera_identity_rows_preserved":True,
        "new_model_forwards":0,"new_optimizer_updates":0,"image_reads":0,"torch_imports":0,
        "independent_external_review":False,
        "boundaries":["Stored FP32 within-model matrices are the basis for NumPy relation statistics.",
                      "Discarded model vectors are not rerun by this CPU verifier.",
                      "Geometry basis scalars were derived from every original vector pair and compared with GPU measurements.",
                      "Current h-to-y bound does not constrain initial-to-final role drift.",
                      "Batch exposure counts include repeated images and relations, not independent trials.",
                      "No AP/CMC, heldout/dev/official access, model selection, or hyperparameter scan."],
        "files":{p.name:{"bytes":p.stat().st_size,"sha256":sha(p)} for p in run.glob("all_source_*.json")},
        "elapsed_seconds":time.time()-started,
    }
    assert not args.output.exists()
    args.output.write_text(json.dumps(proof,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(proof),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--contract",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    verify(parser.parse_args())
