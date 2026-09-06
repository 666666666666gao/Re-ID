#!/usr/bin/env python3
"""Full source-batch relation support at frozen initial/control/V27 checkpoints."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from tools.build_v12_complete_path_oof_targets import (
    _configure_signal, _load_records, build_complete_path_fold_records,
)
from tools.run_signal_preserving_v5 import _training_batch
from tools.train_signal_preserving_v17 import (
    _model_state_sha256, _raw_batch_receipt, _record_index_by_path, _sha256,
)
from tools.train_signal_preserving_v19 import seed_everything
from tools.train_signal_preserving_v27 import build_model, check_batch, load_contract, training_loader
from trifusion.source_style_v27 import make_style_plan

EXPERTS = ("cnn", "transformer", "mamba")
MODALITIES = ("RGB", "NI", "TI")
STATES = ("initial", "control_final", "style_final")
VIEWS = ("original", "registered_style")
PROTOCOLS = ("identity", "cross_camera")
OUTPUTS = ("baseline_only", "fused", *EXPERTS,
           *(f"{e}_{m}_residual" for e in EXPERTS for m in MODALITIES),
           "pure_bank", *(f"pure_{e}" for e in EXPERTS))
MARGIN_FIELDS = ("mean", "minimum", "maximum", "nonpositive", "euclidean_hinge_mean", "euclidean_hinge_positive")
WEIGHT_FIELDS = ("mean", "minimum", "maximum", "effective_sample_size", "weighted_loss",
                 "margin_derivative_l2", "unweighted_margin_derivative_l2", "joint_nonpositive")


def similarities(output):
    bank = torch.cat([output.residual_embeddings[e] for e in EXPERTS], dim=1)
    features = {
        "baseline_only": output.baseline_embedding, "fused": output.fused_embedding,
        **dict(output.branch_embeddings),
        **{f"{e}_{m}_residual": output.modal_residual_embeddings[e][:, i]
           for e in EXPERTS for i, m in enumerate(MODALITIES)},
        "pure_bank": bank,
        **{f"pure_{e}": output.residual_embeddings[e] for e in EXPERTS},
    }
    values = []
    for name in OUTPUTS:
        x = features[name].float()
        assert bool(torch.isfinite(x).all()) and bool((x.norm(dim=1) > 0).all())
        x = F.normalize(x, dim=1)
        values.append(x @ x.T)
    return torch.stack(values)


def relation_statistics(matrix, labels, cameras, protocol):
    assert protocol in PROTOCOLS and matrix.shape == (18, len(labels), len(labels))
    matrix = matrix.double()  # fixed FP64 statistics on the saved FP32 similarities
    same = labels[:, None] == labels[None, :]
    positive = same & ~torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    if protocol == "cross_camera":
        positive &= cameras[:, None] != cameras[None, :]
    q, p, n = torch.where(positive[:, :, None] & ~same[:, None, :])
    assert q.numel() > 0
    margins = matrix[:, q, p] - matrix[:, q, n]
    distances = (2 - 2 * matrix).clamp_min(0).sqrt()
    hinge = (distances[:, q, p] - distances[:, q, n] + 0.3).clamp_min(0)
    margin_values = torch.stack((
        margins.mean(1), margins.amin(1), margins.amax(1), (margins <= 0).sum(1),
        hinge.mean(1), (hinge > 0).sum(1),
    ), dim=1).cpu().tolist()
    slot = margins[5:14]
    fused = margins[1]
    weight = torch.sigmoid(-fused[None] / 0.1) * torch.sigmoid(-slot / 0.1)
    penalty = 0.1 * F.softplus(-slot / 0.1)
    unweighted_derivative = -torch.sigmoid(-slot / 0.1) / slot.numel()
    derivative = weight * unweighted_derivative
    weight_values = torch.stack((
        weight.mean(1), weight.amin(1), weight.amax(1),
        weight.sum(1).square() / weight.square().sum(1),
        (weight * penalty).mean(1), derivative.norm(dim=1), unweighted_derivative.norm(dim=1),
        ((slot <= 0) & (fused[None] <= 0)).sum(1),
    ), dim=1).cpu().tolist()
    patterns = []
    for index in range(9):
        role = margins[15 + index // 3]
        bits = ((margins[0] <= 0).long() * 8 + (margins[14] <= 0).long() * 4
                + (role <= 0).long() * 2 + (fused <= 0).long())
        patterns.append(torch.bincount(bits[slot[index] <= 0], minlength=16).cpu().tolist())
    return {
        "triplets": q.numel(), "slot_triplets": slot.numel(),
        "eligible_anchor_rows": int(positive.any(1).sum()),
        "margin_values": margin_values, "weight_values": weight_values,
        "slot_nonpositive_support_patterns": patterns,
        "v26_auxiliary_loss": float((weight * penalty).mean()),
        "v26_to_uniform_margin_derivative_norm_ratio": float(derivative.norm() / unweighted_derivative.norm()),
        "scope": "all_ordered_batch_triplets_no_optimizer_or_encoder_gradient",
    }


def decomposition_error(matrix):
    return max(
        float((matrix[1] - (0.5 * matrix[0] + matrix[5:14].sum(0) / 18)).abs().max()),
        float((matrix[1] - matrix[2:5].mean(0)).abs().max()),
        float((matrix[14] - matrix[15:18].mean(0)).abs().max()),
    )


def mathematical_check():
    rng = np.random.default_rng(42)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    cameras = torch.tensor([0, 1, 0, 0, 0, 1])
    baseline = torch.tensor(rng.normal(size=(6, 7)), dtype=torch.float64)
    baseline = F.normalize(baseline, dim=1)
    slots = F.normalize(torch.tensor(rng.normal(size=(6, 9, 5)), dtype=torch.float64), dim=2)
    roles = [F.normalize(slots[:, i:i+3].flatten(1), dim=1) for i in (0, 3, 6)]
    bank = F.normalize(torch.cat(roles, dim=1), dim=1)
    full = [F.normalize(torch.cat((baseline, x), dim=1), dim=1) for x in roles]
    fused = F.normalize(torch.cat((baseline, bank), dim=1), dim=1)
    values = [baseline, fused, *full, *(slots[:, i] for i in range(9)), bank, *roles]
    matrix = torch.stack([x @ x.T for x in values])
    assert decomposition_error(matrix) < 1e-12
    array = matrix.numpy()
    max_error = 0.0
    for protocol in PROTOCOLS:
        result = relation_statistics(matrix, labels, cameras, protocol)
        triplets = [(q, p, n) for q in range(6) for p in range(6) for n in range(6)
                    if q != p and labels[q] == labels[p] and labels[q] != labels[n]
                    and (protocol == "identity" or cameras[q] != cameras[p])]
        expected_count = 24 if protocol == "identity" else 16
        assert result["triplets"] == len(triplets) == expected_count
        assert result["eligible_anchor_rows"] == (6 if protocol == "identity" else 4)
        for head in range(18):
            margins = np.array([array[head,q,p]-array[head,q,n] for q,p,n in triplets])
            hinge = np.array([max(0, np.sqrt(max(0,2-2*array[head,q,p]))
                                 - np.sqrt(max(0,2-2*array[head,q,n])) + .3) for q,p,n in triplets])
            wanted = [margins.mean(), margins.min(), margins.max(), (margins<=0).sum(),
                      hinge.mean(), (hinge>0).sum()]
            max_error = max(max_error, float(np.max(np.abs(np.array(result["margin_values"][head])-wanted))))
        for index in range(9):
            m = np.array([array[5+index,q,p]-array[5+index,q,n] for q,p,n in triplets])
            fm = np.array([array[1,q,p]-array[1,q,n] for q,p,n in triplets])
            w = 1/(1+np.exp(fm/.1)) / (1+np.exp(m/.1))
            unweighted = -1/(1+np.exp(m/.1)) / (9*len(triplets))
            derivative = w * unweighted
            wanted = [w.mean(),w.min(),w.max(),w.sum()**2/(w*w).sum(),
                      (w*.1*np.logaddexp(0,-m/.1)).mean(),np.linalg.norm(derivative),
                      np.linalg.norm(unweighted),((m<=0)&(fm<=0)).sum()]
            max_error = max(max_error,float(np.max(np.abs(np.array(result["weight_values"][index])-wanted))))
            assert sum(result["slot_nonpositive_support_patterns"][index]) == int((m<=0).sum())
    assert max_error < 1e-10
    return {"status":"PASS_ALL_SYNTHETIC_MARGINS_WEIGHTS_AND_OUTPUT_DERIVATIVES",
            "maximum_error":max_error,"new_model_forwards":0,"optimizer_updates":0}


def load_fixed_models(config, signal_cfg, fold, split, prior):
    models, bindings = [], []
    for state_name in STATES:
        model, binding = build_model(config, signal_cfg, fold, split)
        initial = _model_state_sha256(model)
        assert initial == prior["preflight"][fold]["endpoints"][0]["initial_state_sha256"]
        checkpoint = None
        if state_name != "initial":
            endpoint = "control" if state_name == "control_final" else "source_style"
            receipt = prior["folds"][fold]["endpoints"][endpoint]
            checkpoint = Path(receipt["checkpoint"])
            assert _sha256(checkpoint) == receipt["checkpoint_sha256"]
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            assert saved["binding"] == binding
            tensors = model.state_dict()
            assert set(saved["v27_state_dict"]) == {k for k in tensors if not k.startswith("baseline.")}
            tensors.update(saved["v27_state_dict"])
            model.load_state_dict(tensors, strict=True)
            del saved, tensors
            assert _model_state_sha256(model) == receipt["training"]["final_state_sha256"]
        model.eval()
        model.baseline.train(True)  # only select the existing frozen stem diagnostic path
        assert not model.encoder.training and not model.baseline.signal.training
        assert not model.fused_neck.training
        models.append(model)
        bindings.append({"state":state_name,"initial_state_sha256":initial,
                         "fixed_state_sha256":_model_state_sha256(model),
                         "checkpoint":str(checkpoint) if checkpoint else None,"binding":binding})
    return models, bindings


def run(args):
    started = time.time()
    assert _sha256(args.contract) == args.contract_sha256
    contract = json.loads(args.contract.read_bytes())
    assert contract["states"] == list(STATES) and contract["views"] == list(VIEWS)
    assert contract["outputs"] == list(OUTPUTS) and contract["protocols"] == list(PROTOCOLS)
    assert contract["planned_batches"] == 1680 and contract["planned_model_forwards"] == 10080
    assert contract["optimizer_updates"] == 0 and contract["seed"] == 42
    for file, expected in contract["source_file_sha256"].items():
        assert _sha256(Path(file)) == expected, file
    assert _sha256(__file__) == contract["runner_sha256"]
    assert shutil.disk_usage(args.output_dir.parent).free >= contract["minimum_free_bytes"]
    prior_dir = Path(contract["v27_run"])
    assert _sha256(prior_dir/"run_summary.json") == contract["v27_summary_sha256"]
    assert _sha256(prior_dir/"terminal_verification.json") == contract["v27_verification_sha256"]
    prior = json.loads((prior_dir/"run_summary.json").read_bytes())
    assert prior["status"] == "Q1_FAIL" and len(prior["folds"]) == 3
    assert int((prior_dir/"terminal_verification.exit").read_text()) == 0
    assert _sha256(Path(contract["v27_config"])) == contract["v27_config_sha256"]
    config, sources = load_contract(Path(contract["v27_config"]))
    signal_cfg, signal_commit, signal_diff = _configure_signal(config)
    records = _load_records(config)
    splits = [build_complete_path_fold_records(records, heldout_ids=set(r["heldout_identity_ids"]))
              for r in sources["fold_receipts"]]
    replay = json.loads(Path(config["SUPERVISION_METADATA"]["PATH"]).read_bytes())
    torch.set_num_threads(4)
    math_check = mathematical_check()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status":"RUNNING_FIXED_SOURCE_STYLE_RELATION_DIAGNOSTIC",
        "started_at":datetime.now().astimezone().isoformat(),
        "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "runner_sha256":_sha256(__file__),"contract_sha256":args.contract_sha256,
        "prior_summary_sha256":contract["v27_summary_sha256"],"math_check":math_check,
        "signal_commit":signal_commit,"signal_diff_sha256":signal_diff,
        "states":list(STATES),"views":list(VIEWS),"outputs":list(OUTPUTS),"protocols":list(PROTOCOLS),
        "margin_fields":list(MARGIN_FIELDS),"weight_fields":list(WEIGHT_FIELDS),
        "new_optimizer_updates":0,"encoder_gradients_computed":False,"model_mode":"fixed_eval_with_frozen_stem_training_flag",
        "dev_image_reads":0,"official_image_reads":0,"heldout_image_reads":0,
        "model_selection":False,"folds":[],"completed_batches":0,"model_forwards":0,
    }
    summary_path = args.output_dir/"style_relation_support.json"
    def save():
        report["elapsed_seconds"] = time.time()-started
        summary_path.write_bytes((json.dumps(report,indent=2)+"\n").encode())
    save()
    cases_path = args.output_dir/"all_batch_cases.jsonl.gz"
    with gzip.open(cases_path,"wt",encoding="utf-8") as cases:
        for fold, split in enumerate(splits):
            assert not split["identity_overlap"]
            actual = [{"file":Path(r[0][0]).name,"identity":r[1],"camera":r[2]} for r in split["train_records"]]
            assert actual == replay["folds"][fold]["source_manifest"]
            expected = replay["folds"][fold]["arms"]["control"]
            models, bindings = load_fixed_models(config,signal_cfg,fold,split,prior)
            seed_everything()
            loader = training_loader(split["train_records"],config,"control")
            index = _record_index_by_path(split["train_records"])
            count = expected["batch_count"]
            array_path = args.output_dir/f"fold_{fold}_similarities.npy"
            arrays = np.lib.format.open_memmap(array_path,mode="w+",dtype=np.float32,shape=(count,3,2,18,64,64))
            receipts_path = args.output_dir/f"fold_{fold}_batch_receipts.jsonl"
            exposure, steps, maximum_decomposition_error = Counter(),0,0.0
            first_eight_equal = []
            with receipts_path.open("x",encoding="utf-8") as receipts, torch.no_grad():
                for epoch in range(1,21):
                    for within_epoch, raw in enumerate(loader,1):
                        wanted = expected["batches"][steps]
                        assert (wanted["epoch"],wanted["step"]) == (epoch,within_epoch)
                        checked = check_batch(raw,index,wanted)
                        receipt = _raw_batch_receipt(raw,record_index_by_path=index)
                        if steps < 8:
                            assert receipt == prior["folds"][fold]["endpoints"]["control"]["training"]["first_eight_batch_receipts"][steps]
                            first_eight_equal.append(True)
                        exposure.update(checked["sampler_indices"])
                        plan = make_style_plan(raw[2].numpy(),fold=fold,step=steps)
                        receipts.write(json.dumps({"fold":fold,"epoch":epoch,"step":within_epoch,"case_index":steps,
                                                   **checked,"raw_receipt":receipt,"style_plan":plan})+"\n")
                        batch, labels = _training_batch(raw)
                        cameras = batch["camera_ids"]
                        signal_matrix = None
                        for state_index,(state_name,model) in enumerate(zip(STATES,models,strict=True)):
                            original_matrix = None
                            for view_index,view in enumerate(VIEWS):
                                model.baseline.style_enabled = view == "registered_style"
                                model.baseline.style_plan = plan
                                with torch.autocast("cuda",dtype=torch.float16):
                                    output = model(batch,return_aux=True)
                                assert output.diagnostics["all_finite"] and output.diagnostics["baseline_exact_prefix"]
                                matrix = similarities(output)
                                error = decomposition_error(matrix)
                                assert error < .005
                                maximum_decomposition_error = max(maximum_decomposition_error,error)
                                if signal_matrix is None: signal_matrix = matrix[0].clone()
                                assert torch.equal(matrix[0],signal_matrix)
                                if view_index == 0: original_matrix = matrix.clone()
                                elif not plan["active"]: assert torch.equal(matrix,original_matrix)
                                arrays[steps,state_index,view_index] = matrix.cpu().numpy()
                                row = {"fold":fold,"case_index":steps,"epoch":epoch,"step":within_epoch,
                                       "state":state_name,"view":view,"plan_active":plan["active"],
                                       "actual_style_active":bool(view_index and plan["active"]),
                                       "decomposition_error":error,
                                       "protocols":{p:relation_statistics(matrix,labels,cameras,p) for p in PROTOCOLS}}
                                assert row["protocols"]["identity"]["triplets"] == 25088
                                assert row["protocols"]["cross_camera"]["triplets"] == wanted["directed_cross_camera_positive_pairs"]*56
                                cases.write(json.dumps(row)+"\n")
                                report["model_forwards"] += 1
                                del output,matrix
                        steps += 1
                        report["completed_batches"] += 1
                    receipts.flush(); cases.flush(); arrays.flush(); save()
                    print(json.dumps({"fold":fold,"epoch":epoch,"completed_batches":report["completed_batches"],
                                      "model_forwards":report["model_forwards"],"elapsed_seconds":report["elapsed_seconds"]}),flush=True)
            assert steps == count == (580,560,540)[fold] and len(first_eight_equal)==8
            for identity in expected["identities"]:
                assert [exposure[i] for i in identity["record_indices"]] == identity["record_exposures"]
            for model,binding in zip(models,bindings,strict=True):
                assert _model_state_sha256(model) == binding["fixed_state_sha256"]
                assert all(p.grad is None for p in model.parameters())
            arrays.flush()
            del arrays,models
            torch.cuda.empty_cache()
            report["folds"].append({"fold":fold,"source_records":len(actual),"source_manifest":actual,
                "bindings":bindings,"batches":steps,"model_forwards":steps*6,
                "maximum_decomposition_error":maximum_decomposition_error,
                "all_model_states_unchanged":True,"all_gradients_absent":True,
                "first_eight_augmentation_receipts_equal_original_v27":True,
                "all_sample_orders_and_record_exposures_equal_registered":True,
                "similarities":{"path":str(array_path),"bytes":array_path.stat().st_size,"sha256":_sha256(array_path)},
                "batch_receipts":{"path":str(receipts_path),"sha256":_sha256(receipts_path)}})
            save()
    assert report["completed_batches"]==1680 and report["model_forwards"]==10080 and len(report["folds"])==3
    report["cases"]={"path":str(cases_path),"rows":10080,"bytes":cases_path.stat().st_size,"sha256":_sha256(cases_path)}
    report["status"]="COMPLETE_FIXED_SOURCE_STYLE_RELATION_DIAGNOSTIC_PENDING_CPU_VERIFICATION"
    report["completed_at"]=datetime.now().astimezone().isoformat()
    save()
    print(json.dumps({"status":report["status"],"model_forwards":report["model_forwards"],"optimizer_updates":0}),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract",type=Path)
    parser.add_argument("--contract-sha256")
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--math-only",action="store_true")
    args=parser.parse_args()
    if args.math_only: print(json.dumps(mathematical_check()))
    else: run(args)
