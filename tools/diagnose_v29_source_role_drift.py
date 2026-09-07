#!/usr/bin/env python3
"""Fixed source role-geometry drift at initial/control/V29 endpoints; zero updates."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import torch
import torch.nn.functional as F
from tools.build_v12_complete_path_oof_targets import _configure_signal, _load_records, build_complete_path_fold_records
from tools.run_signal_preserving_v5 import _training_batch
from tools.train_signal_preserving_v17 import _model_state_sha256, _raw_batch_receipt, _record_index_by_path, _sha256
from tools.train_signal_preserving_v19 import seed_everything
from tools.train_signal_preserving_v29 import build_model, check_batch, load_contract, training_loader
from tools.diagnose_v27_source_style_relations import similarities
from tools.v29_source_drift_math import (
    STATES, VIEWS, EXPERTS, MODALITIES, OUTPUTS, VECTOR_COMPARISONS, VECTOR_FIELDS,
    validate_matrices, validate_vector_statistics,
)
from trifusion.source_style_v27 import make_style_plan


def paired_vectors(left, right):
    left, right = F.normalize(left.float(), dim=-1), F.normalize(right.float(), dim=-1)
    assert left.shape == right.shape and left.shape[1:] == (3, 3, 512)
    a, b = [x.cpu().numpy().astype(np.float64) for x in (left, right)]
    basis = np.stack(((a*a).sum(-1), (b*b).sum(-1), (a*b).sum(-1),
                      np.square(a-b).sum(-1)), axis=-1)
    observed = torch.stack((F.cosine_similarity(left, right, dim=-1),
                            torch.linalg.vector_norm(left-right, dim=-1)), dim=-1)
    values = np.concatenate((basis, observed.cpu().numpy().astype(np.float64)), axis=-1)
    error = validate_vector_statistics(values)
    return values, error


def load_models(config, signal_cfg, fold, split, prior):
    models, bindings = [], []
    for i, state in enumerate(STATES):
        enabled = i == 2
        model, binding = build_model(config, signal_cfg, fold, split, enabled=enabled)
        initial = _model_state_sha256(model)
        assert initial == prior["preflight"][fold]["endpoints"][int(enabled)]["initial_state_sha256"]
        checkpoint = None
        checkpoint_sha = None
        if i:
            endpoint = "control" if i == 1 else "bounded_joint"
            receipt = prior["folds"][fold]["endpoints"][endpoint]
            checkpoint, checkpoint_sha = Path(receipt["checkpoint"]), receipt["checkpoint_sha256"]
            assert _sha256(checkpoint) == checkpoint_sha
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            assert saved["binding"] == binding
            tensors = model.state_dict()
            assert set(saved["v29_state_dict"]) == {k for k in tensors if not k.startswith("baseline.")}
            tensors.update(saved["v29_state_dict"])
            model.load_state_dict(tensors, strict=True)
            del tensors, saved
            assert _model_state_sha256(model) == receipt["training"]["final_state_sha256"]
        fixed = _model_state_sha256(model)
        model.eval()
        model.baseline.train(True)  # Existing frozen stem interface; all learned modules/BN stay eval.
        assert not model.encoder.training and not model.baseline.signal.training and not model.fused_neck.training
        models.append(model)
        bindings.append({"state":state, "initial_state_sha256":initial, "fixed_state_sha256":fixed,
                         "checkpoint":str(checkpoint) if checkpoint else None,
                         "checkpoint_sha256":checkpoint_sha, "binding":binding})
    return models, bindings


def run(args):
    started = time.time()
    contract = json.loads(args.contract.read_bytes())
    assert _sha256(args.contract) == args.contract_sha256
    assert contract["states"] == list(STATES) and contract["views"] == list(VIEWS)
    assert contract["outputs"] == list(OUTPUTS) and contract["vector_fields"] == list(VECTOR_FIELDS)
    assert contract["planned_batches"] == 1680 and contract["planned_model_forwards"] == 10080
    assert contract["optimizer_updates"] == 0
    for path, expected in contract["source_file_sha256"].items():
        assert _sha256(Path(path)) == expected, path
    assert _sha256(Path(__file__)) == contract["runner_sha256"]
    assert shutil.disk_usage(args.output_dir.parent).free >= contract["minimum_free_bytes"]
    prior_dir = Path(contract["v29_run"])
    assert _sha256(prior_dir/"run_summary.json") == contract["v29_summary_sha256"]
    assert _sha256(prior_dir/"complete_terminal_verification.json") == contract["v29_verification_sha256"]
    terminal = json.loads(Path(str(prior_dir)+"_pipeline_exit.json").read_bytes())
    assert terminal["stage"] == "COMPLETE_VERIFIED_Q1_FAIL" and terminal["exit_code"] == 0
    prior = json.loads((prior_dir/"run_summary.json").read_bytes())
    assert prior["status"] == "Q1_FAIL" and len(prior["folds"]) == 3
    assert _sha256(Path(contract["v29_config"])) == contract["v29_config_sha256"]
    config, sources = load_contract(Path(contract["v29_config"]))
    signal_cfg, signal_commit, signal_diff = _configure_signal(config)
    records = _load_records(config)
    splits = [build_complete_path_fold_records(records, heldout_ids=set(r["heldout_identity_ids"]))
              for r in sources["fold_receipts"]]
    replay = json.loads(Path(config["SUPERVISION_METADATA"]["PATH"]).read_bytes())
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "status":"RUNNING_FIXED_SOURCE_ROLE_DRIFT", "started_at":datetime.now().astimezone().isoformat(),
        "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "runner_sha256":_sha256(Path(__file__)), "contract_sha256":args.contract_sha256,
        "prior_summary_sha256":contract["v29_summary_sha256"],
        "signal_commit":signal_commit, "signal_diff_sha256":signal_diff,
        "states":list(STATES), "views":list(VIEWS), "outputs":list(OUTPUTS),
        "vector_comparisons":list(VECTOR_COMPARISONS), "vector_fields":list(VECTOR_FIELDS),
        "experts":list(EXPERTS), "modalities":list(MODALITIES),
        "new_optimizer_updates":0, "encoder_gradients_computed":False,
        "model_mode":"fixed_eval_with_frozen_stem_training_flag",
        "dev_image_reads":0, "official_image_reads":0, "heldout_image_reads":0,
        "retrieval_evaluation":False, "model_selection":False,
        "folds":[], "completed_batches":0, "model_forwards":0,
    }
    def save():
        summary["elapsed_seconds"] = time.time()-started
        (args.output_dir/"source_role_drift.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    save()
    for fold, split in enumerate(splits):
        assert not split["identity_overlap"]
        actual = [{"file":Path(r[0][0]).name,"identity":r[1],"camera":r[2]} for r in split["train_records"]]
        assert actual == replay["folds"][fold]["source_manifest"]
        expected = replay["folds"][fold]["arms"]["control"]
        models, bindings = load_models(config, signal_cfg, fold, split, prior)
        seed_everything()
        loader = training_loader(split["train_records"], config, "control")
        index = _record_index_by_path(split["train_records"])
        count = expected["batch_count"]
        matrix_path = args.output_dir/f"fold_{fold}_similarities.npy"
        vector_path = args.output_dir/f"fold_{fold}_vector_statistics.npy"
        matrix_array = np.lib.format.open_memmap(matrix_path,mode="w+",dtype=np.float32,
                                                 shape=(count,3,2,18,64,64))
        vector_array = np.lib.format.open_memmap(vector_path,mode="w+",dtype=np.float64,
                                                 shape=(count,2,4,64,3,3,6))
        receipts_path = args.output_dir/f"fold_{fold}_batch_receipts.jsonl"
        exposures, steps = Counter(), 0
        max_vector_error, max_joint_decomposition_error = 0., 0.
        with receipts_path.open("x",encoding="utf-8") as receipts, torch.no_grad():
            for epoch in range(1,21):
                for step, raw in enumerate(loader,1):
                    wanted = expected["batches"][steps]
                    assert (wanted["epoch"],wanted["step"]) == (epoch,step)
                    checked = check_batch(raw,index,wanted)
                    raw_receipt = _raw_batch_receipt(raw,record_index_by_path=index)
                    if steps < 8:
                        for endpoint in ("control","bounded_joint"):
                            assert raw_receipt == prior["folds"][fold]["endpoints"][endpoint]["training"]["first_eight_batch_receipts"][steps]
                    exposures.update(checked["sampler_indices"])
                    plan = make_style_plan(raw[2].numpy(),fold=fold,step=steps)
                    batch, _labels = _training_batch(raw)
                    matrices = np.empty((3,2,18,64,64),dtype=np.float32)
                    vectors = np.empty((2,4,64,3,3,6),dtype=np.float64)
                    signal = None
                    per_view_geometry = []
                    for view_index, view in enumerate(VIEWS):
                        slots = []
                        for state_index, model in enumerate(models):
                            model.baseline.style_enabled = view_index == 1
                            model.baseline.style_plan = plan
                            with torch.autocast("cuda",dtype=torch.float16):
                                output = model(batch,return_aux=True)
                            assert output.diagnostics["all_finite"] and output.diagnostics["baseline_exact_prefix"]
                            if signal is None: signal = output.baseline_embedding.clone()
                            assert torch.equal(signal,output.baseline_embedding)
                            matrices[state_index,view_index] = similarities(output).cpu().numpy()
                            h = torch.stack([output.modal_residual_embeddings[e] for e in EXPERTS],dim=1)
                            slots.append(h)
                            if state_index == 2:
                                y = F.normalize(output.fused_embedding[:,model.baseline_embedding_width:].reshape_as(h).float(),dim=-1)
                                ys = F.normalize(y.flatten(1),dim=1)
                                ym = ys @ ys.T
                                expected_fused = .5*torch.from_numpy(matrices[2,view_index,0]).cuda()+.5*ym
                                error = float((expected_fused-torch.from_numpy(matrices[2,view_index,1]).cuda()).abs().max())
                                assert error < 2e-5
                                max_joint_decomposition_error = max(max_joint_decomposition_error,error)
                                per_view_geometry.append(dict(model.last_joint_stats))
                            summary["model_forwards"] += 1
                            del output
                        pairs = ((slots[0],slots[1]),(slots[0],slots[2]),(slots[1],slots[2]),(slots[2],y))
                        for comparison,(left,right) in enumerate(pairs):
                            measured,error = paired_vectors(left,right)
                            vectors[view_index,comparison] = measured
                            max_vector_error = max(max_vector_error,error)
                        del slots,pairs,h,y,ys,ym,expected_fused,left,right
                    validate_matrices(matrices,plan["active"])
                    validate_vector_statistics(vectors)
                    assert vectors[:,3,...,4].min() >= 1/np.sqrt(1.25)-2e-6
                    if not plan["active"]:
                        assert np.array_equal(vectors[0],vectors[1])
                    matrix_array[steps] = matrices
                    vector_array[steps] = vectors
                    receipts.write(json.dumps({"fold":fold,"epoch":epoch,"step":step,"case_index":steps,
                                               **checked,"raw_receipt":raw_receipt,"style_plan":plan,
                                               "candidate_geometry_per_view":per_view_geometry})+"\n")
                    steps += 1
                    summary["completed_batches"] += 1
                receipts.flush();matrix_array.flush();vector_array.flush();save()
                print(json.dumps({"fold":fold,"epoch":epoch,"completed_batches":summary["completed_batches"],
                                  "model_forwards":summary["model_forwards"],"elapsed_seconds":summary["elapsed_seconds"]}),flush=True)
        assert steps == count == (580,560,540)[fold]
        for identity in expected["identities"]:
            assert [exposures[i] for i in identity["record_indices"]] == identity["record_exposures"]
        assert len(exposures) == len(actual) and min(exposures.values()) > 0
        for model,binding in zip(models,bindings,strict=True):
            assert _model_state_sha256(model) == binding["fixed_state_sha256"]
            assert all(p.grad is None for p in model.parameters())
        del matrix_array,vector_array,models,model
        torch.cuda.empty_cache()
        summary["folds"].append({
            "fold":fold,"source_records":len(actual),"source_manifest":actual,"bindings":bindings,
            "batches":steps,"model_forwards":steps*6,
            "maximum_vector_gpu_numpy_error":max_vector_error,
            "maximum_joint_similarity_decomposition_error":max_joint_decomposition_error,
            "all_model_states_unchanged":True,"all_gradients_absent":True,
            "all_source_exposures_match":True,"first_eight_pixels_match_original_both_arms":True,
            **{name:{"path":str(p),"bytes":p.stat().st_size,"sha256":_sha256(p)}
               for name,p in (("similarities",matrix_path),("vectors",vector_path),("receipts",receipts_path))},
        })
        save()
    assert summary["completed_batches"] == 1680 and summary["model_forwards"] == 10080
    summary["status"] = "COMPLETE_FIXED_SOURCE_ROLE_DRIFT_PENDING_CPU_VERIFICATION"
    summary["completed_at"] = datetime.now().astimezone().isoformat()
    save()
    print(json.dumps({"status":summary["status"],"model_forwards":10080,"optimizer_updates":0}),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract",type=Path,required=True)
    parser.add_argument("--contract-sha256",required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    run(parser.parse_args())
