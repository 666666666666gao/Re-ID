#!/usr/bin/env python3
"""Measure every fixed-source V28 R2 slot; no optimizer or retrieval evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
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
from tools.train_signal_preserving_v28_fp32 import build_model, check_batch, load_contract, training_loader
from trifusion.joint_tokens_v28 import normalized_joint_bank
from trifusion.source_style_v27 import make_style_plan

EXPERTS = ("cnn", "transformer", "mamba")
MODALITIES = ("RGB", "NI", "TI")
VIEWS = ("original", "registered_style")
BASIS = ("h2", "c2", "hc", "sum2", "y2", "yh", "yc", "correction_abs_mean")
DERIVED = ("h_norm", "c_norm", "sum_norm", "c_to_h_norm",
           "cos_y_h", "cos_y_c", "cos_h_c", "distance_y_h_unit")
FIELDS = BASIS + DERIVED


def measure(h, c, y):
    h, c, y = h.float(), c.float(), y.float()
    assert h.shape == c.shape == y.shape and h.shape[1:] == (3, 3, 512)
    assert all(bool(torch.isfinite(value).all()) for value in (h, c, y))
    hn, cn, sn = (torch.linalg.vector_norm(value, dim=-1) for value in (h, c, h + c))
    assert bool((hn > 0).all())
    gpu = torch.stack((hn, cn, sn, cn / hn,
                       F.cosine_similarity(y, h, dim=-1, eps=1e-8),
                       F.cosine_similarity(y, c, dim=-1, eps=1e-8),
                       F.cosine_similarity(h, c, dim=-1, eps=1e-8),
                       torch.linalg.vector_norm(y - F.normalize(h, dim=-1), dim=-1)), dim=-1)
    a, b, z = [value.cpu().numpy().astype(np.float64) for value in (h, c, y)]
    h2, c2 = (np.square(value).sum(axis=-1) for value in (a, b))
    hc = (a * b).sum(axis=-1)
    sum2, y2 = np.square(a + b).sum(axis=-1), np.square(z).sum(axis=-1)
    yh, yc = (z * a).sum(axis=-1), (z * b).sum(axis=-1)
    basis = np.stack((h2, c2, hc, sum2, y2, yh, yc, np.abs(b).mean(axis=-1)), axis=-1)
    hn64, cn64, sn64, yn64 = [np.sqrt(value) for value in (h2, c2, sum2, y2)]
    expected = np.stack((hn64, cn64, sn64, cn64 / hn64,
                         yh / (np.maximum(yn64, 1e-8) * np.maximum(hn64, 1e-8)),
                         yc / (np.maximum(yn64, 1e-8) * np.maximum(cn64, 1e-8)),
                         hc / (np.maximum(hn64, 1e-8) * np.maximum(cn64, 1e-8)),
                         np.sqrt(np.maximum(y2 + 1 - 2 * yh / hn64, 0))), axis=-1)
    measured = gpu.cpu().numpy().astype(np.float64)
    assert np.allclose(measured, expected, rtol=1e-5, atol=1e-5)
    error = float(np.max(np.abs(measured - expected) / (1 + np.abs(expected))))
    return np.concatenate((basis, measured), axis=-1), error


def mathematical_check():
    rng = np.random.default_rng(42)
    h = torch.tensor(rng.normal(size=(4, 3, 3, 512)), dtype=torch.float32)
    h = F.normalize(h, dim=-1)
    c = torch.tensor(rng.normal(size=h.shape), dtype=torch.float32)
    c[0] = 0
    c[1] *= 1000
    c[2] = -.75 * h[2]
    c[3] *= .01
    bank = normalized_joint_bank({n: h[:, i] for i,n in enumerate(EXPERTS)},
                                 {n: c[:, i] for i,n in enumerate(EXPERTS)})
    y = F.normalize(bank.reshape_as(h), dim=-1)
    data, error = measure(h,c,y)
    assert np.all(data[0,...,1] == 0)
    assert np.max(np.abs(data[0,...,12] - 1)) < 1e-6
    assert np.max(np.abs(data[0,...,13])) == 0
    assert np.max(np.abs(data[2,...,11] - .75)) < 1e-6
    assert np.max(np.abs(data[2,...,12] - 1)) < 1e-6
    assert np.min(data[1,...,13]) > .999
    return {"status":"PASS_ZERO_LARGE_AND_OPPOSED_CORRECTION_GEOMETRY",
            "maximum_scaled_error":error,"real_model_forwards":0,"image_reads":0,"optimizer_updates":0}


def load_final(config, signal_cfg, fold, split, prior):
    model,binding = build_model(config,signal_cfg,fold,split,enabled=True)
    initial = _model_state_sha256(model)
    assert initial == prior["preflight"][fold]["endpoints"][1]["initial_state_sha256"]
    receipt = prior["folds"][fold]["endpoints"]["joint_tokens"]
    checkpoint = Path(receipt["checkpoint"])
    assert _sha256(checkpoint) == receipt["checkpoint_sha256"]
    saved = torch.load(checkpoint,map_location="cpu",weights_only=True)
    assert saved["binding"] == binding
    tensors = model.state_dict()
    assert set(saved["v28_state_dict"]) == {k for k in tensors if not k.startswith("baseline.")}
    tensors.update(saved["v28_state_dict"])
    model.load_state_dict(tensors,strict=True)
    del saved,tensors
    fixed = _model_state_sha256(model)
    assert fixed == receipt["training"]["final_state_sha256"]
    model.eval()
    model.baseline.train(True)
    assert not model.encoder.training and not model.baseline.signal.training and not model.fused_neck.training
    return model,{"binding":binding,"initial_state_sha256":initial,"fixed_state_sha256":fixed,
                  "checkpoint":str(checkpoint),"checkpoint_sha256":receipt["checkpoint_sha256"]}


def run(args):
    started=time.time()
    assert _sha256(args.contract)==args.contract_sha256
    contract=json.loads(args.contract.read_bytes())
    assert contract["views"]==list(VIEWS) and contract["fields"]==list(FIELDS)
    assert contract["planned_batches"]==1680 and contract["planned_model_forwards"]==3360
    assert contract["optimizer_updates"]==0 and contract["seed"]==42
    assert contract["checkpoint_endpoint"]=="joint_tokens"
    for source,expected in contract["source_file_sha256"].items():
        assert _sha256(Path(source))==expected,source
    assert _sha256(Path(__file__))==contract["runner_sha256"]
    assert shutil.disk_usage(args.output_dir.parent).free >= contract["minimum_free_bytes"]
    prior_dir=Path(contract["v28_run"])
    assert _sha256(prior_dir/"run_summary.json")==contract["v28_summary_sha256"]
    assert _sha256(prior_dir/"complete_terminal_verification.json")==contract["v28_verification_sha256"]
    terminal=json.loads(Path(str(prior_dir)+"_verification_waiter_exit.json").read_bytes())
    assert terminal["exit_code"]==0
    prior=json.loads((prior_dir/"run_summary.json").read_bytes())
    assert prior["status"]=="Q1_FAIL" and len(prior["folds"])==3
    assert _sha256(Path(contract["v28_config"]))==contract["v28_config_sha256"]
    config,sources=load_contract(Path(contract["v28_config"]))
    signal_cfg,signal_commit,signal_diff=_configure_signal(config)
    records=_load_records(config)
    splits=[build_complete_path_fold_records(records,heldout_ids=set(r["heldout_identity_ids"]))
            for r in sources["fold_receipts"]]
    replay=json.loads(Path(config["SUPERVISION_METADATA"]["PATH"]).read_bytes())
    torch.set_num_threads(4)
    math_check=mathematical_check()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    summary={"status":"RUNNING_FIXED_SOURCE_JOINT_SCALE_DIAGNOSTIC",
             "started_at":datetime.now().astimezone().isoformat(),
             "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
             "runner_sha256":_sha256(Path(__file__)),"contract_sha256":args.contract_sha256,
             "prior_summary_sha256":contract["v28_summary_sha256"],"math_check":math_check,
             "signal_commit":signal_commit,"signal_diff_sha256":signal_diff,
             "views":list(VIEWS),"fields":list(FIELDS),"experts":list(EXPERTS),"modalities":list(MODALITIES),
             "new_optimizer_updates":0,"encoder_gradients_computed":False,
             "model_mode":"fixed_eval_with_frozen_stem_training_flag",
             "dev_image_reads":0,"official_image_reads":0,"heldout_image_reads":0,
             "retrieval_evaluation":False,"model_selection":False,"folds":[],
             "completed_batches":0,"model_forwards":0,"slot_observations":0}
    summary_path=args.output_dir/"source_joint_scale.json"
    def save():
        summary["elapsed_seconds"]=time.time()-started
        summary_path.write_bytes((json.dumps(summary,indent=2)+chr(10)).encode())
    save()
    for fold,split in enumerate(splits):
        assert not split["identity_overlap"]
        actual=[{"file":Path(r[0][0]).name,"identity":r[1],"camera":r[2]} for r in split["train_records"]]
        assert actual==replay["folds"][fold]["source_manifest"]
        expected=replay["folds"][fold]["arms"]["control"]
        model,binding=load_final(config,signal_cfg,fold,split,prior)
        seed_everything()
        loader=training_loader(split["train_records"],config,"joint_tokens")
        index=_record_index_by_path(split["train_records"])
        count=expected["batch_count"]
        array_path=args.output_dir/f"fold_{fold}_slot_statistics.npy"
        arrays=np.lib.format.open_memmap(array_path,mode="w+",dtype=np.float64,
                                        shape=(count,2,64,3,3,len(FIELDS)))
        receipts_path=args.output_dir/f"fold_{fold}_batch_receipts.jsonl"
        exposure,steps=Counter(),0
        first_eight=[]
        maximum_error,maximum_bank_error=0.,0.
        captured={}
        def capture_joint(_module,_inputs,output):
            captured["corrections"]=output
        with receipts_path.open("x",encoding="utf-8") as receipts,torch.no_grad(),model.joint.register_forward_hook(capture_joint):
            for epoch in range(1,21):
                for within_epoch,raw in enumerate(loader,1):
                    wanted=expected["batches"][steps]
                    assert (wanted["epoch"],wanted["step"])==(epoch,within_epoch)
                    checked=check_batch(raw,index,wanted)
                    pixel_receipt=_raw_batch_receipt(raw,record_index_by_path=index)
                    if steps<8:
                        assert pixel_receipt==prior["folds"][fold]["endpoints"]["joint_tokens"]["training"]["first_eight_batch_receipts"][steps]
                        first_eight.append(True)
                    exposure.update(checked["sampler_indices"])
                    plan=make_style_plan(raw[2].numpy(),fold=fold,step=steps)
                    receipts.write(json.dumps({"fold":fold,"epoch":epoch,"step":within_epoch,"case_index":steps,
                                               **checked,"raw_receipt":pixel_receipt,"style_plan":plan})+chr(10))
                    batch,_labels=_training_batch(raw)
                    original_signal=None
                    original_stats=None
                    for view_index,view in enumerate(VIEWS):
                        model.baseline.style_enabled=view=="registered_style"
                        model.baseline.style_plan=plan
                        captured.clear()
                        with torch.autocast("cuda",dtype=torch.float16):
                            output=model(batch,return_aux=True)
                        assert output.diagnostics["all_finite"] and output.diagnostics["baseline_exact_prefix"]
                        h=torch.stack([output.modal_residual_embeddings[e] for e in EXPERTS],dim=1)
                        c=torch.stack([captured["corrections"][e] for e in EXPERTS],dim=1)
                        bank=output.fused_embedding[:,model.baseline_embedding_width:]
                        y=F.normalize(bank.reshape_as(h).float(),dim=-1)
                        reconstructed=normalized_joint_bank(
                            {e:h[:,i] for i,e in enumerate(EXPERTS)},
                            {e:c[:,i] for i,e in enumerate(EXPERTS)})
                        normalized_actual=F.normalize(bank.float(),dim=1)
                        bank_error=float((normalized_actual-reconstructed.float()).abs().max())
                        assert bank_error < 2e-6
                        maximum_bank_error=max(maximum_bank_error,bank_error)
                        data,error=measure(h,c,y)
                        assert np.max(np.abs(data[...,8]-1)) < 2e-5
                        maximum_error=max(maximum_error,error)
                        if original_signal is None: original_signal=output.baseline_embedding.clone()
                        assert torch.equal(output.baseline_embedding,original_signal)
                        if view_index==0: original_stats=data.copy()
                        elif not plan["active"]: assert np.array_equal(data,original_stats)
                        arrays[steps,view_index]=data
                        summary["model_forwards"]+=1
                        summary["slot_observations"]+=64*9
                        del output,h,c,y,bank,reconstructed,normalized_actual,data
                    steps+=1
                    summary["completed_batches"]+=1
                receipts.flush();arrays.flush();save()
                print(json.dumps({"fold":fold,"epoch":epoch,"completed_batches":summary["completed_batches"],
                                  "model_forwards":summary["model_forwards"],"elapsed_seconds":summary["elapsed_seconds"]}),flush=True)
        assert steps==count==(580,560,540)[fold] and len(first_eight)==8
        for identity in expected["identities"]:
            assert [exposure[i] for i in identity["record_indices"]]==identity["record_exposures"]
        assert _model_state_sha256(model)==binding["fixed_state_sha256"]
        assert all(p.grad is None for p in model.parameters())
        arrays.flush()
        del arrays,model
        torch.cuda.empty_cache()
        summary["folds"].append({"fold":fold,"source_records":len(actual),"source_manifest":actual,
            "binding":binding,"batches":steps,"model_forwards":steps*2,"slot_observations":steps*2*64*9,
            "maximum_cpu_gpu_scaled_error":maximum_error,"maximum_actual_bank_error":maximum_bank_error,
            "all_model_states_unchanged":True,"all_gradients_absent":True,
            "first_eight_augmentation_receipts_equal_original_v28":True,
            "all_sample_orders_and_record_exposures_equal_registered":True,
            "statistics":{"path":str(array_path),"bytes":array_path.stat().st_size,"sha256":_sha256(array_path)},
            "batch_receipts":{"path":str(receipts_path),"bytes":receipts_path.stat().st_size,"sha256":_sha256(receipts_path)}})
        save()
    assert summary["completed_batches"]==1680 and summary["model_forwards"]==3360
    assert summary["slot_observations"]==1935360 and len(summary["folds"])==3
    summary["status"]="COMPLETE_FIXED_SOURCE_JOINT_SCALE_DIAGNOSTIC_PENDING_CPU_VERIFICATION"
    summary["completed_at"]=datetime.now().astimezone().isoformat()
    save()
    print(json.dumps({"status":summary["status"],"model_forwards":3360,"slot_observations":1935360,"optimizer_updates":0}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract",type=Path)
    parser.add_argument("--contract-sha256")
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--math-only",action="store_true")
    args=parser.parse_args()
    if args.math_only: print(json.dumps(mathematical_check()))
    else: run(args)
