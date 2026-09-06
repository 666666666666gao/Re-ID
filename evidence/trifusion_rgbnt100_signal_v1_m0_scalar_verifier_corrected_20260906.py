from collections import Counter
from pathlib import Path
from statistics import mean
import datetime,hashlib,json,math,time
ROOT=Path(r"C:/Users/gb/.trifusion_github_publish_22c3bee")
started=time.perf_counter()
base=ROOT/"evidence/rgbnt100_signal_v1_engineering"
summary=json.loads((base/"m0/summary.json").read_text())
protocol=json.loads((ROOT/"protocols/rgbnt100_train_oof_v1.json").read_text())
config=json.loads((ROOT/"configs/RGBNT100/Signal-source-oof-v1.json").read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert summary["status"]=="PASS_ENGINEERING_ONLY" and summary["mode"]=="preflight"
assert summary["optimizer_steps"]==24 and summary["heldout_image_forwards"]==0
assert summary["official_test_image_access"]==summary["fixed_rgbnt201_dev_image_access"]==summary["expert_training"]==0
assert summary["config_sha256"]==sha(ROOT/"configs/RGBNT100/Signal-source-oof-v1.json")
assert summary["protocol_receipt_sha256"]==sha(base/"t0.json")
max_loss=0.
max_mean=0.
all_exposures=[]
checks=[]
for actual,fold in zip(summary["folds"],protocol["folds"],strict=True):
    training=actual["training"]
    assert training["optimizer_steps"]==len(training["steps"])==8
    assert training["epochs"]==len(training["history"])==1
    assert training["trainable_tensors"]==training["gradient_tensors"]==195
    assert not training["trainable_without_gradient"] and training["overflow_events"]==0
    assert training["initial_state_sha256"]!=training["final_state_sha256"]
    assert training["frozen_token_selection_initial_sha256"]==training["frozen_token_selection_final_sha256"]
    assert actual["strict_reload_exact_feature_parity"] and actual["feature_width"]==3072
    assert actual["clean_source_feature_forwards"]==16 and actual["heldout_image_forwards"]==0
    exposures=[]
    cross=0
    positive=0
    for number,step in enumerate(training["steps"],1):
        assert step["epoch"]==1 and step["step"]==number
        indices=step["sampled_record_indices"]
        assert len(indices)==64 and set(indices)<=set(fold["source_record_indices"])
        rows=[protocol["records"][i] for i in indices]
        assert sorted(Counter(r["identity"] for r in rows).values())==[8]*8
        assert len(step["id_triplet_head_losses"])==4
        composed=sum(step["id_triplet_head_losses"])+.1*step["gram_loss"]+.1*step["patch_loss"]
        max_loss=max(max_loss,abs(composed-step["loss"]))
        assert math.isfinite(step["loss"]) and step["amp_scale_after"]>=step["amp_scale_before"]
        exposures.extend(indices)
        for i,a in enumerate(rows):
            for b in rows[i+1:]:
                if a["identity"]==b["identity"]:
                    positive+=1
                    cross+=int(a["camera"]!=b["camera"])
    max_mean=max(max_mean,abs(mean(s["loss"] for s in training["steps"])-training["history"][0]["mean_loss"]))
    initial_lrs=set()
    for group in training["optimizer_groups"]:
        expected=.0007*(2 if "bias" in group["name"] else 1)
        if "base" in group["name"] and "adapter" not in group["name"]:expected=.000005
        assert group["initial_lr"]==expected,group["name"]
        assert group["lr"]==.1*.0007
        initial_lrs.add(expected)
    clean_lrs=sorted(.1*.0007+(v-.1*.0007)/5 for v in initial_lrs)
    actual_lrs=training["history"][0]["learning_rates"]
    assert len(clean_lrs)==len(actual_lrs)
    factors=[a/b for a,b in zip(actual_lrs,clean_lrs,strict=True)]
    assert max(factors)-min(factors)<1e-14
    assert all(1-.67<x<1+.67 for x in factors)
    all_exposures.extend(exposures)
    checks.append({"fold":fold["fold"],"updates":8,"source_exposures":len(exposures),
        "source_identity_count_exposed":len({protocol["records"][i]["identity"] for i in exposures}),
        "source_record_count_exposed":len(set(exposures)),"same_identity_pairs":positive,"cross_camera_pairs":cross,
        "total_parameters":training["total_parameters"],"trainable_parameters":training["trainable_parameters"],
        "peak_allocated_mib":actual["peak_allocated_mib"],"training_8_steps_seconds":training["history"][0]["elapsed_seconds"],
        "mean_loss":training["history"][0]["mean_loss"],"initial_parameter_group_lrs":sorted(initial_lrs),
        "epoch1_actual_lrs":actual_lrs,"epoch1_common_noise_factor":factors[0]})
assert len(all_exposures)==1536
result={"verified_at":datetime.datetime.now().astimezone().isoformat(),"status":"PASS_ALL24_M0_SCALARS_AND_SOURCE_ACCOUNTING",
"summary_sha256":sha(base/"m0/summary.json"),"folds":checks,"optimizer_steps":24,"source_exposures":1536,
"clean_source_reload_feature_forwards":48,"max_loss_composition_difference":max_loss,
"max_epoch_mean_difference":max_mean,"lr_scope":"original optimizer group rules and common epoch1 noise scaling verified from JSON; no local Torch RNG execution",
"precision_scope":"Recorded float components regrouped with Python doubles; no new tolerance-based scientific gate",
"local_model_tensor_image_calls":0,"elapsed_seconds":time.perf_counter()-started}
p=ROOT/"evidence/trifusion_rgbnt100_signal_v1_m0_scalar_verification_20260906.json"
assert not p.exists()
p.write_bytes((json.dumps(result,indent=2)+"\n").encode())
print(json.dumps(result))
