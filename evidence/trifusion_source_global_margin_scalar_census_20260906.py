"""Derive fixed unit-L2 source margins from the already archived full cosine census."""
from collections import defaultdict
from pathlib import Path
from statistics import mean
import datetime
import hashlib
import json
import math
import time

ROOT=Path(r"C:/Users/gb/.trifusion_github_publish_22c3bee")
started=time.perf_counter()
path=ROOT/"evidence/trifusion_v24_source_diagnostic_20260906.json"
source_sha=hashlib.sha256(path.read_bytes()).hexdigest()
assert source_sha=="a66f17a450fb0eca2404fd23721545eed0dd7b061550230a7d51368da24fa271"
data=json.loads(path.read_text())
models=[]
all_rows=[]
for fold in data["folds"]:
    source=fold["source_manifest"]
    for name,model in fold["models"].items():
        geometry=model["sample_geometry"]
        columns=geometry["per_record"]
        positive=columns["hardest_positive_cosine"]
        negative=columns["nearest_negative_cosine"]
        rows=[]
        per_identity=defaultdict(list)
        for i,(p,n,record) in enumerate(zip(positive,negative,source,strict=True)):
            assert -1<p<1 and -1<n<1 and p>n
            hp=geometry["hardest_positive_record_index"][i]
            hn=geometry["nearest_negative_record_index"][i]
            assert hp!=i and source[hp]["source_label"]==record["source_label"]
            assert source[hn]["source_label"]!=record["source_label"]
            pd,nd=math.sqrt(2-2*p),math.sqrt(2-2*n)
            gap=nd-pd
            hinge=max(0.,.3-gap)
            row={"fold":fold["fold"],"model":name,"source_record_position":i,**record,
                 "hardest_positive_cosine":p,"nearest_negative_cosine":n,
                 "derived_positive_unit_distance":pd,"derived_negative_unit_distance":nd,
                 "derived_distance_gap":gap,"margin_0_3_hinge":hinge,"active":hinge>0,
                 "hardest_positive_cross_camera":source[hp]["camera"]!=record["camera"],
                 "nearest_negative_same_camera":source[hn]["camera"]==record["camera"]}
            rows.append(row)
            per_identity[record["source_label"]].append(row)
        active=[r for r in rows if r["active"]]
        id_rows=[]
        for label,items in sorted(per_identity.items()):
            id_rows.append({"source_label":label,"fit_registry_identity":items[0]["fit_registry_identity"],
                "records":len(items),"active_records":sum(r["active"] for r in items),
                "mean_hinge":mean(r["margin_0_3_hinge"] for r in items),
                "minimum_distance_gap":min(r["derived_distance_gap"] for r in items)})
        models.append({"fold":fold["fold"],"model":name,"records":len(rows),"identities":len(id_rows),
            "active_records":len(active),"active_fraction":len(active)/len(rows),
            "active_identities":sum(r["active_records"]>0 for r in id_rows),
            "mean_full_source_hinge":mean(r["margin_0_3_hinge"] for r in rows),
            "minimum_distance_gap":min(r["derived_distance_gap"] for r in rows),
            "minimum_abs_distance_to_margin_threshold":min(abs(r["derived_distance_gap"]-.3) for r in rows),
            "active_hardest_positive_cross_camera":sum(r["hardest_positive_cross_camera"] for r in active),
            "active_nearest_negative_same_camera":sum(r["nearest_negative_same_camera"] for r in active),
            "identity_rows":id_rows})
        all_rows.extend(rows)
assert len(models)==9 and len(all_rows)==18756
aggregate=[]
for name in data["folds"][0]["models"]:
    selected=[r for r in all_rows if r["model"]==name]
    aggregate.append({"model":name,"records":len(selected),"active_records":sum(r["active"] for r in selected),
        "active_fraction":mean(r["active"] for r in selected),"mean_full_source_hinge":mean(r["margin_0_3_hinge"] for r in selected)})
report={"status":"COMPLETE_POSTHOC_SCALAR_DERIVATION","recorded_at":datetime.datetime.now().astimezone().isoformat(),
"input_sha256":source_sha,"models":models,"aggregate":aggregate,"per_record":all_rows,
"source_models":9,"source_record_model_rows":18756,"fixed_margin":.3,
"formula":"d=sqrt(2-2cos), hinge=max(0,0.3+d_hardest_positive-d_nearest_negative)",
"limits":["Derived from saved FP32 cosines, not bitwise cdist recomputation",
"Full clean source positive AND negative extrema, not actual augmented batches",
"Does not isolate negative versus positive coverage, prove FIFO slow drift, change V24 failure, or supply unknown-identity results"],
"local_model_tensor_image_calls":0,"new_training_or_retrieval":0,"elapsed_seconds":time.perf_counter()-started}
out=ROOT/"evidence/trifusion_source_global_margin_scalar_census_20260906.json"
assert not out.exists()
out.write_bytes((json.dumps(report,indent=2)+"\n").encode())
print(json.dumps({k:v for k,v in report.items() if k not in ("per_record","models")}))
print(json.dumps([{k:v for k,v in m.items() if k!="identity_rows"} for m in models]))
