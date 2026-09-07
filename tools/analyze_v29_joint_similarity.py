#!/usr/bin/env python3
"""Descriptive, source-only analysis of saved V29 joint bank similarities."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from tools.v29_source_drift_math import indices

VIEWS = ("original", "registered_style")
STRATA = ("all", "active", "inactive")
KINDS = ("all_off_diagonal", "positive_same_camera", "positive_cross_camera",
         "negative_same_camera", "negative_cross_camera", "same_record",
         "positive_distinct_record", "all_distinct_record")
RF = ("count", "before_margin_sum", "after_margin_sum", "affine_residual_sum",
      "affine_residual_square_sum", "fixed_residual_square_sum",
      "margin_change_square_sum", "positive_residual_count", "negative_residual_count",
      "before_nonpositive_count", "after_nonpositive_count")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def moments(x, y):
    return np.array([x.size, x.sum(), y.sum(), (x*x).sum(), (x*y).sum(), (y*y).sum()])


def fit(m):
    n, sx, sy, xx, xy, yy = m
    if n == 0:
        return {k: None for k in ("x_mean", "y_mean", "slope", "intercept", "r2", "rmse")}
    vx, vy, cov = xx-sx*sx/n, yy-sy*sy/n, xy-sx*sy/n
    # Same-record or other empty/constant subsets can have no measurable variance.
    if vx <= 1e-12*n or vy <= 1e-12*n:
        return dict(x_mean=sx/n, y_mean=sy/n, slope=None, intercept=None, r2=None, rmse=None)
    a = cov/vx
    sse = vy-cov*cov/vx
    assert sse >= -1e-10*n
    return dict(x_mean=sx/n, y_mean=sy/n, slope=a, intercept=(sy-a*sx)/n,
                r2=1-max(0., sse)/vy, rmse=np.sqrt(max(0., sse)/n))


def pair_masks(labels, cameras, records):
    off = ~np.eye(len(labels), dtype=bool)
    identity = labels[:, None] == labels[None, :]
    camera = cameras[:, None] == cameras[None, :]
    record = records[:, None] == records[None, :]
    masks = (off, off & identity & camera, off & identity & ~camera,
             off & ~identity & camera, off & ~identity & ~camera,
             off & record, off & identity & ~record, off & ~record)
    assert sum(int(x.sum()) for x in masks[1:5]) == int(off.sum())
    assert np.all(~record | identity)
    return masks


def relation_sums(x, y, a):
    residual = y-a*x
    return np.array([x.size, x.sum(), y.sum(), residual.sum(), (residual**2).sum(),
                     ((y-.8*x)**2).sum(), ((y-x)**2).sum(), (residual>0).sum(),
                     (residual<0).sum(), (x<=0).sum(), (y<=0).sum()])


def relation_row(key, value, identity=False):
    names = ("fold", "view", "stratum", "protocol", "identity") if identity else (
        "fold", "view", "stratum", "protocol")
    row = dict(zip(names, key, strict=True))
    row.update(zip(RF, value, strict=True))
    n = int(value[0])
    row["count"] = n
    for name in RF[1:]:
        if name.endswith("_count"):
            row[name] = int(row[name])
        else:
            row[name.removesuffix("_sum")+"_mean"] = row[name]/n if n else None
    return row


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"rows": len(rows), "bytes": path.stat().st_size, "sha256": sha(path)}


def check_math():
    rng = np.random.default_rng(42)
    x = rng.uniform(-.5, .9, size=2016)
    y = .8*x+.2
    f = fit(moments(x, y))
    assert abs(f["slope"]-.8)<1e-12 and abs(f["intercept"]-.2)<1e-12
    assert abs(f["r2"]-1)<1e-12
    noisy = y + .01*np.sin(np.arange(len(x)))
    g = fit(moments(x, noisy))
    direct = np.linalg.lstsq(np.stack((x, np.ones_like(x)), axis=1), noisy, rcond=None)[0]
    assert np.allclose([g["slope"], g["intercept"]], direct, atol=1e-12, rtol=0)
    assert abs(g["rmse"]-np.sqrt(np.mean((noisy-direct[0]*x-direct[1])**2)))<1e-12
    labels = np.array([0, 0, 1, 1, 2, 2])
    cameras = np.array([0, 1, 0, 0, 1, 1])
    records = np.array([0, 1, 2, 2, 3, 4])
    masks = pair_masks(labels, cameras, records)
    assert [int(m.sum()) for m in masks] == [30, 4, 2, 8, 16, 2, 4, 28]
    matrix = rng.uniform(size=(6, 6))
    for protocol in ("identity", "cross_camera"):
        q,p,n = indices(labels, cameras, protocol)
        explicit = [(i,j,k) for i in range(6) for j in range(6) for k in range(6)
                    if i!=j and labels[i]==labels[j] and labels[i]!=labels[k]
                    and (protocol=="identity" or cameras[i]!=cameras[j])]
        assert list(zip(q,p,n)) == explicit
        mx = matrix[q,p]-matrix[q,n]
        my = (.8*matrix+.2)[q,p]-(.8*matrix+.2)[q,n]
        totals = relation_sums(mx,my,.8)
        explicit_margin = np.array([matrix[i,j]-matrix[i,k] for i,j,k in explicit])
        assert np.allclose(mx,explicit_margin,atol=0,rtol=0)
        assert abs(totals[1]-sum(explicit_margin)) < 1e-12
        assert abs(totals[2]-.8*sum(explicit_margin)) < 1e-12
        assert totals[4]<1e-25 and totals[5]<1e-25
        assert np.max(np.abs(my-.8*mx)) < 1e-15
    assert fit(np.zeros(6))["slope"] is None
    assert fit(moments(np.ones(3), np.ones(3)))["slope"] is None
    fields=rng.uniform(-1,1,size=(3,64,64)).astype(np.float32)
    legacy=(.5*fields[0]+.5*fields[1]).astype(np.float64)
    exact=.5*fields[0].astype(np.float64)+.5*fields[1].astype(np.float64)
    assert np.max(np.abs(legacy-exact))>0
    assert np.array_equal((fields[2].astype(np.float64)-exact)
                          -(fields[2].astype(np.float64)-legacy),legacy-exact)
    return {"status":"PASS_AFFINE_MATH_EXPLICIT_PAIRS_AND_RELATIONS", "seed":42}


def analyze(args):
    started = time.time()
    contract = json.loads(args.contract.read_bytes())
    assert sha(__file__) == contract["runner_sha256"]
    for path,expected in contract["dependency_sha256"].items():
        assert sha(path)==expected
    run = Path(contract["source_run"])
    assert sha(run/"source_role_drift.json") == contract["source_summary_sha256"]
    assert sha(run/"complete_source_drift_verification.json") == contract["source_verification_sha256"]
    summary = json.loads((run/"source_role_drift.json").read_bytes())
    proof = json.loads((run/"complete_source_drift_verification.json").read_bytes())
    assert proof["status"] == "PASS_COMPLETE_V29_SOURCE_ROLE_DRIFT_AND_RELATIONS"
    assert summary["completed_batches"] == 1680 and summary["model_forwards"] == 10080
    prior_groups = json.loads((run/"all_source_relation_groups.json").read_bytes())
    assert not args.output_dir.exists()
    args.output_dir.mkdir()
    math_check = check_math()
    pair_rows, batch_rows, relation_rows, identity_rows, hashes = [], [], [], [], []
    max_sse_error, max_previous_error, total_cases = 0., 0., 0
    maximum_rounding, maximum_exact_legacy_difference = 0., 0.
    for fold in summary["folds"]:
        number = fold["fold"]
        for field in ("similarities", "receipts"):
            item = fold[field]
            assert Path(item["path"]).stat().st_size == item["bytes"]
            assert sha(item["path"]) == item["sha256"]
            hashes.append(item)
        matrices = np.load(fold["similarities"]["path"], mmap_mode="r")
        receipts = [json.loads(s) for s in Path(fold["receipts"]["path"]).read_text().splitlines()]
        assert matrices.shape == (len(receipts),3,2,18,64,64)
        assert len(receipts) == (580,560,540)[number]
        manifest = fold["source_manifest"]
        ids = {r["identity"]: r["file"].split("_",1)[0] for r in manifest}
        assert len(ids) == 94
        pairs = {(vi,st,ki):np.zeros(6) for vi in range(2) for st in STRATA for ki in KINDS}
        theory_sse = {k:0. for k in pairs}
        cached = []
        for case, receipt in enumerate(receipts):
            assert receipt["case_index"] == case
            records = np.array(receipt["sampler_indices"])
            labels = np.array([manifest[i]["identity"] for i in records])
            cameras = np.array([manifest[i]["camera"] for i in records])
            active = "active" if receipt["style_plan"]["active"] else "inactive"
            masks = pair_masks(labels,cameras,records)
            cached.append((labels,cameras,active,masks))
            for vi,view in enumerate(VIEWS):
                z = matrices[case,2,vi].astype(np.float64)
                x,y = z[14],2*z[1]-z[0]
                assert np.isfinite(x).all() and np.isfinite(y).all()
                for ki,mask in zip(KINDS,masks,strict=True):
                    xx,yy = x[mask],y[mask]
                    mm = moments(xx,yy)
                    error = float(((yy-.8*xx-.2)**2).sum())
                    for st in ("all",active):
                        pairs[vi,st,ki] += mm
                        theory_sse[vi,st,ki] += error
                local = fit(moments(x[masks[0]],y[masks[0]]))
                batch_rows.append(dict(fold=number,case=case,view=view,stratum=active,**local))
        fits = {k:fit(m) for k,m in pairs.items()}
        direct_sse = {k:0. for k in pairs}
        relations = {(vi,st,p):np.zeros(len(RF)) for vi in range(2) for st in STRATA
                     for p in ("identity","cross_camera")}
        legacy_joint = {key:0. for key in relations}
        identities = {(*key,i):np.zeros(len(RF)) for key in relations for i in ids}
        for case,(labels,cameras,active,masks) in enumerate(cached):
            legal = {p:indices(labels,cameras,p) for p in ("identity","cross_camera")}
            for vi in range(2):
                z = matrices[case,2,vi].astype(np.float64)
                x,y = z[14],2*z[1]-z[0]
                legacy=(.5*matrices[case,2,vi,0]+.5*matrices[case,2,vi,14]).astype(np.float64)
                rounding=float(np.max(np.abs(legacy-(.5*z[0]+.5*z[14]))))
                maximum_rounding=max(maximum_rounding,rounding)
                assert rounding<=np.finfo(np.float32).eps
                for st in ("all",active):
                    for ki,mask in zip(KINDS,masks,strict=True):
                        f = fits[vi,st,ki]
                        if f["slope"] is not None:
                            direct_sse[vi,st,ki] += float(((y[mask]-f["slope"]*x[mask]-f["intercept"])**2).sum())
                    a = fits[vi,st,"all_off_diagonal"]["slope"]
                    assert a is not None
                    for protocol,(q,p,n) in legal.items():
                        mx,my = x[q,p]-x[q,n], y[q,p]-y[q,n]
                        relations[vi,st,protocol] += relation_sums(mx,my,a)
                        legacy_joint[vi,st,protocol] += float(((z[1,q,p]-z[1,q,n])
                                                               -(legacy[q,p]-legacy[q,n])).sum())
                        for identity in np.unique(labels):
                            mask = labels[q]==identity
                            identities[vi,st,protocol,int(identity)] += relation_sums(mx[mask],my[mask],a)
        for key,m in pairs.items():
            vi,st,ki = key
            f = fits[key]
            n = int(m[0])
            if f["rmse"] is not None:
                error = abs(direct_sse[key]-n*f["rmse"]**2)/n
                max_sse_error = max(error,max_sse_error)
                assert error<1e-10
            pair_rows.append(dict(fold=number,view=VIEWS[vi],stratum=st,pair_kind=ki,count=n,
                                  **f,fixed_08_02_rmse=np.sqrt(theory_sse[key]/n) if n else None,
                                  **dict(zip(("n","sum_x","sum_y","sum_x2","sum_xy","sum_y2"),m,strict=True))))
        for (vi,st,p),value in relations.items():
            rows = [identities[vi,st,p,i] for i in ids]
            assert np.allclose(sum(rows),value,atol=1e-6,rtol=1e-10)
            prior = [g for g in prior_groups if g["fold"]==number and g["protocol"]==p
                     and (st=="all" or g["stratum"]==st)]
            assert int(value[0]) == sum(g["triplets"] for g in prior)
            previous = sum(g["joint_changes"][vi][1]-g["joint_changes"][vi][0] for g in prior)
            error = abs(legacy_joint[vi,st,p]-previous)/value[0]
            max_previous_error = max(max_previous_error,error)
            assert error<1e-10
            difference=abs(.5*(value[2]-value[1])-previous)/value[0]
            maximum_exact_legacy_difference=max(maximum_exact_legacy_difference,difference)
            assert difference<=2*maximum_rounding+1e-10
            relation_rows.append(relation_row((number,VIEWS[vi],st,p),value))
        for (vi,st,p,i),value in identities.items():
            identity_rows.append(relation_row((number,VIEWS[vi],st,p,ids[i]),value,True))
        total_cases += len(receipts)
        print(json.dumps(dict(completed_fold=number,completed_cases=total_cases,elapsed_seconds=time.time()-started)),flush=True)
        del matrices,cached
    assert total_cases==1680
    assert (len(pair_rows),len(batch_rows),len(relation_rows),len(identity_rows))==(144,3360,36,3384)
    exports = {}
    for filename,rows in (("pair_affine.csv",pair_rows),("batch_affine.csv",batch_rows),
                          ("relation_residual.csv",relation_rows),("identity_relation_residual.csv",identity_rows)):
        exports[filename] = write_csv(args.output_dir/filename,rows)
    result = dict(status="PASS_COMPLETE_SOURCE_JOINT_SIMILARITY_ANALYSIS",completed_at=datetime.now().astimezone().isoformat(),
                  elapsed_seconds=time.time()-started,contract_sha256=sha(args.contract),runner_sha256=sha(__file__),
                  source_summary_sha256=contract["source_summary_sha256"],inputs=hashes,math_check=math_check,
                  checked_cases=total_cases,directed_off_diagonal_pairs_per_view=1680*64*63,
                  maximum_direct_sse_mean_error=max_sse_error,maximum_previous_joint_mean_error=max_previous_error,
                  maximum_fp32_decomposition_rounding=maximum_rounding,
                  maximum_exact_vs_legacy_joint_mean_difference=maximum_exact_legacy_difference,
                  model_forwards=0,optimizer_updates=0,image_reads=0,retrieval_evaluations=0,exports=exports,
                  interpretation="Descriptive fits only; no deployed score modification, parameter selection, independence or unseen-identity claim.")
    with (args.output_dir/"analysis.json").open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only",action="store_true")
    parser.add_argument("--contract",type=Path)
    parser.add_argument("--output-dir",type=Path)
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(check_math()))
    else:
        assert args.contract is not None and args.output_dir is not None
        analyze(args)
