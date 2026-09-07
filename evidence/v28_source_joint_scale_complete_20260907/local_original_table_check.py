from pathlib import Path
from collections import Counter, defaultdict
import argparse,csv,hashlib,json,math

EXPERTS=("cnn","transformer","mamba")
MODALITIES=("RGB","NI","TI")
VIEWS=("original","registered_style")
METRICS=("h_norm","c_norm","sum_norm","c_to_h_norm","cos_y_h","cos_y_c","cos_h_c","distance_y_h_unit")
COUNT_KEYS=("zero_c","zero_sum","ratio_gt1","ratio_gt10","ratio_gt100",
            "cos_y_h_negative","cos_y_h_lt_half","cos_y_c_ge_099","closer_to_c")
MEAN_KEYS=("correction_abs_mean",*(n+"_mean" for n in METRICS))


def main(directory):
    intake=json.loads((directory/"intake.json").read_bytes())
    assert intake["files_received"]==16 and intake["all_file_bytes_and_sha256_match"]
    assert intake["model_image_npy_or_raw_vector_files_transferred"]==0
    for name,row in intake["files"].items():
        raw=(directory/Path(name).name).read_bytes()
        assert len(raw)==row["bytes"] and hashlib.sha256(raw).hexdigest()==row["sha256"]
    summary_path=directory/"source_joint_scale.json"
    summary=json.loads(summary_path.read_bytes())
    proof_path=directory/"complete_source_scale_verification.json"
    proof=json.loads(proof_path.read_bytes())
    assert proof["status"]=="PASS_COMPLETE_SOURCE_JOINT_SCALE_ARRAYS_AND_GEOMETRY"
    assert proof["source_summary_sha256"]==hashlib.sha256(summary_path.read_bytes()).hexdigest()
    assert summary["execution_commit"]==proof["execution_commit"]=="4158c95ca639721e584d1e7271219f0f3b557cc3"
    assert summary["model_forwards"]==3360 and summary["slot_observations"]==proof["checked_slots"]==1935360
    assert proof["checked_scalar_values"]==30965760
    assert summary["new_optimizer_updates"]==0 and not summary["retrieval_evaluation"]
    cells,identities=proof["all_cells"],proof["all_identity_rows"]
    assert len(cells)==162 and len(identities)==5076
    for filename,rows,key in [("all_fold_view_role_modality_strata.csv",cells,"cells_csv_sha256"),
                              ("all_source_identity_role_modality.csv",identities,"identity_csv_sha256")]:
        path=directory/filename
        assert hashlib.sha256(path.read_bytes()).hexdigest()==proof[key]
        with path.open(encoding="utf-8",newline="") as handle: csv_rows=list(csv.DictReader(handle))
        assert len(csv_rows)==len(rows)
        for csv_row,row in zip(csv_rows,rows,strict=True):
            assert set(csv_row)==set(row)
            for name,value in row.items():
                if isinstance(value,str): assert csv_row[name]==value
                elif isinstance(value,int): assert int(csv_row[name])==value
                else: assert math.isfinite(value) and float(csv_row[name])==value
    by_cell={}
    by_identity=defaultdict(list)
    for row in cells:
        key=(row["fold"],row["view"],row["role"],row["modality"],row["stratum"])
        assert key not in by_cell
        by_cell[key]=row
    for row in identities:
        by_identity[(row["fold"],row["view"],row["role"],row["modality"])].append(row)
    maximum_error=0.
    maximum_scaled_error=0.
    def close(actual,expected):
        nonlocal maximum_error,maximum_scaled_error
        error=abs(actual-expected)
        maximum_error=max(maximum_error,error)
        maximum_scaled_error=max(maximum_scaled_error,error/(1+abs(expected)))
        assert math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-10),(actual,expected)
    def combine(target,parts):
        total=sum(p["observations"] for p in parts)
        assert total==target["observations"]
        for key in COUNT_KEYS: assert sum(p[key] for p in parts)==target[key]
        for key in MEAN_KEYS:
            close(sum(p[key]*p["observations"] for p in parts)/total,target[key])
        for name in METRICS:
            assert min(p[name+"_min"] for p in parts)==target[name+"_min"]
            assert max(p[name+"_max"] for p in parts)==target[name+"_max"]
    for row in [*cells,*identities]:
        n=row["observations"];assert n>0
        assert 0<=row["ratio_gt100"]<=row["ratio_gt10"]<=row["ratio_gt1"]<=n
        assert all(0<=row[key]<=n for key in COUNT_KEYS)
        for name in METRICS:
            assert row[name+"_min"]<=row[name+"_p05"]<=row[name+"_median"]<=row[name+"_p95"]<=row[name+"_max"]
            assert row[name+"_min"]-1e-10<=row[name+"_mean"]<=row[name+"_max"]+1e-10
    coverage=[]
    total_batches=total_active=0
    for fold in summary["folds"]:
        number=fold["fold"]
        manifest=fold["source_manifest"]
        assert len(manifest)==(2126,2075,2051)[number]
        original_ids={}
        for row in manifest:
            original_ids.setdefault(row["identity"],set()).add(row["file"].split("_",1)[0])
        assert len(original_ids)==94 and all(len(v)==1 for v in original_ids.values())
        receipt_path=directory/f"fold_{number}_batch_receipts.jsonl"
        assert hashlib.sha256(receipt_path.read_bytes()).hexdigest()==fold["batch_receipts"]["sha256"]
        batches=[json.loads(line) for line in receipt_path.read_bytes().splitlines()]
        assert len(batches)==(580,560,540)[number]
        exposures=Counter()
        active=0
        for index,row in enumerate(batches):
            assert row["fold"]==number and row["case_index"]==index
            ids=row["sampler_indices"]
            assert len(ids)==64 and all(0<=i<len(manifest) for i in ids)
            assert sorted(Counter(manifest[i]["identity"] for i in ids).values())==[8]*8
            assert hashlib.sha256(json.dumps(row["paths"],separators=(",",":")).encode()).hexdigest()==row["sample_order_sha256"]
            assert row["raw_receipt"]["paths"]==row["paths"] and row["raw_receipt"]["sampler_indices"]==ids
            cameras=[manifest[i]["camera"] for i in ids]
            assert all(cameras[j]!=cameras[i] for i,j in enumerate(row["style_plan"]["donors"]))
            active+=int(row["style_plan"]["active"])
            exposures.update(ids)
        assert len(exposures)==len(manifest) and min(exposures.values())>0
        for view in VIEWS:
            for expert in EXPERTS:
                for modality in MODALITIES:
                    key=(number,view,expert,modality)
                    all_row=by_cell[(*key,"all")]
                    parts=[by_cell[(*key,s)] for s in ("plan_active","plan_inactive")]
                    assert all_row["observations"]==len(batches)*64
                    assert parts[0]["observations"]==active*64 and parts[1]["observations"]==(len(batches)-active)*64
                    combine(all_row,parts)
                    people=by_identity[key]
                    assert len(people)==94 and {p["encoded_identity"] for p in people}==set(original_ids)
                    assert all(p["original_identity"] in original_ids[p["encoded_identity"]] for p in people)
                    combine(all_row,people)
        for expert in EXPERTS:
            for modality in MODALITIES:
                a=by_cell[(number,"original",expert,modality,"plan_inactive")]
                b=by_cell[(number,"registered_style",expert,modality,"plan_inactive")]
                assert {k:v for k,v in a.items() if k!="view"}=={k:v for k,v in b.items() if k!="view"}
        coverage.append({"fold":number,"source_records":len(manifest),"records_with_exposure":len(exposures),
                         "source_identities":94,"batch_count":len(batches),"active_batches":active,
                         "record_exposures":sum(exposures.values())})
        total_batches+=len(batches);total_active+=active
    assert total_batches==1680 and total_active==819
    for view in VIEWS:
        rows=[r for r in cells if r["view"]==view and r["stratum"]=="all"]
        overall=proof["overall_by_view"][view]
        total=sum(r["observations"] for r in rows)
        assert total==overall["observations"]==967680
        for key in COUNT_KEYS: assert sum(r[key] for r in rows)==overall[key]
        for key in MEAN_KEYS: close(sum(r[key]*r["observations"] for r in rows)/total,overall[key])
    result={"status":"PASS_LOCAL_COMPLETE_SOURCE_SCALE_CSV_COVERAGE_AND_AGGREGATION",
            "checked_cells":162,"checked_identity_rows":5076,"checked_batches":1680,
            "covered_slot_observations":1935360,"source_record_coverage":coverage,
            "all_csv_values_equal_json":True,"all_identity_and_stratum_aggregations_match":True,
            "all_inactive_view_strata_equal":True,"maximum_recomposition_error":maximum_error,
            "maximum_scaled_recomposition_error":maximum_scaled_error,
            "source_verification_sha256":hashlib.sha256(proof_path.read_bytes()).hexdigest(),
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "local_torch_model_image_or_npy_calls":0,"new_optimizer_updates":0,
            "boundary":"This local check validates all saved tables, source coverage and recomposition. Raw-vector and all-NPY-scalar checks ran on the server; this is not an external independent audit."}
    target=directory/"local_complete_source_scale_aggregation.json"
    assert not target.exists()
    target.write_bytes((json.dumps(result,indent=2)+chr(10)).encode())
    print(json.dumps(result))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--directory",type=Path,required=True)
    main(parser.parse_args().directory)
