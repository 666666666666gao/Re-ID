#!/usr/bin/env python3
"""Independent small-case checks for source drift and identity relation arithmetic."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from tools.v29_source_drift_math import (
    CHANGE_FIELDS, STATE_PAIRS, case_relations, validate_matrices, validate_vector_statistics,
)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def matrix_from_slots(base, slots, corrected):
    roles = [unit(slots[:, i:i+3].reshape(len(base), -1)) for i in (0, 3, 6)]
    bank = unit(np.concatenate(roles, axis=1))
    joint_bank = unit(corrected.reshape(len(base), -1))
    full = [unit(np.concatenate((base, role), axis=1)) for role in roles]
    fused = unit(np.concatenate((base, joint_bank), axis=1))
    features = [base, fused, *full, *(slots[:, i] for i in range(9)), bank, *roles]
    return np.stack([x @ x.T for x in features])


def scalar_changes(before, after, labels, cameras, protocol, identity=None):
    rows = []
    for q in range(len(labels)):
        for p in range(len(labels)):
            for n in range(len(labels)):
                if q == p or labels[q] != labels[p] or labels[q] == labels[n]:
                    continue
                if identity is not None and labels[q] != identity:
                    continue
                if protocol == "cross_camera" and cameras[q] == cameras[p]:
                    continue
                rows.append((q, p, n))
    result = np.zeros((2, 18, len(CHANGE_FIELDS)), dtype=np.float64)
    for view in range(2):
        for output in range(18):
            for q, p, n in rows:
                bp, bn = float(before[view,output,q,p]), float(before[view,output,q,n])
                ap, an = float(after[view,output,q,p]), float(after[view,output,q,n])
                bm, am = bp-bn, ap-an
                bh = max(0., math.sqrt(max(0., 2-2*bp))-math.sqrt(max(0., 2-2*bn))+.3)
                ah = max(0., math.sqrt(max(0., 2-2*ap))-math.sqrt(max(0., 2-2*an))+.3)
                result[view,output] += [bm,am,abs(am-bm),am<bm,bm>0 and am<=0,
                                      bm<=0 and am>0,bm<=0,am<=0,bh,ah,bh>0,ah>0]
    return result, len(rows)


def run():
    rng = np.random.default_rng(42)
    labels = np.array([0,0,1,1,2,2])
    cameras = np.array([0,1,0,0,0,1])
    base = np.eye(3)[labels]
    original = np.broadcast_to(base[:,None,:],(6,9,3)).copy()
    inputs = []
    for view in range(2):
        h = unit(original + rng.normal(size=original.shape)*(.01 if view == 0 else .04))
        damaged = h.copy()
        damaged[0] = h[2]
        y = damaged.copy()
        y[1] = h[4]
        inputs.append((h,damaged,y))
    matrices = np.stack([
        np.stack([matrix_from_slots(base, values[0], values[0]) for values in inputs]),
        np.stack([matrix_from_slots(base,-values[0],-values[0]) for values in inputs]),
        np.stack([matrix_from_slots(base,values[1],values[2]) for values in inputs]),
    ])
    assert np.array_equal(matrices[0],matrices[1])
    max_error = 0.
    for protocol, expected_count in (("identity",24),("cross_camera",16)):
        result = case_relations(matrices,labels,cameras,protocol)
        assert result["triplets"] == expected_count
        for i,(a,b) in enumerate(STATE_PAIRS):
            wanted,count = scalar_changes(matrices[a],matrices[b],labels,cameras,protocol)
            assert count == expected_count
            max_error = max(max_error,float(np.max(np.abs(wanted-result["changes"][i]))))
        assert result["changes"][0,...,2:6].max() == 0
        assert result["changes"][1,...,4].sum() > 0
        reconstructed = sum(np.asarray(r["changes"]) for r in result["relation_by_identity"])
        assert np.allclose(reconstructed,result["changes"],rtol=1e-12,atol=1e-12)
        for row in result["relation_by_identity"]:
            for i,(a,b) in enumerate(STATE_PAIRS):
                wanted,count = scalar_changes(matrices[a],matrices[b],labels,cameras,protocol,row["identity"])
                assert count == row["triplets"]
                max_error = max(max_error,float(np.max(np.abs(wanted-np.asarray(row["changes"])[i]))))
        joint_sum = sum(np.asarray(r["joint_changes"]) for r in result["relation_by_identity"])
        assert np.allclose(joint_sum,result["joint_changes"],rtol=1e-12,atol=1e-12)
        joint_stable_sum = sum(np.asarray(r["joint_stable"]) for r in result["relation_by_identity"])
        assert np.array_equal(joint_stable_sum,result["joint_stable"])
        for i in range(3):
            rows = [r for r in result["stable_by_identity"] if r["comparison"] == i]
            assert np.array_equal(sum(np.asarray(r["lost_per_view_output"]) for r in rows),
                                  result["stable"][i,...,1])
            assert np.array_equal(sum(np.asarray(r["hinge_violated_per_view_output"]) for r in rows),
                                  result["stable"][i,...,2])
    assert max_error < 1e-10
    labels64 = np.repeat(np.arange(8),8)
    base64 = np.eye(8)[labels64]
    h64 = unit(base64[:,None,:] + .01*rng.normal(size=(64,9,8)))
    mat64 = matrix_from_slots(base64,h64,h64).astype(np.float32)
    validate_matrices(np.broadcast_to(mat64,(3,2,18,64,64)),False)
    left = unit(rng.normal(size=(64,3,3,16)))
    right = -left
    values = np.stack(((left*left).sum(-1),(right*right).sum(-1),(left*right).sum(-1),
                      ((left-right)**2).sum(-1),np.full((64,3,3),-1.),np.full((64,3,3),2.)),axis=-1)
    vector_error = validate_vector_statistics(values)
    assert np.allclose(left.reshape(64,-1) @ left.reshape(64,-1).T,
                       right.reshape(64,-1) @ right.reshape(64,-1).T)
    return {"status":"PASS_SOURCE_ROLE_DRIFT_SCALAR_ENUMERATION_AND_ROTATION_COUNTEREXAMPLE",
            "checked_at":datetime.now().astimezone().isoformat(),
            "maximum_scalar_change_error":max_error,"maximum_vector_error":vector_error,
            "identity_triplets":24,"cross_camera_triplets":16,
            "orthogonal_sign_flip_vector_chord":2.,
            "orthogonal_sign_flip_pairwise_change":0.,
            "all_identity_sums_match":True,"legitimate_identity_zero_included":True,
            "no_cross_camera_positive_identity_zero_count_included":True,
            "numpy_version":np.__version__,"new_model_forwards":0,"image_reads":0,
            "optimizer_updates":0,"torch_imports":0,
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "math_module_sha256":hashlib.sha256(Path("tools/v29_source_drift_math.py").read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cuda",action="store_true")
    args = parser.parse_args()
    proof = run()
    if args.cuda:
        import torch
        from tools.diagnose_v29_source_role_drift import paired_vectors
        from trifusion.joint_geometry_v29 import bounded_slots
        torch.manual_seed(42)
        h = torch.nn.functional.normalize(torch.randn(64,3,3,512,device="cuda"),dim=-1)
        correction = torch.randn_like(h)*1000
        y, _update = bounded_slots(h,correction)
        maximum = 0.
        for right in (h,-h,y):
            values,error = paired_vectors(h,right)
            maximum = max(maximum,error)
            assert values.shape == (64,3,3,6)
        assert float(torch.nn.functional.cosine_similarity(h,y,dim=-1).min()) >= 1/math.sqrt(1.25)-2e-6
        proof["cuda_pair_measurement"] = {"status":"PASS","maximum_error":maximum,"slot_pairs":1728,
                                          "gpu":torch.cuda.get_device_name(),"torch_version":torch.__version__}
        proof["torch_imports"] = 1
    assert not args.output.exists()
    args.output.write_text(json.dumps(proof,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(proof))
