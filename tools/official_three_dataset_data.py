"""Dataset-specific record and loader bindings for fixed official training."""

from pathlib import Path

from tools.train_rgbnt100_signal_oof import loader_for as rgbnt100_loader


def records_for(protocol, split):
    root = Path(protocol["dataset_root"])
    name = protocol["dataset"]
    rows = protocol["records"][split]
    records = []
    for row in rows:
        paths = [str(root / path) for path in row["paths"]]
        image = paths[0] if name == "RGBNT100" else paths
        identity = row["label"] if split == "train" else row["identity"]
        records.append((image, identity, row["camera"], row["view"]))
    return records


def loader_for(protocol, records, *, training, method, seed=42):
    name = protocol["dataset"]
    if name == "RGBNT100":
        return rgbnt100_loader(records, training, seed=seed)
    if name == "MSVR310":
        from tools.train_msvr310_signal_oof import loader_for as existing_loader
        return existing_loader(records, training, seed=seed)
    assert name == "RGBNT201"
    if training:
        if method in ("V27", "PLAIN_V27"):
            from trifusion.aligned_data import build_cross_camera_train_loader
            factory = build_cross_camera_train_loader
        else:
            assert method in ("R2", "R2_TOP1", "R2_UNIFORM", "PLAIN_V8", "SIGNAL_V8")
            from trifusion.aligned_data import build_aligned_train_loader
            factory = build_aligned_train_loader
        return factory(records, batch_size=64, num_instances=8, num_workers=4, seed=seed)
    from tools.train_signal_preserving_v18 import loader_for as existing_loader
    return existing_loader(records, {"DATA": {"EVAL_BATCH_SIZE": 64, "NUM_WORKERS": 4}})
